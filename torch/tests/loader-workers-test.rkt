#lang racket/base

(module+ test
  (require (only-in racket/list append* index-of range remove-duplicates)
           (only-in racket/match match match-define)
           rackunit
           "../data/loader.rkt"
           (only-in (submod "../data/loader.rkt" private) sampler-draws traversal)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           "../main.rkt"
           (only-in "../vision/transforms.rkt" random-crop random-horizontal-flip)
           (only-in "private/race-harness.rkt"
                    check-known-failure check-ledger ledger-snapshot make-watch
                    settle! watch-hook with-race-hook)
           (only-in "private/wedge.rkt" shut-down-custodian! with-process-deadline))

  (with-process-deadline 540
    (define blamed #rx"blaming: [(][^)]*loader-workers-test[.]rkt")

    (define ((contract-violation? rx) e)
      (and (exn:fail:contract? e)
           (regexp-match? blamed (exn-message e))
           (regexp-match? rx (exn-message e))))

    (define (record . vs)
      (for/list ([v (in-list vs)])
        (if (tensor? v) (list (tensor->list v) (dtype v) (shape v)) v)))

    (define (batches seq)
      (for/list ([vs (in-values-sequence seq)]) (apply record vs)))

    (define (within seconds thunk)
      (define outcome #f)
      (define t
        (thread (lambda ()
                  (set! outcome
                        (with-handlers ([(lambda (_v) #t)
                                         (lambda (v) (lambda () (raise v)))])
                          (call-with-values thunk
                                            (lambda vs (lambda () (apply values vs)))))))))
      (cond
        [(sync/timeout seconds t) (outcome)]
        [else
         (kill-thread t)
         (fail (format "no result within ~a s" seconds))]))

    (define (workers-of c)
      (filter thread? (custodian-managed-list c (current-custodian))))

    (define (live c)
      (filter (lambda (t) (not (thread-dead? t))) (workers-of c)))

    (define (dead-within? threads seconds)
      (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
      (for/and ([t (in-list threads)])
        (define left (max 0 (/ (- deadline (current-inexact-milliseconds)) 1000)))
        (and (sync/timeout left (thread-dead-evt t)) #t)))

    (define (dead-after-collections? threads)
      (for/or ([_ (in-range 20)])
        (collect-garbage 'major)
        (settle!)
        (dead-within? threads 0.25)))

    (define (under c thunk)
      (parameterize ([current-custodian c]) (thunk)))

    (define (keyword-pairs options)
      (match options
        ['() '()]
        [(list* k v more) (cons (cons k v) (keyword-pairs more))]))

    (define (loader-of ds workers prefetch . options)
      (define pairs
        (sort (keyword-pairs (if (zero? workers)
                                 options
                                 (list* '#:workers workers '#:prefetch prefetch options)))
              keyword<? #:key car))
      (keyword-apply dataloader (map car pairs) (map cdr pairs) (list ds)))

    (define (first-label items)
      (match-define (cons (list _ y) _) items)
      (item y))

    (define (workers-seen-by c loop!)
      (define seen '())
      (define outer (current-custodian))
      (define (see!)
        (define now (under outer (lambda () (live c))))
        (set! seen (remove-duplicates (append seen now) eq?)))
      (within 60 (lambda () (under c (lambda () (loop! see!)))))
      seen)

    (define xs (reshape (arange 20) 10 2))
    (define ys (arange 10 #:dtype 'int64))
    (define pairs (tensor-dataset xs ys))

    (define-dataset by-ref (xs ys)
      #:length (car (shape xs))
      #:ref (i) (values (select xs 0 i) (select ys 0 i)))

    (define (doubling items)
      (~> items (map car _) stack (* 2)))

    (define (traversal-outcome ds workers prefetch b shuffle? drop-last? explicit? collate)
      (manual-seed! 17)
      (define g (and explicit? (make-generator 17)))
      (define loader
        (loader-of ds workers prefetch '#:batch-size b '#:shuffle? shuffle?
                   '#:drop-last? drop-last? '#:generator g '#:collate collate))
      (within 60
              (lambda ()
                (list (batches loader)
                      (and g (tensor->list (randperm 10 #:generator g)))
                      (tensor->list (randn 3))))))

    (define (check-parity ds collate)
      (for* ([shuffle? '(#f #t)] [drop-last? '(#f #t)] [b '(3 5)] [explicit? '(#t #f)])
        (define reference (traversal-outcome ds 0 #f b shuffle? drop-last? explicit? collate))
        (for* ([n '(1 2 4)] [p '(1 2 3)])
          (check-equal? (traversal-outcome ds n p b shuffle? drop-last? explicit? collate)
                        reference
                        (format "workers ~a prefetch ~a batch ~a shuffle ~a drop ~a generator ~a"
                                n p b shuffle? drop-last? explicit?)))))

    (test-case "#:workers and #:prefetch violations blame the caller"
      (check-exn (contract-violation? #rx"workers") (lambda () (dataloader pairs #:workers -1)))
      (check-exn (contract-violation? #rx"workers") (lambda () (dataloader pairs #:workers 1.5)))
      (check-exn (contract-violation? #rx"prefetch")
                 (lambda () (dataloader pairs #:workers 1 #:prefetch 0))))

    (define-dataset placed (dev)
      #:length 4
      #:ref (i) i
      #:device dev
      #:batch (indices collate)
      (collate (for/list ([i (in-list (indices->list indices))]) (list i))))

    (test-case "workers are refused for a dataset on an accelerator"
      (for ([dev (list (cuda-device 0) (mps-device))])
        (check-exn (contract-violation? #rx"workers build CPU tensors")
                   (lambda () (dataloader (placed dev) #:workers 1)))
        (check-true (dataloader? (dataloader (placed dev) #:workers 0)))))

    (test-case "#:prefetch needs workers"
      (check-exn (contract-violation? #rx"needs #:workers above 0")
                 (lambda () (dataloader pairs #:prefetch 3)))
      (check-exn (contract-violation? #rx"needs #:workers above 0")
                 (lambda () (dataloader pairs #:workers 0 #:prefetch 3))))

    (test-case "workers 0 starts no threads; a traversal starts N under its custodian"
      (define c (make-custodian))
      (under c (lambda () (for ([(_x _y) (dataloader pairs #:batch-size 2)]) (void))))
      (check-equal? (workers-of c) '())
      (define loader (under c (lambda () (dataloader pairs #:batch-size 2 #:workers 3))))
      (check-equal? (workers-of c) '())
      (define-values (more? next) (under c (lambda () (sequence-generate loader))))
      (check-equal? (length (live c)) 3)
      (void more? next)
      (shut-down-custodian! c))

    (test-case "worker k builds the batches j with j mod N = k"
      (define builders (make-hasheqv))
      (define lock (make-semaphore 1))
      (define (noting items)
        (define j (quotient (first-label items) 2))
        (call-with-semaphore lock (lambda () (hash-set! builders j (current-thread))))
        (default-collate items))
      (define n 3)
      (within 60 (lambda ()
                   (for ([(_x _y) (dataloader pairs #:batch-size 2 #:workers n #:collate noting)])
                     (void))))
      (check-equal? (hash-count builders) 5)
      (check-equal? (length (remove-duplicates (hash-values builders) eq?)) n)
      (for* ([j (in-range 5)] [j2 (in-range 5)])
        (check-equal? (eq? (hash-ref builders j) (hash-ref builders j2))
                      (= (modulo j n) (modulo j2 n)))))

    (test-case "workers 0 and N agree on a tensor-dataset"
      (check-parity pairs default-collate))

    (test-case "workers 0 and N agree through dataset-ref and collate"
      (check-parity (by-ref xs ys) default-collate)
      (check-parity pairs doubling))

    (test-case "in-epochs and the loader as a sequence agree with workers 0"
      (define (run workers)
        (define g (make-generator 4))
        (define loader (loader-of pairs workers 2 '#:batch-size 3 '#:shuffle? #t
                                  '#:generator g))
        (within 60 (lambda ()
                     (list (batches (in-epochs loader 3))
                           (for/list ([(x y) loader]) (record x y))
                           (tensor->list (randperm 10 #:generator g))))))
      (define reference (run 0))
      (for ([n '(1 2 3)])
        (check-equal? (run n) reference (format "workers ~a" n))))

    (define-dataset slow-first (n)
      #:length n
      #:ref (i)
      (~> (* 10 i) (- 50 _) (/ 1000) sleep)
      (tensor (list i) #:dtype 'int64))

    (test-case "batches arrive in order when later ones finish first"
      (check-equal? (within 60 (lambda ()
                                 (for/list ([b (dataloader (slow-first 5) #:workers 5
                                                           #:prefetch 1)])
                                   (tensor->list b))))
                    '((0) (1) (2) (3) (4))))

    (define-dataset slow-batches (n)
      #:length n
      #:ref (i) i
      #:batch (indices collate)
      (sleep 0.4)
      (collate (for/list ([i (in-list (indices->list indices))]) (list i))))

    (test-case "a finished batch does not wait behind the worker's next build"
      (define waits
        (within 60
                (lambda ()
                  (define since (current-inexact-milliseconds))
                  (for/list ([_ (dataloader (slow-batches 6) #:workers 1 #:prefetch 2
                                            #:collate length)])
                    (define wait (- (current-inexact-milliseconds) since))
                    (sleep 0.6)
                    (set! since (current-inexact-milliseconds))
                    wait))))
      (match-define (cons _ later) waits)
      (check-equal? (length later) 5)
      (for ([wait (in-list later)])
        (check-true (< wait 100) (format "waited ~a ms for a finished batch" wait))))

    (define-dataset ticking (n)
      #:length n
      #:ref (i)
      (sleep (* 0.001 (modulo i 3)))
      (tensor (list i) #:dtype 'int64))

    (test-case "breaks while waiting lose no batch"
      (define-values (more? next)
        (~> (ticking 60) (dataloader #:workers 2) sequence-generate))
      (define got '())
      (define breaks 0)
      (define (pull!)
        (match (with-handlers ([exn:break? (lambda (_e) (set! breaks (add1 breaks)) 'broken)])
                 (parameterize-break #t (more?)))
          ['broken 'broken]
          [#t (set! got (cons (item (next)) got)) 'pulled]
          [#f 'done]))
      (define consumer
        (thread (lambda ()
                  (parameterize-break #f
                    (let loop ()
                      (match (pull!)
                        ['done 'done]
                        [_ (loop)]))))))
      (define breaker
        (thread (lambda ()
                  (let loop ()
                    (sleep)
                    (break-thread consumer)
                    (loop)))))
      (define finished? (sync/timeout 30 consumer))
      (kill-thread breaker)
      (kill-thread consumer)
      (check-not-false finished? "the traversal finished")
      (check-true (> breaks 10) (format "the consumer was broken ~a times" breaks))
      (check-equal? (reverse got) (range 60)))

    (test-case "at most workers × prefetch batches are requested ahead"
      (define trace '())
      (define lock (make-semaphore 1))
      (define (note! e) (call-with-semaphore lock (lambda () (set! trace (cons e trace)))))
      (define (starting items)
        (note! (list 'start (first-label items)))
        (default-collate items))
      (define-values (n p) (values 2 2))
      (define ahead (* n p))
      (within 60 (lambda ()
                   (for ([(_x y) (dataloader pairs #:workers n #:prefetch p #:collate starting)])
                     (sleep 0.02)
                     (note! (list 'done (item y))))))
      (define events (reverse trace))
      (define (at e) (index-of events e))
      (for ([j (in-range 10)])
        (check-equal? (length (filter (lambda (e) (equal? e (list 'start j))) events)) 1
                      (format "batch ~a starts once" j)))
      (check-false (findf (lambda (e) (match e [(list 'start j) (>= j 10)] [_ #f])) events))
      (for ([j (in-range (add1 ahead) 10)])
        (check-true (> (at (list 'start j)) (at (list 'done (- j ahead 1))))
                    (format "batch ~a starts after batch ~a is done" j (- j ahead 1)))))

    (define (replay seed n r*-drawn?)
      (define g (make-generator seed))
      (draw-seed #:generator g)
      (randperm n #:generator g)
      (when r*-drawn? (randperm n #:generator g))
      (tensor->list (randperm n #:generator g)))

    (test-case "the permutation is drawn when the traversal starts"
      (define g (make-generator 9))
      (define c (make-custodian))
      (under c (lambda ()
                 (for ([_ (in-range 0)]
                       [(_x _y) (dataloader pairs #:batch-size 2 #:shuffle? #t #:generator g
                                       #:workers 1 #:prefetch 1)])
                   (void))))
      (check-equal? (tensor->list (randperm 10 #:generator g)) (replay 9 10 #f))
      (shut-down-custodian! c))

    (test-case "the trailing permutation is drawn at its request"
      (for* ([np '((1 1) (2 1) (2 2))] [b '(4 5)] [d '(0 1 all)])
        (match-define (list workers prefetch) np)
        (define g (make-generator 9))
        (define ahead (* workers prefetch))
        (define count (quotient (+ 10 b -1) b))
        (define taken (if (eq? d 'all) count d))
        (define r* (if (positive? (remainder 10 b)) (sub1 count) count))
        (define c (make-custodian))
        (within 60 (lambda ()
                     (under c (lambda ()
                                (for ([_ (in-range taken)]
                                      [(_x _y) (dataloader pairs #:batch-size b #:shuffle? #t
                                                      #:generator g #:workers workers
                                                      #:prefetch prefetch)])
                                  (void))))))
        (check-equal? (tensor->list (randperm 10 #:generator g))
                      (replay 9 10 (< r* (+ ahead taken)))
                      (format "P' ~a, batch ~a, taken ~a" ahead b taken))
        (shut-down-custodian! c)))

    (test-case "in-epochs prefetches nothing across epochs"
      (define g (make-generator 9))
      (define threads
        (workers-seen-by (make-custodian)
                         (lambda (see!)
                           (for ([_ (in-range 3)]
                                 [(_e _x _y) (in-epochs (dataloader pairs #:batch-size 4
                                                                    #:shuffle? #t
                                                                    #:generator g
                                                                    #:workers 2)
                                                        3)])
                             (see!)))))
      (check-equal? (tensor->list (randperm 10 #:generator g)) (replay 9 10 #t))
      (check-equal? (length threads) 2)
      (check-true (dead-within? threads 5)))

    (test-case "an empty traversal draws as workers 0 does and stops its workers"
      (define small (~> (arange 6) (reshape 3 2) tensor-dataset))
      (define (run workers)
        (define g (make-generator 9))
        (define loader (loader-of small workers 2 '#:batch-size 4 '#:shuffle? #t
                                  '#:drop-last? #t '#:generator g))
        (list (within 60 (lambda () (batches loader)))
              (tensor->list (randperm 3 #:generator g))))
      (match-define (and outcome (list traversed _)) (run 2))
      (check-equal? outcome (run 0))
      (check-equal? traversed '())
      (define c (make-custodian))
      (define-values (more? next)
        (under c (lambda ()
                   (sequence-generate (dataloader small #:batch-size 4 #:drop-last? #t
                                                  #:workers 2)))))
      (define threads (live c))
      (check-equal? (length threads) 2)
      (check-false (more?))
      (check-true (dead-within? threads 5))
      (void next)
      (shut-down-custodian! c))

    (define (seeded-prng seed)
      (define prng (make-pseudo-random-generator))
      (parameterize ([current-pseudo-random-generator prng])
        (random-seed (bitwise-and seed #x7FFFFFFF)))
      prng)

    (define-dataset noisy (n jitter?)
      #:length n
      #:ref (i)
      (when jitter? (sleep (* 0.003 (modulo (* 7 i) 5))))
      (define image (full i 1 3 6 6))
      (define augmented
        (random-horizontal-flip (random-crop image #:padding 2)))
      (+ (+ (select augmented 0 0) (* (randn 3 6 6) 0.01)) (random)))

    (define (noisy-collate items)
      (values (stack (map car items)) (randperm 4)))

    (define (noisy-run workers prefetch #:jitter? [jitter? #f])
      (define loader
        (dataloader (noisy 12 jitter?) #:batch-size 5 #:shuffle? #t
                    #:generator (make-generator 7) #:collate noisy-collate
                    #:workers workers #:prefetch prefetch))
      (within 60 (lambda () (batches loader))))

    (test-case "a seeded worker run with random transforms is reproducible"
      (for ([n '(2 3)])
        (check-equal? (noisy-run n 2) (noisy-run n 2) (format "workers ~a" n))))

    (test-case "worker k draws from make-generator (+ base k)"
      (for ([n '(2 3)])
        (define g (make-generator 7))
        (define base (draw-seed #:generator g))
        (define perm (randperm 12 #:generator g))
        (define gens (make-hasheqv))
        (define prngs (make-hasheqv))
        (define replayed
          (for/list ([j (in-range 3)])
            (define k (modulo j n))
            (parameterize ([current-generator
                            (hash-ref! gens k (lambda () (make-generator (+ base k))))]
                           [current-pseudo-random-generator
                            (hash-ref! prngs k (lambda () (seeded-prng (+ base k))))])
              (call-with-values
               (lambda ()
                 (dataset-batch (noisy 12 #f) (narrow perm 0 (* j 5) (min 5 (- 12 (* j 5))))
                                noisy-collate))
               record))))
        (check-equal? (noisy-run n 2) replayed (format "workers ~a" n))))

    (test-case "batches do not depend on prefetch or scheduling"
      (define reference (noisy-run 3 1))
      (for ([p '(2 4)])
        (check-equal? (noisy-run 3 p) reference (format "prefetch ~a" p))
        (check-equal? (noisy-run 3 p #:jitter? #t) reference (format "prefetch ~a, jitter" p))))

    (test-case "workers leave the calling thread's streams alone"
      (define (after ds)
        (manual-seed! 5)
        (define g (make-generator 9))
        (define prng (seeded-prng 3))
        (parameterize ([current-generator g] [current-pseudo-random-generator prng])
          (within 60 (lambda ()
                       (for ([_ (dataloader ds #:batch-size 5 #:shuffle? #t
                                            #:generator (make-generator 7)
                                            #:collate (lambda (items) (stack (map car items)))
                                            #:workers 3)])
                         (void))))
          (list (tensor->list (randn 3)) (tensor->list (randperm 5)) (random))))
      (check-equal? (after (noisy 12 #f)) (after (tensor-dataset (zeros 12 3 6 6)))))

    (struct exn:fail:test exn:fail ())

    (define (failing-at bad raised slow?)
      (define-dataset failing (n)
        #:length n
        #:ref (i)
        (when (and slow? (< i bad)) (sleep 0.1))
        (when (= i bad) (raise raised))
        (tensor (list i) #:dtype 'int64))
      (failing 6))

    (define (caught thunk)
      (with-handlers ([(lambda (_v) #t) values]) (thunk) 'nothing-raised))

    (test-case "a worker's exception is raised at its batch's position"
      (define raised (exn:fail:test "batch 3 failed" (current-continuation-marks)))
      (define seen '())
      (define v
        (within 60 (lambda ()
                     (caught (lambda ()
                               (for ([b (dataloader (failing-at 3 raised #t) #:workers 2)])
                                 (set! seen (cons (tensor->list b) seen))))))))
      (check-equal? (reverse seen) '((0) (1) (2)))
      (check-eq? v raised)
      (check-equal? (exn-message v) "batch 3 failed"))

    (test-case "a raised non-exception value is raised as is"
      (check-eq? (within 60 (lambda ()
                              (caught (lambda ()
                                        (for ([_ (dataloader (failing-at 1 'boom #f) #:workers 2)])
                                          (void))))))
                 'boom))

    (test-case "a failing native op in a worker reports its own message"
      (define (failing-collate _items) (@ (ones 2 3) (ones 4 2)))
      (define (message workers)
        (define v
          (within 60 (lambda ()
                       (caught (lambda ()
                                 (for ([_ (loader-of pairs workers 2 '#:collate failing-collate)])
                                   (void)))))))
        (and (exn? v) (exn-message v)))
      (check-regexp-match #rx"2x3 and 4x2" (message 0))
      (check-regexp-match #rx"2x3 and 4x2" (message 2)))

    (test-case "a traversal that raised stops its workers and raises again"
      (define raised (exn:fail:test "batch 3 failed" (current-continuation-marks)))
      (define c (make-custodian))
      (define-values (more? next)
        (under c (lambda ()
                   (~> (failing-at 3 raised #t)
                       (dataloader #:workers 2)
                       in-dataloader
                       sequence-generate))))
      (within 60 (lambda () (for ([_ (in-range 3)]) (next))))
      (define threads (live c))
      (check-equal? (length threads) 2)
      (check-eq? (within 60 (lambda () (caught next))) raised)
      (check-true (dead-within? threads 5))
      (check-eq? (within 10 (lambda () (caught next))) raised)
      (void more?)
      (shut-down-custodian! c))

    (define-dataset stalling (n stall-at stall builder)
      #:length n
      #:ref (i)
      (when (= i stall-at)
        (set-box! builder (current-thread))
        (sleep stall))
      (tensor (list i) #:dtype 'int64))

    (define (stalled-builder builder)
      (for/or ([_ (in-range 1000)])
        (or (unbox builder) (and (sync/timeout 0.01 never-evt) #f))))

    (define (consuming c ds outcome)
      (thread (lambda ()
                (set-box! outcome
                          (caught (lambda ()
                                    (under c (lambda ()
                                               (for ([_ (dataloader ds #:workers 2)])
                                                 (void))))))))))

    (test-case "a worker that dies raises in the consumer"
      (define builder (box #f))
      (define outcome (box #f))
      (define c (make-custodian))
      (define consumer (consuming c (stalling 4 1 30 builder) outcome))
      (define worker (stalled-builder builder))
      (check-pred thread? worker)
      (kill-thread worker)
      (check-not-false (sync/timeout 10 consumer))
      (check-pred exn:fail? (unbox outcome))
      (check-regexp-match #rx"worker [0-9]+ exited before delivering batch"
                          (exn-message (unbox outcome)))
      (shut-down-custodian! c))

    (test-case "a worker's death stops the others and raises the same value again"
      (define builder (box #f))
      (define c (make-custodian))
      (define-values (more? next)
        (under c (lambda ()
                   (~> (stalling 40 1 30 builder)
                       (dataloader #:workers 2 #:prefetch 4)
                       in-dataloader
                       sequence-generate))))
      (within 60 next)
      (define threads (live c))
      (check-equal? (length threads) 2)
      (kill-thread (stalled-builder builder))
      (define raised (within 10 (lambda () (caught next))))
      (check-pred exn:fail? raised)
      (check-true (dead-within? threads 5))
      (check-eq? (within 10 (lambda () (caught next))) raised)
      (void more?)
      (shut-down-custodian! c))

    (test-case "exhaustion stops the workers without a collection"
      (settle!)
      (define base (ledger-snapshot))
      (define threads
        (workers-seen-by (make-custodian)
                         (lambda (see!)
                           (for ([(_x _y) (dataloader pairs #:batch-size 3 #:shuffle? #t
                                                      #:workers 2)])
                             (see!)))))
      (check-equal? (length threads) 2)
      (check-true (dead-within? threads 5))
      (settle!)
      (check-ledger base #f))

    (test-case "the last delivery stops the workers"
      (settle!)
      (define base (ledger-snapshot))
      (define threads
        (workers-seen-by (make-custodian)
                         (lambda (see!)
                           (for ([_ (in-range 4)]
                                 [(_x _y) (dataloader pairs #:batch-size 3 #:workers 2)])
                             (see!)))))
      (check-equal? (length threads) 2)
      (check-true (dead-within? threads 5))
      (settle!)
      (check-ledger base #f))

    (test-case "breaking out of the loop stops the workers after a collection"
      (settle!)
      (define base (ledger-snapshot))
      (define threads
        (workers-seen-by (make-custodian)
                         (lambda (see!)
                           (for ([_ (in-range 1)]
                                 [_b (~> (arange 16)
                                         (reshape 8 2)
                                         tensor-dataset
                                         (dataloader #:workers 2 #:prefetch 2))])
                             (see!)))))
      (check-equal? (length threads) 2)
      (check-true (dead-after-collections? threads))
      (settle!)
      (check-ledger base #f))

    (test-case "custodian shutdown stops the workers"
      (settle!)
      (define base (ledger-snapshot))
      (define builder (box #f))
      (define c (make-custodian))
      (define consumer
        (parameterize ([current-custodian c])
          (consuming c (stalling 4 1 30 builder) (box #f))))
      (check-pred thread? (stalled-builder builder))
      (define workers (remq consumer (live c)))
      (check-equal? (length workers) 2)
      ;; a worker still running when its custodian shuts down wedges Racket
      ;; 9.3 (see private/wedge.rkt); by now both are parked
      (sleep 0.2)
      (custodian-shutdown-all c)
      (check-true (for/and ([t (in-list workers)]) (thread-dead? t)))
      (check-true (thread-dead? consumer))
      (settle!)
      (check-ledger base #f))

    (test-case "a break while waiting leaves no live workers"
      (settle!)
      (define base (ledger-snapshot))
      (define builder (box #f))
      (define outcome (box #f))
      (define c (make-custodian))
      (define consumer (consuming c (stalling 4 0 2 builder) outcome))
      (check-pred thread? (stalled-builder builder))
      (define threads (live c))
      (check-equal? (length threads) 2)
      (break-thread consumer)
      (check-not-false (sync/timeout 1 consumer))
      (check-pred exn:break? (unbox outcome))
      (set-box! outcome #f)
      (check-true (dead-after-collections? threads))
      (settle!)
      (check-ledger base #f))

    (define (first-batch-boxes next)
      (call-with-values next (lambda vs (map make-weak-box vs))))

    (test-case "a delivered batch is unreachable from the workers"
      (define builder (box #f))
      (define c (make-custodian))
      (define-values (more? next)
        (under c (lambda ()
                   (~> (stalling 4 1 30 builder) (dataloader #:workers 2) in-dataloader
                       sequence-generate))))
      (define boxes (within 60 (lambda () (first-batch-boxes next))))
      (check-pred thread? (stalled-builder builder))
      (define waiter (thread next))
      (check-false (sync/timeout 0.2 waiter) "batch 1 is still being built")
      (collect-garbage 'major)
      (check-equal? (map weak-box-value boxes) '(#f))
      (check-equal? (length (live c)) 2 "the workers were alive throughout")
      (kill-thread waiter)
      (void more?)
      (shut-down-custodian! c))

    (test-case "freeing each delivered batch at once never double-releases"
      (settle!)
      (define base (ledger-snapshot))
      (define w (make-watch))
      (with-race-hook (watch-hook w)
        (within 60 (lambda ()
                     (for ([(x y) (dataloader pairs #:batch-size 2 #:shuffle? #t
                                              #:workers 2)])
                       (tensor-free! x)
                       (tensor-free! y))))
        (settle!))
      (check-ledger base w))

    (define (module-forms path)
      (parameterize ([read-accept-reader #t] [read-accept-lang #t])
        (call-with-input-file path read)))

    (define (atoms v)
      (match v
        [(cons a d) (append (atoms a) (atoms d))]
        ['() '()]
        [_ (list v)]))

    (define (sublists v)
      (match v
        [(cons a d) (cons v (append (sublists a) (sublists d)))]
        [_ '()]))

    (define (ffi-unsafe-imports forms)
      (for/list ([form (in-list (sublists forms))]
                 #:when (match form
                          [(list* 'only-in 'ffi/unsafe _) #t]
                          [_ #f]))
        (match-define (list* _ _ names) form)
        names))

    (test-case "the loader frees nothing and never enters atomic mode"
      (define sources
        (for/list ([name '("loader.rkt" "workers.rkt")])
          (module-forms (collection-file-path name "torch" "data"))))
      (define forbidden
        '(tensor-free! tr-tensor-free/checked free-handle! call-as-atomic start-atomic
          ffi/unsafe/atomic))
      (for ([forms (in-list sources)])
        (define names (atoms forms))
        (for ([name (in-list forbidden)])
          (check-false (memq name names) (format "~a appears" name)))
        (define imports (ffi-unsafe-imports forms))
        (for ([imported (in-list imports)])
          (check-equal? imported '(register-finalizer)))
        (check-equal? (length (filter (lambda (a) (eq? a 'ffi/unsafe)) names))
                      (length imports)
                      "ffi/unsafe is required only through only-in"))
      (match-define (list _ workers) sources)
      (check-not-false
       (for/or ([form (in-list (sublists workers))])
         (match form
           [(list* '#:pool ''own _) #t]
           [_ #f]))
       "#:pool 'own"))

    (define-dataset breaking-device (n armed)
      #:length n
      #:ref (i) i
      #:device (cond
                 [(unbox armed)
                  (set-box! armed #f)
                  (break-thread (current-thread))
                  #f]
                 [else #f])
      #:batch (indices _collate)
      (tensor (indices->list indices) #:dtype 'int64))

    (define (drawn-indices thunk)
      (with-handlers ([exn:break? exn-message])
        (~> (map tensor->list (thunk)) append* (sort <))))

    (test-case "a break in the first permutation draw is retried with workers 0"
      (define armed (box #f))
      (define-values (more? next)
        (~> (breaking-device 6 armed)
            (dataloader #:batch-size 2 #:shuffle? #t)
            in-dataloader
            sequence-generate))
      (set-box! armed #t)
      (check-pred exn:break? (within 10 (lambda () (caught next))))
      (check-false (unbox armed) "the break was injected")
      (define (rest)
        (match (more?)
          [#t (cons (next) (rest))]
          [#f '()]))
      (check-equal? (within 10 (lambda () (drawn-indices rest))) (range 6)))

    (test-case "a break in the worker sampler's first draw is retried"
      (define armed (box #f))
      (define-values (indices _finish!)
        (~> (breaking-device 6 armed)
            (dataloader #:batch-size 2 #:shuffle? #t #:workers 2)
            sampler-draws))
      (set-box! armed #t)
      (check-pred exn:break? (within 10 (lambda () (caught (lambda () (indices 0))))))
      (check-false (unbox armed) "the break was injected")
      (check-equal? (within 10 (lambda () (drawn-indices (lambda () (map indices (range 3))))))
                    (range 6)))

    (test-case "a batch re-asked after a break is delivered again"
      (define c (make-custodian))
      (define-values (count batch _finish!)
        (under c (lambda () (traversal (dataloader pairs #:batch-size 3 #:workers 2)))))
      (define (taken k)
        (within 10 (lambda () (call-with-values (lambda () (batch k)) record))))
      (define first-take (taken 0))
      (check-equal? (taken 0) first-take)
      (check-not-equal? (taken 1) first-take)
      (check-equal? count 4)
      (shut-down-custodian! c))

    (test-case "the consumer waits in a Racket sync"
      (define-dataset sleepy (n)
        #:length n
        #:ref (i)
        (when (zero? i) (sleep 1))
        (tensor (list i) #:dtype 'int64))
      (define (ticks-during thunk)
        (define ticks 0)
        (define ticker
          (thread (lambda ()
                    (let loop ()
                      (sleep 0.01)
                      (set! ticks (add1 ticks))
                      (loop)))))
        (define result (thunk))
        (kill-thread ticker)
        (values result ticks))
      (define-values (_slept baseline) (ticks-during (lambda () (sleep 1))))
      (define c (make-custodian))
      (define-values (cpu ticks)
        (ticks-during
         (lambda ()
           (within 60 (lambda ()
                        (define start (current-process-milliseconds (current-thread)))
                        (under c (lambda ()
                                   (for ([_b (dataloader (sleepy 4) #:workers 2)])
                                     (void))))
                        (- (current-process-milliseconds (current-thread)) start))))))
      (check-true (< cpu 200) (format "the consumer spent ~a ms of CPU" cpu))
      (check-true (positive? baseline) "the ticker ran over a plain sleep")
      (check-true (>= (* 2 ticks) baseline)
                  (format "the ticker ran ~a times, ~a over a plain one-second sleep" ticks baseline))
      (check-true (dead-within? (workers-of c) 5) "the traversal leaves no worker running")
      (shut-down-custodian! c))

    (test-case "workers that make tensors finish while the consumer waits"
      (check-equal? (within 60 (lambda ()
                                 (for/sum ([_ (~> (arange 256)
                                                  (reshape 128 2)
                                                  tensor-dataset
                                                  (dataloader #:batch-size 2 #:shuffle? #t
                                                              #:workers 4))])
                                   1)))
                    64))

    (define (gated-run workers)
      (define n 8)
      (define gates (for/vector #:length n ([_ (in-range n)]) (make-semaphore 0)))
      (define done (for/vector #:length n ([_ (in-range n)]) (make-semaphore 0)))
      (define w (ones 2 3 #:requires-grad? #t))
      (semaphore-post (vector-ref gates 0))
      (define (gated items)
        (define j (quotient (first-label items) 2))
        (sync/timeout 10 (vector-ref gates j))
        (begin0 (~> items (map car _) stack (@ w))
          (semaphore-post (vector-ref done j))))
      (define (step k thunk)
        (if (even? k) (with-autocast #:device 'cpu (thunk)) (with-no-grad (thunk))))
      (within 60
              (lambda ()
                (for/list ([xb (loader-of (tensor-dataset (reshape (arange 32) 16 2)
                                                          (arange 16 #:dtype 'int64))
                                          workers 2 '#:batch-size 2 '#:collate gated)]
                           [k (in-naturals)])
                  (define loss (and (requires-grad? xb) (sum xb)))
                  (cond
                    [(zero? workers) (semaphore-post (vector-ref gates (min (add1 k) (sub1 n))))]
                    [else
                     (step k (lambda ()
                               (when (< (add1 k) n)
                                 (semaphore-post (vector-ref gates (add1 k)))
                                 (sync/timeout 10 (vector-ref done (add1 k))))
                               (when loss (backward! loss))))])
                  (list (tensor->list xb) (dtype xb) (requires-grad? xb))))))

    (test-case "autocast and grad mode on the trainer do not reach worker collates"
      (define reference (gated-run 0))
      (check-true (andmap (lambda (b) (match b [(list _ 'float32 #t) #t] [_ #f])) reference))
      (check-known-failure
       "#194"
       "batches built while the trainer is in autocast or no-grad match workers 0"
       (equal? (gated-run 2) reference)))

    (test-case "worker batches are CPU tensors"
      (define acc (accelerator-if-available))
      (unless (eq? (device-type acc) 'cpu)
        (define-dataset leaking (n)
          #:length n
          #:ref (i) (to (tensor (list i)) acc))
        (define v
          (within 60 (lambda ()
                       (caught (lambda ()
                                 (for ([_ (dataloader (leaking 4) #:workers 2)])
                                   (void)))))))
        (check-pred exn:fail? v)
        (check-regexp-match #rx"workers build CPU tensors" (exn-message v))))))
