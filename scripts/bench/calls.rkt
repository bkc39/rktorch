#lang racket/base

(require (only-in torch reclaim-native-memory!)
         (only-in torch/foreign/raw/fault with-crossing-hook)
         (only-in torch/foreign/raw/pressure-settings native-collect-at-troughs)
         (only-in torch/tests/private/race-harness with-race-hook)
         (only-in "harness.rkt" summarize)
         (only-in "workloads/phases.rkt"
                  call-in-phased-device phased-example phased-phases phased-settings run-phases!
                  step-phases!))

(provide call-counts
         call-records)

(struct tally (crossings frees tensors))

(define (fresh-tally) (tally (box 0) (box 0) (box 0)))

(define (bump! b)
  (let loop ()
    (define v (unbox b))
    (unless (box-cas! b v (add1 v)) (loop))))

(define (counts t)
  (list (unbox (tally-crossings t)) (unbox (tally-frees t)) (unbox (tally-tensors t))))

(define (op-calls crossings frees tensors)
  (- crossings frees (* 2 tensors)))

(define (call-counts p)
  (define whole (fresh-tally))
  (define current (box (fresh-tally)))
  (define rows '())
  (define (count! field)
    (bump! (field whole))
    (bump! (field (unbox current))))
  (reclaim-native-memory!)
  (call-in-phased-device
   p
   (lambda ()
     (parameterize ([native-collect-at-troughs #f])
       (with-crossing-hook (lambda () (count! tally-crossings))
         (with-race-hook (lambda (name _subject)
                           (case name
                             [(ledger-entry-added) (count! tally-tensors)]
                             [(free-released finalizer-releasing) (count! tally-frees)]
                             [else (void)]))
           (step-phases! p #:around (lambda (name thunk)
                                      (define phase (fresh-tally))
                                      (set-box! current phase)
                                      (thunk)
                                      (set-box! current (fresh-tally))
                                      (set! rows (cons (cons name (counts phase)) rows)))))))))
  (values (reverse rows) (counts whole)))

(define (call-records p #:warmup [warmup 2] #:rounds [rounds 3])
  (for ([_ (in-range warmup)])
    (run-phases! p))
  (define-values (per-round wholes)
    (for/lists (per-round wholes) ([_ (in-range rounds)]) (call-counts p)))
  (define (phase-record name rows)
    (define-values (crossings frees tensors)
      (for/lists (crossings frees tensors) ([row (in-list rows)]) (apply values row)))
    (define ops (map op-calls crossings frees tensors))
    (hasheq 'schema 1 'suite "calls" 'group "phase" 'impl "rktorch"
            'case (phased-example p) 'variant (symbol->string name) 'unit "calls"
            'settings (phased-settings p)
            'samples crossings 'stats (summarize crossings)
            'frees frees 'free_stats (summarize frees)
            'tensors tensors 'tensor_stats (summarize tensors)
            'op_calls ops 'op_call_stats (summarize ops)))
  (append (for/list ([ph (in-list (phased-phases p))])
            (define name (car ph))
            (phase-record name (for/list ([round (in-list per-round)])
                                 (cdr (assq name round)))))
          (list (phase-record 'step wholes))))

(module+ test
  (require (only-in racket/list last)
           (only-in racket/match match-define)
           rackunit
           (only-in torch ones topk)
           (only-in (submod torch/foreign unsafe) tensor-free!)
           (only-in torch/foreign/raw/fault crossing-hook)
           (only-in torch/foreign/raw/race-points race-hook)
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "workloads/phases.rkt" mnist-phases phased resnet-phases))

  (define (field r k) (hash-ref r k))

  (call-with-process-deadline
   600
   (lambda ()
     (test-case "every phase of a step counts its crossings, frees and accounted tensors"
       (for ([p (in-list (list (mnist-phases 'cpu #:batch 4 #:samples 16)
                               (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4)))])
         (define records (call-records p #:warmup 1 #:rounds 2))
         (define names (map (lambda (r) (field r 'variant)) records))
         (check-equal? names (append (map (lambda (ph) (symbol->string (car ph))) (phased-phases p))
                                     '("step")))
         (for ([r (in-list records)])
           (check-equal? (length (field r 'samples)) 2)
           (check-true (andmap positive? (field r 'samples)) (field r 'variant))
           (check-true (andmap (lambda (n) (>= n 0)) (field r 'op_calls)) (field r 'variant)))))

     (test-case "the step, counted whole, makes the op calls its phases make"
       (for ([p (in-list (list (mnist-phases 'cpu #:batch 4 #:samples 16)
                               (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4)))])
         (define records (call-records p #:warmup 1 #:rounds 2))
         (define step (last records))
         (define phases (reverse (cdr (reverse records))))
         (check-equal? (field step 'op_calls)
                       (apply map + (map (lambda (r) (field r 'op_calls)) phases))
                       (phased-example p))))

     (test-case "counting runs without trough collections"
       (define seen 'unset)
       (define p (phased "probe" 'cpu (hasheq)
                         (list (cons 'probe (lambda () (set! seen (native-collect-at-troughs)))))
                         void))
       (call-records p #:warmup 0 #:rounds 1)
       (check-false seen))

     (test-case "frees and every output of a multi-output op are counted apart from op calls"
       (define x (ones 4 4))
       (define kept #f)
       (define p (phased "probe" 'cpu (hasheq)
                         (list (cons 'alloc (lambda () (set! kept (ones 2 2))))
                               (cons 'free (lambda () (tensor-free! (ones 2 2))))
                               (cons 'topk (lambda () (topk x 1))))
                         void))
       (match-define (list alloc-phase free-phase topk-phase _step)
         (call-records p #:warmup 0 #:rounds 1))
       (check-equal? (field alloc-phase 'frees) '(0))
       (check-equal? (field free-phase 'frees) '(1))
       (check-equal? (field free-phase 'tensors) '(1))
       (check-equal? (field free-phase 'op_calls) (field alloc-phase 'op_calls))
       (check-equal? (field topk-phase 'tensors) '(2)))

     (test-case "the hooks are restored after counting"
       (check-false (unbox crossing-hook))
       (check-false (unbox race-hook))))))
