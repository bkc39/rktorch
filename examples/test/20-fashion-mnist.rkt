#lang racket/base

(module+ main
  (require (only-in torch accelerator-if-available)
           (only-in "private/env.rkt" env-number)
           (only-in "private/fashion-mnist-kit.rkt" print-epoch-rows run-headline))
  (define device (accelerator-if-available))
  (printf "device: ~a\n" device)
  (define-values (rows _run _total) (run-headline #:epochs (env-number "EPOCHS" 5) #:device device))
  (print-epoch-rows rows))

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/list append* append-map count fifth filter-map first fourth last
                    second third)
           (only-in racket/match match match-define match-lambda)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           torch
           torch/data/loader
           torch/nn
           torch/scribblings/examples/20-fashion-mnist
           (only-in torch/tests/private/python-env
                    python-cuda-available? python-module-available? python-torch-available? tol)
           (only-in torch/train run-global-step run-grad-scaler run-optimizers)
           (only-in torch/vision/fashion-mnist load-fashion-mnist-fixture)
           (only-in "private/fashion-mnist-kit.rkt"
                    epoch-recorder fixture-batch-size full-batch-size run-example twin-config
                    twin-json)
           (only-in "private/lightning-trace.rkt"
                    compare-trace flat-state lightning-name trace-callback trace-gaps
                    trace-labels trace-logger trace-losses trace-lrs trace-norms trace-recorder
                    trace-rows)
           (only-in "../../scripts/fashion-mnist-results.rkt" fashion-mnist-record))

  (define-runtime-path bundle-path "../../torch/scribblings/results/fashion-mnist.rktd")

  (match-define (hash-table ('seed seed) ('lr lr) ('weight-decay weight-decay)
                            ('accumulate accumulate) ('clip clip-val))
    fashion-recipe)

  (define epochs 2)
  (define total-steps 16)
  (define warmup 4)

  (define (state-of classifier)
    (flat-state (append (named-parameters classifier) (named-buffers classifier))))

  (define (traced device precision)
    (define tr (trace-recorder))
    (define-values (recorder rows) (epoch-recorder))
    (define-values (r classifier stepping)
      (run-example #:device device #:precision precision #:epochs epochs
                   #:callbacks (list (trace-callback tr) recorder)
                   #:logger (trace-logger tr)))
    (values tr r classifier stepping (rows)))

  (define (hand-loop precision)
    (manual-seed! seed)
    (define net (fashion-net))
    (define-values (xs ys) (load-fashion-mnist-fixture 'train))
    (define-values (vx vy) (load-fashion-mnist-fixture 'test))
    (define loader
      (dataloader (tensor-dataset xs ys) #:batch-size fixture-batch-size
                  #:shuffle? #t #:generator (make-generator seed)))
    (define val (dataloader (tensor-dataset vx vy) #:batch-size 64))
    (define opt (adamw (weight-decay-param-groups net #:weight-decay weight-decay) #:lr lr))
    (define schedule
      (sequential-lr opt
                     (list (linear-lr opt #:start-factor 0.1 #:total-iters warmup)
                           (cosine-annealing-lr opt #:t-max (- total-steps warmup)))
                     #:milestones (list warmup)))
    (define params (append-map param-group-params (optimizer-param-groups opt)))
    (define (loss-of xb yb)
      (match precision
        ['bf16-mixed
         (with-autocast #:device 'cpu #:dtype 'bfloat16 #:cache-enabled? #f
           (cross-entropy (net xb) yb))]
        [_ (cross-entropy (net xb) yb)]))
    (define (validate! limit)
      (eval! net)
      (with-no-grad
        (for ([_ (in-range limit)] [(xb yb) (in-dataloader val)])
          (loss-of xb yb)))
      (train! net))
    (define (step-batch! i xb yb)
      (define loss (/ (loss-of xb yb) accumulate))
      (when (even? i) (zero-grads! opt))
      (backward! loss)
      (define rates
        (and (odd? i) (list (learning-rate opt #:group 0) (learning-rate opt #:group 1))))
      (when rates
        (clip-grad-norm! params clip-val)
        (step! opt)
        (step! schedule))
      (cons (item loss) rates))
    (define (train-epoch!)
      (begin0
        (for/list ([i (in-range (dataloader-length loader))] [(xb yb) (in-dataloader loader)])
          (step-batch! i xb yb))
        (validate! 64)))
    (validate! 2)
    (define steps (append* (for/list ([_ (in-range epochs)]) (train-epoch!))))
    (values (flat-state (append (named-parameters net) (named-buffers net)))
            (map car steps) (filter-map cdr steps)))

  (define-values (trace-a run-a classifier-a stepping-a rows-a) (traced (cpu-device) '32-true))
  (define-values (trace-b run-b classifier-b _stepping-b _rows-b)
    (traced (cpu-device) 'bf16-mixed))

  (define (without-prefix state)
    (map (match-lambda [(cons name v) (cons (regexp-replace #rx"^net[.]" name "") v)]) state))

  (test-case "G4-L7: fit runs the loop a hand-written one runs, in (a) and (b)"
    (for ([config (in-list (list (list '32-true trace-a classifier-a)
                                 (list 'bf16-mixed trace-b classifier-b)))])
      (match-define (list precision tr classifier) config)
      (define-values (state losses lrs) (hand-loop precision))
      (check-equal? (without-prefix (state-of classifier)) state (format "~a: state" precision))
      (check-equal? (trace-losses tr) losses (format "~a: losses" precision))
      (check-equal? (trace-lrs tr) lrs (format "~a: lrs" precision))))

  (test-case "G4-L8: the network's parameters and its two groups"
    (define net (fashion-net))
    (check-equal? (map (match-lambda [(cons name t) (cons name (shape t))]) (named-parameters net))
                  '(("0.conv.weight" 16 1 3 3) ("0.conv.bias" 16)
                    ("0.bn.weight" 16) ("0.bn.bias" 16)
                    ("1.conv.weight" 32 16 3 3) ("1.conv.bias" 32)
                    ("1.bn.weight" 32) ("1.bn.bias" 32)
                    ("3.weight" 128 1568) ("3.bias" 128)
                    ("5.weight" 10 128) ("5.bias" 10)))
    (check-equal? (for/sum ([p (in-list (parameters net))]) (numel p)) 207018)
    (match-define (list decay no-decay) (weight-decay-param-groups net #:weight-decay weight-decay))
    (define (sizes g)
      (define ts (param-group-params g))
      (list (length ts) (for/sum ([t (in-list ts)]) (numel t))))
    (check-equal? (sizes decay) '(4 206736))
    (check-equal? (sizes no-decay) '(8 282))
    (check-equal? (hash-ref (param-group-hyperparameters no-decay) 'weight-decay) 0.0))

  (test-case "G4-L9: sixteen optimizer steps, four of warmup, over 32 batches"
    (check-equal? stepping-a total-steps)
    (check-equal? (warmup-steps stepping-a) warmup)
    (check-equal? (run-global-step run-a) total-steps)
    (check-equal? (length (trace-labels trace-a)) 32)
    (define lrs (trace-lrs trace-a))
    (check-equal? (length lrs) total-steps)
    (for ([pair (in-list lrs)])
      (match-define (list decayed undecayed) pair)
      (check-equal? decayed undecayed "the groups' rates"))
    (check-equal? (first (list-ref lrs 0)) (* lr 0.1))
    (check-equal? (first (list-ref lrs warmup)) lr))

  (test-case "G4-L10: the clip is live in (a)"
    (check-true (> (apply max (trace-norms trace-a)) clip-val)))

  (test-case "the epoch rows are the logged epoch reductions"
    (define (logged key)
      (for*/list ([entry (in-list (trace-rows trace-a))]
                  [v (in-value (hash-ref (second entry) key #f))]
                  #:when v)
        v))
    (check-equal? (map first rows-a) '(1 2))
    (check-equal? (map second rows-a) (logged 'train/loss-epoch))
    (check-equal? (map third rows-a) (logged 'val/loss))
    (check-equal? (map fourth rows-a) (logged 'val/acc))
    (check-true (andmap (lambda (row) (positive? (fifth row))) rows-a)))

  (test-case "G4-L20: only (c) has a scaler"
    (check-false (run-grad-scaler run-a))
    (check-false (run-grad-scaler run-b)))

  (define synthetic-rows '((1 0.9 0.8 0.7 10.0) (2 0.5 0.45 0.84 9.5)))

  (define synthetic-versions
    (hasheq 'python "3.12.0" 'torch "2.14.0" 'lightning "2.6.5" 'cuda "none"))

  (define (atoms? v)
    (match v
      [(? hash?) (for/and ([x (in-hash-values v)]) (atoms? x))]
      [_ (or (string? v) (real? v))]))

  (test-case "G4-L21: the record is strings and reals around the two sides' rows"
    (for ([dev (in-list (list (cpu-device) (cuda-device)))])
      (define record
        (fashion-mnist-record synthetic-rows (reverse synthetic-rows) dev
                              #:config (twin-config #:batch full-batch-size #:epochs 2
                                                    #:total 470 #:warmup 118)
                              #:versions synthetic-versions
                              #:commit "0123abcd" #:seconds '(20.0 30.0)))
      (for ([(key value) (in-hash record)] #:unless (memq key '(racket lightning)))
        (check-true (atoms? value) (format "~a: ~s" key value)))
      (for ([side (in-list '(racket lightning))])
        (define s (hash-ref record side))
        (check-equal? (sort (hash-keys s) symbol<?) '(accuracy epochs seconds))
        (check-equal? (hash-ref s 'accuracy) (fourth (last (hash-ref s 'epochs)))))
      (check-equal? (hash-ref (hash-ref record 'racket) 'accuracy) 0.84)
      (check-equal? (hash-ref (hash-ref record 'lightning) 'seconds) 30.0)
      (check-equal? (hash-ref record 'seed) seed)
      (check-equal? (hash-ref record 'python-torch) "2.14.0")
      (match-define (hash-table ('total total) ('warmup w)) (hash-ref record 'config))
      (check-equal? (list total w) '(470 118))))

  (test-case "G4-L22: the committed headline, both sides within 0.01"
    (define bundle (file->value bundle-path))
    (define ours (hash-ref bundle 'racket))
    (define theirs (hash-ref bundle 'lightning))
    (define n (length (hash-ref ours 'epochs)))
    (define total (~> (/ 60000 full-batch-size) ceiling (/ accumulate) ceiling (* n)))
    (check-equal? (length (hash-ref theirs 'epochs)) n)
    (check-equal? (hash-ref bundle 'seed) seed)
    (check-equal? (hash-ref bundle 'precision) "16-mixed")
    (check-equal? (hash-ref bundle 'config)
                  (twin-config #:batch full-batch-size #:epochs n #:total total
                               #:warmup (warmup-steps total)))
    (check-true (string? (hash-ref bundle 'python-torch)))
    (check-true (<= (abs (- (hash-ref ours 'accuracy) (hash-ref theirs 'accuracy))) 0.01))
    (check-true (>= (hash-ref ours 'accuracy) 0.88)))

  (define (initial-state)
    (manual-seed! seed)
    (state-of (FashionClassifier)))

  (define (check-names r classifier json)
    (define names (hash-ref json 'names))
    (define (named-of xs) (map (match-lambda [(cons name _) (lightning-name name)]) xs))
    (check-equal? (named-of (named-parameters classifier)) (hash-ref names 'parameters))
    (check-equal? (named-of (named-buffers classifier)) (hash-ref names 'buffers))
    (define by-tensor
      (for/hasheq ([np (in-list (named-parameters classifier))])
        (match-define (cons name t) np)
        (values t name)))
    (match-define (list decay no-decay)
      (for/list ([g (in-list (~> r run-optimizers car optimizer-param-groups))])
        (map (lambda (t) (hash-ref by-tensor t)) (param-group-params g))))
    (check-equal? (map lightning-name decay) (hash-ref names 'decay))
    (check-equal? (map lightning-name no-decay) (hash-ref names 'no_decay)))

  (define (check-twin label device precision tr r classifier tolerance acc-tolerance)
    (define json (twin-json device precision 'fixture epochs))
    (test-case (format "G4-L11: ~a: the twin's config is the chapter's" label)
      (check-equal? (hash-ref json 'config)
                    (twin-config #:batch fixture-batch-size #:epochs epochs
                                 #:total total-steps #:warmup warmup)))
    (test-case (format "G4-L12: ~a: names and groups" label)
      (check-names r classifier json))
    (test-case (format "G4-L13: ~a: the initial state is the twin's, bit for bit" label)
      (for ([entry (in-list (initial-state))])
        (match-define (cons name vals) entry)
        (check-equal? vals (hash-ref (hash-ref json 'init) (string->symbol name)) name)))
    (printf "[20-fashion-mnist] ~a largest gaps: ~s\n" label
            (trace-gaps tr json (state-of classifier)))
    (test-case (format "G4-L14..L19: ~a: the trace is the twin's" label)
      (compare-trace label tr json (state-of classifier) tolerance
                     #:acc-tolerance acc-tolerance))
    (test-case (format "G4-L20: ~a: scales" label)
      (check-equal? (count values (hash-ref json 'scales))
                    (if (run-grad-scaler r) total-steps 0))))

  (cond
    [(not (and (python-torch-available?) (python-module-available? "lightning")))
     (displayln "[20-fashion-mnist] parity skipped: python3 torch or lightning not available")]
    [else
     (check-twin "(a)" (cpu-device) '32-true trace-a run-a classifier-a tol tol)
     (check-twin "(b)" (cpu-device) 'bf16-mixed trace-b run-b classifier-b tol tol)])

  (cond
    [(not (and (python-torch-available?) (python-module-available? "lightning")
               (cuda-available?) (python-cuda-available?)))
     (displayln "[20-fashion-mnist] (c) skipped: CUDA, or python3 torch with CUDA, not available")]
    [else
     (define-values (trace-c run-c classifier-c _stepping-c _rows-c)
       (traced (cuda-device) '16-mixed))
     (check-true (grad-scaler? (run-grad-scaler run-c)))
     (check-twin "(c)" (cuda-device) '16-mixed trace-c run-c classifier-c 5e-3 (/ 1.0 128))]))
