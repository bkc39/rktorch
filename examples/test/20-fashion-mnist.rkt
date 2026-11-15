#lang racket/base

(module twin racket/base
  (require (only-in racket/runtime-path #%datum define-runtime-path))
  (provide fashion-twin)
  (define-runtime-path fashion-twin "../python/20_fashion_mnist.py"))

(module+ main
  (require (only-in racket/format ~r)
           (only-in racket/list fourth last)
           (only-in racket/match match-define)
           torch/scribblings/examples/20-fashion-mnist
           (only-in "private/env.rkt" env-number))
  (printf "device: ~a\n" (pick-device))
  (define-values (rows _run) (train-fashion-mnist #:epochs (env-number "EPOCHS" 5)))
  (for ([row (in-list rows)])
    (match-define (list epoch train-loss val-loss acc seconds) row)
    (printf "epoch ~a: loss ~a, val loss ~a, val acc ~a, ~as\n" epoch
            (~r train-loss #:precision '(= 4)) (~r val-loss #:precision '(= 4))
            (~r acc #:precision '(= 4)) (~r seconds #:precision '(= 1))))
  (printf "test acc ~a\n" (~r (fourth (last rows)) #:precision '(= 4))))

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/list append-map count fourth last)
           (only-in racket/match match match-define)
           (only-in racket/math exact-ceiling)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           torch
           torch/data/loader
           torch/nn
           torch/scribblings/examples/20-fashion-mnist
           (only-in torch/tests/private/python-env
                    call-with-python-env python-cuda-available? python-module-available?
                    python-result python-torch-available? tol)
           (only-in torch/train
                    lambda-callback run-estimated-stepping-batches run-global-step
                    run-grad-scaler run-optimizers)
           (only-in torch/vision/fashion-mnist load-fashion-mnist-fixture)
           (only-in "private/lightning-trace.rkt"
                    compare-trace flat-state lightning-name trace-callback trace-labels
                    trace-gaps trace-logger trace-lrs trace-losses trace-norms trace-recorder)
           (only-in "../../scripts/fashion-mnist-results.rkt" fashion-mnist-record))

  (define-runtime-path bundle-path "../../torch/scribblings/results/fashion-mnist.rktd")

  (define epochs 2)
  (define total-steps 16)
  (define warmup 4)

  (define (state-of classifier)
    (flat-state (append (named-parameters classifier) (named-buffers classifier))))

  (define (traced device precision)
    (define tr (trace-recorder))
    (define stepping (box #f))
    (define (note-total! r) (set-box! stepping (run-estimated-stepping-batches r)))
    (define-values (r classifier)
      (run-example #:device device #:precision precision #:epochs epochs
                   #:callbacks (list (trace-callback tr)
                                     (lambda-callback #:on-train-start note-total!))
                   #:logger (trace-logger tr)))
    (values tr r classifier (unbox stepping)))

  (define (hand-loop precision)
    (manual-seed! seed)
    (define net (fashion-net))
    (define-values (xs ys) (load-fashion-mnist-fixture 'train))
    (define-values (vx vy) (load-fashion-mnist-fixture 'test))
    (define loader
      (dataloader (tensor-dataset xs ys) #:batch-size (hash-ref batch-size 'fixture)
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
    (validate! 2)
    (define recorded-losses '())
    (define recorded-lrs '())
    (for ([_ (in-range epochs)])
      (for ([i (in-range (dataloader-length loader))] [(xb yb) (in-dataloader loader)])
        (define loss (/ (loss-of xb yb) accumulate))
        (when (even? i) (zero-grads! opt))
        (backward! loss)
        (set! recorded-losses (cons (item loss) recorded-losses))
        (when (odd? i)
          (set! recorded-lrs
                (cons (list (learning-rate opt #:group 0) (learning-rate opt #:group 1))
                      recorded-lrs))
          (clip-grad-norm! params clip-val)
          (step! opt)
          (step! schedule)))
      (validate! 64))
    (values (flat-state (append (named-parameters net) (named-buffers net)))
            (reverse recorded-losses) (reverse recorded-lrs)))

  (define-values (trace-a run-a classifier-a stepping-a) (traced 'cpu '32-true))
  (define-values (trace-b run-b classifier-b _stepping-b) (traced 'cpu 'bf16-mixed))

  (define (without-prefix state)
    (for/list ([entry (in-list state)])
      (cons (regexp-replace #rx"^net[.]" (car entry) "") (cdr entry))))

  (test-case "G4-L7: fit runs the loop a hand-written one runs, in (a) and (b)"
    (for ([config (in-list (list (list '32-true trace-a classifier-a)
                                 (list 'bf16-mixed trace-b classifier-b)))])
      (match-define (list precision tr classifier) config)
      (define-values (state losses lrs) (hand-loop precision))
      (check-equal? (without-prefix (state-of classifier)) state (format "~a: state" precision))
      (check-equal? (reverse (trace-losses tr)) losses (format "~a: losses" precision))
      (check-equal? (reverse (trace-lrs tr)) lrs (format "~a: lrs" precision))))

  (test-case "G4-L8: the network's parameters and its two groups"
    (define net (fashion-net))
    (check-equal? (map (lambda (np) (cons (car np) (shape (cdr np)))) (named-parameters net))
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
    (check-equal? (exact-ceiling (/ stepping-a 4)) warmup)
    (check-equal? (run-global-step run-a) total-steps)
    (check-equal? (length (trace-labels trace-a)) 32)
    (define lrs (reverse (trace-lrs trace-a)))
    (check-equal? (length lrs) total-steps)
    (for ([pair (in-list lrs)])
      (check-true (= (car pair) (cadr pair)) (format "group rates differ: ~a" pair)))
    (check-equal? (car (list-ref lrs 0)) (* lr 0.1))
    (check-equal? (car (list-ref lrs warmup)) lr))

  (test-case "G4-L10: the clip is live in (a)"
    (check-true (> (apply max (trace-norms trace-a)) clip-val)))

  (test-case "G4-L20: only (c) has a scaler"
    (check-false (run-grad-scaler run-a))
    (check-false (run-grad-scaler run-b)))

  (define synthetic-rows '((1 0.9 0.8 0.7 10.0) (2 0.5 0.45 0.84 9.5)))

  (define (atoms? v)
    (match v
      [(? hash?) (for/and ([x (in-hash-values v)]) (atoms? x))]
      [_ (or (string? v) (real? v))]))

  (test-case "G4-L21: the record is strings and reals around the two sides' rows"
    (for ([dev (in-list (list (cpu-device) (cuda-device)))])
      (define record
        (fashion-mnist-record synthetic-rows (reverse synthetic-rows) dev
                              #:commit "0123abcd" #:seconds '(20.0 30.0)))
      (for ([(key value) (in-hash record)] #:unless (memq key '(racket lightning)))
        (check-true (atoms? value) (format "~a: ~s" key value)))
      (for ([side (in-list '(racket lightning))])
        (define s (hash-ref record side))
        (check-equal? (sort (hash-keys s) symbol<?) '(accuracy epochs seconds))
        (check-equal? (hash-ref s 'accuracy) (fourth (last (hash-ref s 'epochs)))))
      (check-equal? (hash-ref (hash-ref record 'racket) 'accuracy) 0.84)
      (check-equal? (hash-ref (hash-ref record 'lightning) 'seconds) 30.0)
      (check-equal? (hash-ref record 'seed) seed)))

  (test-case "G4-L22: the committed headline, both sides within 0.01"
    (define bundle (file->value bundle-path))
    (define ours (hash-ref bundle 'racket))
    (define theirs (hash-ref bundle 'lightning))
    (check-equal? (length (hash-ref ours 'epochs)) (length (hash-ref theirs 'epochs)))
    (check-equal? (hash-ref bundle 'seed) seed)
    (check-equal? (hash-ref bundle 'precision) "16-mixed")
    (check-equal? (hash-ref bundle 'config)
                  (hasheq 'lr lr 'weight-decay weight-decay 'clip clip-val
                          'accumulate accumulate 'batch (hash-ref batch-size 'full)
                          'epochs (length (hash-ref ours 'epochs))))
    (check-true (<= (abs (- (hash-ref ours 'accuracy) (hash-ref theirs 'accuracy))) 0.01))
    (check-true (>= (hash-ref ours 'accuracy) 0.88)))

  (define (twin-json device precision)
    (call-with-python-env
     #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device))
                 (cons "RKTORCH_PARITY_PRECISION" (symbol->string precision))
                 (cons "RKTORCH_TWIN_MODE" "fixture")
                 (cons "EPOCHS" (number->string epochs)))
     (lambda () (python-result "python/20_fashion_mnist.py"))))

  (define (initial-state)
    (manual-seed! seed)
    (state-of (FashionClassifier)))

  (define (check-names r classifier json)
    (define names (hash-ref json 'names))
    (define (named-of xs) (map (lambda (np) (lightning-name (car np))) xs))
    (check-equal? (named-of (named-parameters classifier)) (hash-ref names 'parameters))
    (check-equal? (named-of (named-buffers classifier)) (hash-ref names 'buffers))
    (define by-tensor
      (for/hasheq ([np (in-list (named-parameters classifier))]) (values (cdr np) (car np))))
    (match-define (list decay no-decay)
      (for/list ([g (in-list (optimizer-param-groups (car (run-optimizers r))))])
        (map (lambda (t) (hash-ref by-tensor t)) (param-group-params g))))
    (check-equal? (map lightning-name decay) (hash-ref names 'decay))
    (check-equal? (map lightning-name no-decay) (hash-ref names 'no_decay)))

  (define (check-twin label device precision tr r classifier tolerance acc-tolerance)
    (define json (twin-json device precision))
    (test-case (format "G4-L11: ~a: the twin's config is the chapter's" label)
      (check-equal? (hash-ref json 'config)
                    (hasheq 'lr lr 'wd weight-decay 'clip clip-val 'A accumulate
                            'batch (hash-ref batch-size 'fixture) 'epochs epochs
                            'W warmup 'T total-steps 'seed seed 'lightning "2.6.5")))
    (test-case (format "G4-L12: ~a: names and groups" label)
      (check-names r classifier json))
    (test-case (format "G4-L13: ~a: the initial state is the twin's, bit for bit" label)
      (for ([entry (in-list (initial-state))])
        (match-define (cons name values) entry)
        (define theirs (hash-ref (hash-ref json 'init) (string->symbol name)))
        (check-true (and (= (length values) (length theirs)) (andmap = values theirs))
                    name)))
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
     (check-twin "(a)" 'cpu '32-true trace-a run-a classifier-a tol tol)
     (check-twin "(b)" 'cpu 'bf16-mixed trace-b run-b classifier-b tol tol)
     (when (and (cuda-available?) (python-cuda-available?))
       (define-values (trace-c run-c classifier-c _stepping-c) (traced 'cuda '16-mixed))
       (check-true (grad-scaler? (run-grad-scaler run-c)))
       (check-twin "(c)" 'cuda '16-mixed trace-c run-c classifier-c 5e-3 (/ 1.0 128)))]))
