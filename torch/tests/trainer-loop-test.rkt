#lang racket/base

(module+ test
  (require (for-syntax racket/base (only-in syntax/parse/pre ~optional ~seq expr id))
           (only-in racket/list drop-right make-list)
           (only-in racket/match match-define)
           (only-in racket/math exact-ceiling)
           rackunit
           (only-in syntax/parse/define define-syntax-parse-rule)
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../foreign/ops.rkt" move-generation)
           (only-in "../main.rkt"
                    cpu-device cuda-available? cuda-device device device-type
                    grad-enabled? item maybe-grad ones tensor->list)
           (only-in "../nn.rkt"
                    Dropout Linear eval! in-layers layer-copy layer-training?
                    mse-loss parameters sgd state-dict train!)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt"
                    collecting-finalized collecting-logger collecting-metrics))

  (struct tally (notes skip-at false-at))

  (define (new-tally #:skip-at [skip-at #f] #:false-at [false-at #f])
    (tally (make-hasheq) skip-at false-at))

  (define (note! t key v)
    (hash-update! (tally-notes t) key (lambda (vs) (append vs (list v))) '()))

  (define (notes t key) (hash-ref (tally-notes t) key '()))

  (define (place r) (list (run-epoch r) (run-batch-index r)))

  (define (grad-values net)
    (for/list ([p (in-list (parameters net))])
      (define g (maybe-grad p))
      (and g (tensor->list g))))

  (define (modes layer)
    (for/list ([m (in-sequences (list layer) (in-layers layer))])
      (layer-training? m)))

  (define-trainer Counted (net t)
    (define-init (tl) (set! net (Linear 1 1)) (set! t tl))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (note! t 'training-step (place run))
      (note! t 'device (device-type (device x)))
      (define loss (mse-loss (net x) y))
      (note! t 'step-loss (item loss))
      (and (not (equal? (tally-false-at t) (place run))) loss))
    (define-validation-step (x y)
      (note! t 'validation-step (list (grad-enabled?) (layer-training? net)))
      (mse-loss (net x) y))
    (define-hook (on-before-zero-grad optimizer)
      (note! t 'zero-grad (grad-values net)))
    (define-hook (on-before-backward loss)
      (note! t 'backward-loss (item loss))
      (when (equal? (tally-skip-at t) (place run)) (skip-step! run)))
    (define-hook (on-after-backward)
      (note! t 'after-backward (place run)))
    (define-hook (on-before-optimizer-step optimizer)
      (note! t 'optimizer-step (grad-values net)))
    (define-hook (on-validation-start)
      (note! t 'validation (list (run-epoch run) (run-global-step run)
                                 (grad-values net))))
    (define-hook (on-train-epoch-start)
      (note! t 'epoch (run-epoch run))))

  (define (ones-loader n #:batch-size [b 1])
    (dataloader (tensor-dataset (ones n 1) (ones n 1)) #:batch-size b))

  (define (expected-batches b limit)
    (case limit
      [(1.0) b]
      [(0.5) (quotient b 2)]
      [(2) (min 2 b)]))

  (test-case "L-C1, L-C2: steps and training steps per window and epoch"
    (for* ([a (in-range 1 5)]
           [b (in-range 1 8)]
           [epochs (in-range 1 4)]
           [limit (in-list '(1.0 0.5 2))])
      (define t (new-tally))
      (define (fit-counted)
        (fit (Counted t) (ones-loader b)
             #:max-epochs epochs #:accumulate-grad-batches a
             #:limit-train-batches limit #:logger #f))
      (define be (expected-batches b limit))
      (define label (format "A=~a b=~a E=~a limit=~a" a b epochs limit))
      (cond
        [(zero? be)
         (check-exn #rx"#:limit-train-batches selects no batch from this loader"
                    fit-counted label)]
        [else
         (define r (fit-counted))
         (define steps (* epochs (exact-ceiling (/ be a))))
         (check-equal? (length (notes t 'optimizer-step)) steps label)
         (check-equal? (length (notes t 'zero-grad)) steps label)
         (check-equal? (run-global-step r) steps label)
         (check-equal? (length (notes t 'training-step)) (* epochs be) label)])))

  (test-case "L-C3: #:max-steps stops after exactly that many steps"
    (define t (new-tally))
    (define r (fit (Counted t) (ones-loader 3) #:max-steps 5 #:logger #f))
    (check-equal? (run-global-step r) 5)
    (check-equal? (length (notes t 'training-step)) 5))

  (test-case "L-C4: validation every n epochs and every k batches"
    (define t (new-tally))
    (fit (Counted t) (ones-loader 2) #:val (ones-loader 2) #:max-epochs 4
         #:check-val-every-n-epoch 2 #:num-sanity-val-steps 0 #:logger #f)
    (check-equal? (for/list ([v (in-list (notes t 'validation))])
                    (match-define (list epoch _ _) v)
                    epoch)
                  '(1 3))
    (define u (new-tally))
    (fit (Counted u) (ones-loader 5) #:val (ones-loader 2) #:max-epochs 2
         #:val-check-interval 2 #:num-sanity-val-steps 0 #:logger #f)
    (check-equal? (for/list ([v (in-list (notes u 'validation))])
                    (match-define (list _ step _) v)
                    step)
                  '(2 4 7 9)))

  (test-case "L-C5: skip-step! and a #f loss drop the window's step"
    (for ([t (list (new-tally #:skip-at '(0 0)) (new-tally #:false-at '(0 0)))])
      (define r (fit (Counted t) (ones-loader 4) #:max-epochs 1
                     #:accumulate-grad-batches 2 #:logger #f))
      (check-equal? (run-global-step r) 1)
      (check-equal? (length (notes t 'optimizer-step)) 1)
      (check-equal? (length (notes t 'zero-grad)) 2)
      (check-equal? (notes t 'after-backward) '((0 2) (0 3)))))

  (test-case "L-C6: on-before-zero-grad sees the last step's gradients"
    (define t (new-tally))
    (fit (Counted t) (ones-loader 4) #:max-epochs 2 #:accumulate-grad-batches 2
         #:logger #f)
    (match-define (cons first-zeroed later-zeroed) (notes t 'zero-grad))
    (check-equal? first-zeroed '(#f #f))
    (check-equal? later-zeroed (drop-right (notes t 'optimizer-step) 1)))

  (test-case "L-C7: on-before-backward gets the loss divided by A"
    (define t (new-tally))
    (fit (Counted t) (ones-loader 7) #:max-epochs 1 #:accumulate-grad-batches 3
         #:logger #f)
    (check-equal? (length (notes t 'step-loss)) 7)
    (check-equal? (length (notes t 'backward-loss)) 7)
    (for ([step (in-list (notes t 'step-loss))]
          [divided (in-list (notes t 'backward-loss))])
      (check-= divided (/ step 3) (* 1e-6 (max 1.0 (abs step))))))

  (test-case "L-C8: a fraction limit selects the floor of its share"
    (define t (new-tally))
    (fit (Counted t) (ones-loader 7) #:max-epochs 1 #:limit-train-batches 0.5
         #:logger #f)
    (check-equal? (length (notes t 'training-step)) 3))

  (define-callback sanity-watch (t)
    (define-hook (on-sanity-check-start) (note! t 'sanity #t)))

  (define-callback saver (t)
    #:kind checkpoint
    (define-hook (on-train-epoch-end) (note! t 'saver #t)))

  (define-callback stopper (t)
    #:kind monitor
    (define-hook (on-train-epoch-end) (note! t 'stopper #t)))

  (test-case "L-C9: #:fast-dev-run runs n batches once, with no sanity, logger, monitor or checkpoint"
    (define t (new-tally))
    (define l (collecting-logger))
    (fit (Counted t) (ones-loader 5) #:val (ones-loader 5) #:max-epochs 3
         #:fast-dev-run 2 #:logger l
         #:callbacks (list (sanity-watch t) (saver t) (stopper t)))
    (check-equal? (length (notes t 'training-step)) 2)
    (check-equal? (length (notes t 'validation-step)) 2)
    (check-equal? (notes t 'epoch) '(0))
    (check-equal? (notes t 'sanity) '())
    (check-equal? (notes t 'saver) '())
    (check-equal? (notes t 'stopper) '())
    (check-equal? (collecting-metrics l) '()))

  (test-case "L-M1, L-M3: evaluation runs without grad, in eval mode, from released grads"
    (define t (new-tally))
    (define tr (Counted t))
    (define before (modes tr))
    (define r (fit tr (ones-loader 4) #:val (ones-loader 2) #:max-epochs 2
                   #:logger #f))
    (check-equal? (notes t 'validation-step) (make-list 6 '(#f #f)))
    (check-equal? (modes (run-trainer r)) before)
    (check-equal? (length (notes t 'validation)) 3)
    (for ([v (in-list (notes t 'validation))])
      (match-define (list _ _ grads) v)
      (check-true (andmap not grads))))

  (define-trainer Moded (net drop t)
    (define-init (d tl) (set! net (Linear 1 1)) (set! drop d) (set! t tl))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y)
      (note! t 'modes (list (layer-training? net) (layer-training? drop)))
      (mse-loss (net (drop x)) y)))

  (test-case "L-M2: a layer in eval mode before fit stays in eval mode"
    (define t (new-tally))
    (define d (Dropout))
    (eval! d)
    (fit (Moded d t) (ones-loader 3) #:max-epochs 1 #:logger #f)
    (check-equal? (notes t 'modes) '((#t #f) (#t #f) (#t #f))))

  (define-callback carrier (head t)
    (define-init (tl) (set! head (Linear 1 1)) (set! t tl))
    (define-hook (on-fit-start)
      (note! t 'callback-device (device-type (device (car (parameters head)))))))

  (test-case "L-D1: #:device moves the trainer and callbacks, and sends batches there"
    (when (cuda-available?)
      (define t (new-tally))
      (fit (Counted t) (ones-loader 2) #:max-epochs 1 #:device (cuda-device)
           #:logger #f #:callbacks (list (carrier t)))
      (check-equal? (notes t 'device) '(cuda cuda))
      (check-equal? (notes t 'callback-device) '(cuda))))

  (test-case "L-D2: fitting on the trainer's own device moves nothing"
    (define t (new-tally))
    (define tr (Counted t))
    (define params (parameters tr))
    (define generation (move-generation))
    (define r (fit tr (ones-loader 2) #:max-epochs 1 #:device (cpu-device) #:logger #f))
    (check-equal? (move-generation) generation)
    (check-true (andmap eq? params (parameters (run-trainer r)))))

  (test-case "L-D3: with #:device #f, batches go to the trainer's device"
    (define t (new-tally))
    (fit (Counted t) (ones-loader 2) #:max-epochs 1 #:logger #f)
    (check-equal? (notes t 'device) '(cpu cpu)))

  (define-trainer Fielded (net seen)
    (define-init () (set! net (Linear 1 1)) (set! seen 0))
    (define-forward () seen)
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-train-epoch-start) (set! seen (add1 seen))))

  (define-trainer Fixed (net)
    (define-init () (set! net (Linear 1 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-fit-start) (set! net (Linear 1 1))))

  (define-trainer Pair (a b lr label)
    (define-init () (set! a (Linear 1 2)) (set! b (Linear 2 1)) (set! lr 0.1)
      (set! label "pair"))
    (define-hparams lr)
    (define-forward (x) (b (a x)))
    (define-optimizers (sgd (parameters a) #:lr lr))
    (define-training-step (x y) (mse-loss (b (a x)) y)))

  (test-case "L-F1: a plain field written in a hook is seen afterwards"
    (define r (fit (Fielded) (ones-loader 2) #:max-epochs 3 #:logger #f))
    (check-equal? ((run-trainer r)) 3))

  (test-case "L-F2: a field classified at construction is fixed"
    (check-exn (lambda (e)
                 (and (exn:fail:contract? e)
                      (regexp-match? #rx"net" (exn-message e))
                      (regexp-match? #rx"Fixed" (exn-message e))))
               (lambda () (fit (Fixed) (ones-loader 2) #:max-epochs 1 #:logger #f))))

  (test-case "L-F3: parameters and state-dict are the child fields'"
    (define tr (Pair))
    (check-equal? (map car (state-dict tr)) '("a.weight" "a.bias" "b.weight" "b.bias"))
    (check-equal? (length (parameters tr)) 4))

  (test-case "L-F4: train! and eval! are idempotent and reach every child"
    (define tr (Pair))
    (eval! tr)
    (eval! tr)
    (check-equal? (modes tr) '(#f #f #f))
    (train! tr)
    (train! tr)
    (check-equal? (modes tr) '(#t #t #t)))

  (test-case "L-F5: layer-copy gives independent parameters and the same hooks"
    (define t (new-tally))
    (define tr (Counted t))
    (define copy (layer-copy tr))
    (check-false (ormap eq? (parameters tr) (parameters copy)))
    (check-equal? (map tensor->list (parameters tr)) (map tensor->list (parameters copy)))
    (fit copy (ones-loader 2) #:max-epochs 1 #:logger #f)
    (check-equal? (length (notes t 'training-step)) 2)
    (check-not-equal? (map tensor->list (parameters tr))
                      (map tensor->list (parameters copy))))

  (define-syntax-parse-rule (define-order-callback name:id (~optional (~seq #:kind kind:id)))
    (define-callback name (label t)
      (~? (~@ #:kind kind))
      (define-hook (on-fit-start) (note! t 'order (list 'fit-start label)))
      (define-hook (on-train-epoch-end) (note! t 'order (list 'epoch-end label)))))

  (define-order-callback tag)
  (define-order-callback watch #:kind monitor)
  (define-order-callback keep #:kind checkpoint)
  (define-order-callback other)

  (define-trainer Ordered (net t)
    (define-init (tl) (set! net (Linear 1 1)) (set! t tl))
    (define-callbacks (tag "method" t))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-fit-start) (note! t 'order '(fit-start trainer)))
    (define-hook (on-train-epoch-end) (note! t 'order '(epoch-end trainer))))

  (test-case "callbacks: define-callbacks first, replacing its type; checkpoint last"
    (define t (new-tally))
    (fit (Ordered t) (ones-loader 2) #:max-epochs 1 #:logger #f
         #:callbacks (list (keep "keep" t) (tag "replaced" t) (watch "watch" t)
                           (other "other" t)))
    (check-equal? (notes t 'order)
                  '((fit-start "method") (fit-start "watch") (fit-start "other")
                    (fit-start "keep") (fit-start trainer)
                    (epoch-end "method") (epoch-end "other") (epoch-end trainer)
                    (epoch-end "watch") (epoch-end "keep"))))

  (define planned (exn:fail "planned" (current-continuation-marks)))

  (define-trainer Failing (net t)
    (define-init (tl) (set! net (Linear 1 1)) (set! t tl))
    (define-optimizers (sgd (parameters net) #:lr 0.01))
    (define-training-step (x y) (raise planned))
    (define-hook (teardown stage) (note! t 'teardown stage)))

  (define-callback breaking (t)
    (define-hook (on-exception e) (error 'breaking "inside on-exception")))

  (define-callback catching (t)
    (define-hook (on-exception e) (note! t 'caught e))
    (define-hook (teardown stage) (note! t 'teardown stage)))

  (test-case "exceptions: re-raised eq?, every on-exception and teardown run once"
    (define t (new-tally))
    (define l (collecting-logger))
    (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-train))
    (define raised
      (with-handlers ([values values])
        (fit (Failing t) (ones-loader 2) #:max-epochs 1 #:logger l
             #:callbacks (list (breaking t) (catching t)))))
    (check-eq? raised planned)
    (check-equal? (notes t 'caught) (list planned))
    (check-equal? (notes t 'teardown) '(fit fit))
    (check-equal? (collecting-finalized l) '(failed))
    (check-not-false (sync/timeout 0 receiver))))
