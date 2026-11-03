#lang racket/base

(module twin racket/base
  (require (only-in racket/runtime-path #%datum define-runtime-path))
  (provide vit-twin)
  (define-runtime-path vit-twin "../python/18_vit.py"))

(module+ main
  (require (only-in racket/format ~r)
           (only-in racket/match match-define)
           torch/scribblings/examples/18-vit
           (only-in "private/env.rkt" env-number))
  (printf "device: ~a\n" (pick-device))
  (define-values (rows _classifier)
    (train-cifar10 #:epochs (env-number "EPOCHS" 300)))
  (for ([row (in-list rows)])
    (match-define (list epoch loss acc seconds) row)
    (printf "epoch ~a: loss ~a, ~as, test acc ~a\n" epoch
            (~r loss #:precision '(= 4)) (~r seconds #:precision '(= 1))
            (~r acc #:precision '(= 4)))))

(module+ test
  (require (only-in racket/list first last third)
           (only-in racket/match match-define match-lambda)
           (only-in racket/math nan?)
           rackunit
           torch
           torch/data/loader
           torch/nn
           (only-in torch/private/util with-temporary-directory)
           torch/scribblings/examples/18-vit
           (only-in torch/tests/private/python-env
                    call-with-python-env check-training-twin
                    python-cuda-available? python-torch-available? tol)
           (only-in torch/train fit lambda-callback run-epoch)
           (only-in torch/vision/cifar10 load-cifar10-fixture)
           (only-in (submod ".." twin) vit-twin)
           (only-in "../../scripts/vit-results.rkt" vit-record))

  (define steps 20)

  (define (finite-losses? losses)
    (andmap (lambda (l) (and (rational? l) (not (nan? l)))) losses))

  (define-values (losses classifier trained-on) (run-example #:device 'cpu))
  (check-equal? trained-on 'cpu)
  (check-equal? (length losses) steps)
  (check-true (finite-losses? losses) (format "non-finite loss: ~a" losses))
  (check-true (< (last losses) (first losses))
              (format "losses did not decrease: ~a" losses))

  (test-case "fit runs the loop a hand-written one runs"
    (define hand-losses
      (with-default-device 'cpu
        (manual-seed! 0)
        (define-values (xs ys) (load-cifar10-fixture))
        (define net (small-vit))
        (define opt (adam (parameters net) #:lr 1e-3 #:weight-decay 1e-4))
        (define schedule
          (one-cycle-lr opt #:max-lr 1e-3 #:total-steps steps #:pct-start 0.1))
        (for/list ([_ (in-range steps)])
          (zero-grads! opt)
          (define loss (cross-entropy (net xs) ys))
          (backward! loss)
          (step! opt)
          (step! schedule)
          (item loss))))
    (check-equal? losses hand-losses))

  (test-case "the parameters are the network's, under the trainer's field"
    (check-equal? (map tensor-shape (parameters classifier))
                  (map tensor-shape (parameters (small-vit))))
    (check-not-false (assoc "net.head.weight" (named-parameters classifier)))
    (check-equal? (tensor-shape (classifier (zeros 2 3 32 32))) '(2 10)))

  (test-case
   "augment-batch keeps the shape and the [-1, 1] range and moves pixels"
    (with-default-device 'cpu
      (manual-seed! 0)
      (define-values (xs _ys) (load-cifar10-fixture))
      (define augmented (augment-batch xs))
      (check-equal? (tensor-shape augmented) (tensor-shape xs))
      (check-true (<= -1.001 (item (min augmented)) (item (max augmented))
                      1.001))
      (check-false (equal? (tensor->list augmented) (tensor->list xs)))))

  (test-case "only training batches are augmented"
    (with-default-device 'cpu
      (manual-seed! 0)
      (define-values (all-xs all-ys) (load-cifar10-fixture))
      (define xs (narrow all-xs 0 0 16))
      (define ys (narrow all-ys 0 0 16))
      (define batch (dataloader (tensor-dataset xs ys) #:batch-size 16))
      (define augmented 0)
      (define (count! xb)
        (set! augmented (add1 augmented))
        xb)
      (fit (CifarViT (small-vit) #:total-steps 2 #:augment count!)
           batch #:val batch #:max-steps 2 #:logger #f)
      (check-equal? augmented 2)))

  (test-case "a row is the epoch's mean batch loss and the test accuracy"
    (define-values (xs ys) (load-cifar10-fixture))
    (define fixture (tensor-dataset xs ys))
    (define batch-losses (make-hasheqv))
    (define (record! r loss _extra _xb _yb)
      (hash-update! batch-losses (add1 (run-epoch r))
                    (lambda (losses) (cons (item loss) losses)) '()))
    (define-values (rows classifier)
      (train-cifar10
       #:epochs 2 #:batch 64 #:device 'cpu #:train fixture #:test fixture
       #:callbacks (list (lambda-callback #:on-train-batch-end record!))))
    (check-equal? (map first rows) '(1 2))
    (for ([row (in-list rows)])
      (match-define (list epoch loss _acc seconds) row)
      (define losses (hash-ref batch-losses epoch))
      (check-equal? (length losses) 4)
      (check-= loss (/ (apply + losses) (length losses)) 1e-5)
      (check-true (positive? seconds)))
    (eval! classifier)
    (define accuracy
      (with-no-grad
        (~> (classifier xs) (argmax 1) (eq ys) (to-dtype 'float32) mean
            item)))
    (check-= (third (last rows)) accuracy 1e-6))

  (test-case "the record is strings and numbers, and the chapter's config"
    (for ([dev (in-list (list (cpu-device) (mps-device)))])
      (define record
        (vit-record '((1 2.0 0.5 3.0) (2 1.5 0.6 3.0)) dev 6.0
                    #:commit "0123abcd"))
      (check-equal? (hash-ref record 'model) vit-config)
      (check-equal? (hash-ref record 'seed) seed)
      (check-equal? (hash-ref record 'precision) (symbol->string precision))
      (check-equal? (hash-ref record 'hardware)
                    (symbol->string (device-type dev)))
      (check-equal? (hash-ref record 'accuracy) 0.6)
      (for ([(key value) (in-hash record)]
            #:unless (memq key '(model epochs)))
        (check-true (or (string? value) (real? value))
                    (format "~a on ~a: ~s" key dev value)))))

  ;; Device RNG streams differ from the CPU's, so the on-device arm checks
  ;; convergence, never equality with the CPU losses above.
  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (define-values (a-losses a-classifier _a-dev) (run-example #:device accel))
    (check-equal? (device (first (parameters a-classifier))) accel)
    (check-true (finite-losses? a-losses)
                (format "non-finite loss on ~a: ~a" accel a-losses))
    (check-true (< (last a-losses) (first a-losses))
                (format "~a losses did not decrease: ~a" accel a-losses)))

  (define (train-on device)
    (define-values (twin-losses trained _device) (run-example #:device device))
    (values twin-losses
            (~> (named-parameters trained)
                (sort string<? #:key car)
                (map (match-lambda [(cons _ p) (reshape p -1)]) _)
                cat)))

  (define (check-twin device tolerance)
    (with-temporary-directory (dir)
      (define init (build-path dir "init.safetensors"))
      (with-default-device device
        (manual-seed! 0)
        (save-state! (small-vit) init))
      (call-with-python-env
       #:env (list (cons "RKTORCH_VIT_INIT" (path->string init)))
       (lambda ()
         (check-training-twin "18_vit" vit-twin train-on device tolerance)))))

  (cond
    [(not (python-torch-available?))
     (displayln "[18-vit] parity skipped: python3 `torch` not available")]
    [(and (cuda-available?) (python-cuda-available?))
     (check-twin 'cpu tol)
     (check-twin 'cuda tol)]
    [else (check-twin 'cpu tol)]))
