#lang racket/base

;; WIP trainer for the Conv-MNIST capstone (#5). Trains a small convnet on the
;; full MNIST set, on the GPU when one is available (set-default-device! 'cuda)
;; and on the CPU otherwise — the same code path either way. Verified: 98% test
;; accuracy in 3 epochs on an RTX 3090 Ti (~1.2s/epoch); 97% in 1 epoch on CPU.
;;
;; Run it:
;;   GPU:  nix develop .#cuda --command racket examples/racket/05-mnist.rkt
;;   CPU:  nix run .#copy-native-libs && \
;;         nix develop --command racket examples/racket/05-mnist.rkt
;; Point $RKTORCH_MNIST_DIR at a data disk to cache the dataset there; EPOCHS
;; overrides the epoch count.
;;
;; Step 4 of #5 turns this into the literate scribble/lp2 example + a Python
;; twin + a fixture-based parity cross-test. It lives under module+main so
;; requiring/compiling the file does no work (no download at load time).
;;
;; conv2d names both the functional op (torch) and the nn layer (torch/nn); we
;; keep the nn layer. max-pool2d/flatten: keep the functional ops from torch.
;; (The F.* vs nn.* split is issue #11; this except-in pattern is the workaround.)

(module+ main
  (require (except-in torch conv2d)
           (except-in torch/nn max-pool2d flatten)
           (only-in torch/data/mnist load-mnist))

  (define dev (if (cuda-available?) 'cuda 'cpu))
  (set-default-device! dev)
  (printf "device: ~a\n" dev)
  (manual-seed! 0)

  (define-values (train-x train-y) (load-mnist 'train))
  (define-values (test-x test-y) (load-mnist 'test))
  (define n-train (car (tensor-shape train-x)))
  (define n-test (car (tensor-shape test-x)))
  (printf "train ~a  test ~a\n" (tensor-shape train-x) (tensor-shape test-x))

  ;; LeNet-ish: 28 -c3-> 26 -pool-> 13 -c3-> 11 -pool-> 5 ; 32*5*5 = 800
  (define-module convnet ()
    #:submodules ([c1 (conv2d 1 16 3)]
                  [c2 (conv2d 16 32 3)]
                  [f1 (linear 800 128)]
                  [f2 (linear 128 10)])
    #:forward (x)
    (let* ([h (max-pool2d (relu (c1 x)) 2)]
           [h (max-pool2d (relu (c2 h)) 2)]
           [h (relu (f1 (flatten h 1)))])
      (f2 h)))

  (define net (convnet))
  (define opt (adam (parameters net) #:lr 0.001))
  (define batch 128)

  ;; Test accuracy, batched, under no-grad / eval mode.
  (define (accuracy)
    (eval! net)
    (begin0
      (call-with-no-grad
       (lambda ()
         (define correct
           (for/sum ([start (in-range 0 n-test 1000)])
             (define len (min 1000 (- n-test start)))
             (define preds (argmax (net (narrow test-x 0 start len)) 1))
             (item (sum (eq preds (narrow test-y 0 start len))))))
         (exact->inexact (/ correct n-test))))
      (train! net)))

  (define epochs (string->number (or (getenv "EPOCHS") "3")))
  (for ([epoch (in-range epochs)])
    (define t0 (current-inexact-milliseconds))
    (for ([start (in-range 0 n-train batch)])
      (define len (min batch (- n-train start)))
      (zero-grads! opt)
      (define loss (cross-entropy (net (narrow train-x 0 start len))
                                  (narrow train-y 0 start len)))
      (backward! loss)
      (step! opt))
    (printf "epoch ~a: test acc ~a  (~as)\n"
            (add1 epoch) (accuracy)
            (/ (round (- (current-inexact-milliseconds) t0)) 1000.0))))
