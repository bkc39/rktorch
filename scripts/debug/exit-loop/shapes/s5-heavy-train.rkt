(require torch/nn)
(define-module mlp ()
  #:submodules ([fc1 (Linear 256 512)]
                [fc2 (Linear 512 512)]
                [fc3 (Linear 512 10)])
  #:forward (x)
  (fc3 (relu (fc2 (relu (fc1 x))))))
(define net (with-default-device dev (mlp)))
(define opt (adam (parameters net) #:lr 0.001))
(with-default-device dev
  (for ([step (in-range 400)])
    (zero-grads! opt)
    (define loss (mse-loss (net (randn 64 256)) (randn 64 10)))
    (backward! loss)
    (step! opt)
    (when (zero? (modulo step 100)) (printf "step ~a loss ~a\n" step (item loss)))))
(with-default-device dev (randn 1200 1200))
(with-default-device dev
  (for ([_ (in-range 3000)]) (void (matmul (randn 256 256) (randn 256 256)))))
(printf "REPRO-SHAPE-OK s5\n")
(finalizer-failures)
(native-memory-use)
