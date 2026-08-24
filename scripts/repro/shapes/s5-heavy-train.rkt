(require torch)
(define requested (string-downcase (or (getenv "REPRO_DEVICE") "cpu")))
(define dev
  (case requested
    [("cuda") (if (cuda-available?) (cuda-device) 'unavailable)]
    [("mps")  (if (mps-available?)  (mps-device)  'unavailable)]
    [else (cpu-device)]))
(when (eq? dev 'unavailable)
  (printf "REPRO-DEVICE-UNAVAILABLE ~a\n" requested)
  (exit 3))
(printf "REPRO device=~a\n" dev)
(require torch/nn)
(define net
  (with-default-device dev
    (sequential (linear 256 512) (relu) (linear 512 512) (relu) (linear 512 10))))
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
(finalizer-failures)
(native-memory-use)
