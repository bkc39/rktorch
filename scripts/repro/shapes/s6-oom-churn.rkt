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
(define held
  (with-default-device dev (for/list ([_ (in-range 300)]) (randn 128 128))))
(define ooms 0)
(define others 0)
(with-default-device dev
  (for ([_ (in-range 150)])
    (with-handlers ([exn:fail:rktorch:oom? (lambda (_) (set! ooms (add1 ooms)))]
                    [exn:fail? (lambda (_) (set! others (add1 others)))])
      ;; 2^60 floats: beyond ADDRESS SPACE, so every backend refuses it
      ;; outright and it classifies as OOM -> collect-and-drain! ->
      ;; cuda/mps empty-cache -> one retry -> typed exn.  A merely huge size
      ;; (e.g. 1e12) is worse than useless here: the host accepts the
      ;; reservation and the process is SIGKILLed while zero-filling it,
      ;; before the classifier ever runs.
      (void (zeros 1152921504606846976)))
    (void (matmul (randn 256 256) (randn 256 256)))))
(printf "ooms=~a other=~a\n" ooms others)
(length held)
(finalizer-failures)
(native-memory-use)
