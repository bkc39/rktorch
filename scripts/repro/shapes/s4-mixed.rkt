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
  (with-default-device dev (for/list ([_ (in-range 500)]) (randn 128 128))))
(with-default-device dev (for ([_ (in-range 3000)]) (void (randn 256 256))))
(with-default-device dev (randn 512 512))
(length held)
(finalizer-failures)
(native-memory-use)
