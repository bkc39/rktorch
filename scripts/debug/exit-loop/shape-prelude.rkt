;; Prepended to every shape by exit-loop.sh; not a shape itself.
;; Binds `dev` from REPRO_DEVICE and aborts if that device is unavailable,
;; rather than silently degrading to CPU and reporting a meaningless clean run.
(require torch)
(define requested (string-downcase (or (getenv "REPRO_DEVICE") "cpu")))
(define dev
  (case requested
    [("cuda") (if (cuda-available?) (cuda-device) 'unavailable)]
    [("mps")  (if (mps-available?)  (mps-device)  'unavailable)]
    [("cpu") (cpu-device)]
    [else 'unknown]))
(when (eq? dev 'unknown)
  (printf "REPRO-DEVICE-UNKNOWN ~a\n" requested)
  (exit 3))
(when (eq? dev 'unavailable)
  (printf "REPRO-DEVICE-UNAVAILABLE ~a\n" requested)
  (exit 3))
(printf "REPRO device=~a\n" dev)
