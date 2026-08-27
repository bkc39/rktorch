(define held
  (with-default-device dev (for/list ([_ (in-range 2000)]) (randn 128 128))))
(length held)
(finalizer-failures)
