(require torch)
(define held (with-default-device 'mps (for/list ([_ (in-range 2000)]) (randn 128 128))))
(length held)
(finalizer-failures)
