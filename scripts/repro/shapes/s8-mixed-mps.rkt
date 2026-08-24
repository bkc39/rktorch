(require torch)
(define held
  (with-default-device 'mps
    (for/list ([_ (in-range 500)]) (randn 128 128))))
(with-default-device 'mps
  (for ([_ (in-range 3000)]) (void (randn 256 256))))
(with-default-device 'mps (randn 512 512))
(length held)
(finalizer-failures)
(native-memory-use)
