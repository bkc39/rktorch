(require torch)
(with-default-device 'mps
  (for ([_ (in-range 5000)]) (void (randn 256 256))))
(finalizer-failures)
(native-memory-use)
