(require torch)
(define held (with-default-device 'mps (for/list ([_ (in-range 300)]) (randn 128 128))))
(define oks 0)
(define ooms 0)
(with-default-device 'mps
  (for ([i (in-range 150)])
    (with-handlers ([exn:fail:rktorch:oom? (lambda (_) (set! ooms (add1 ooms)))]
                    [exn:fail? (lambda (_) (set! oks (add1 oks)))])
      ;; past Metal's per-buffer cap -> classified OOM -> retry -> mps empty-cache
      (void (zeros 1000000000000)))
    (void (matmul (randn 256 256) (randn 256 256)))))
(printf "ooms=~a other=~a\n" ooms oks)
(length held)
(finalizer-failures)
(native-memory-use)
