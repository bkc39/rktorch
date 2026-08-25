;; Body only. `dev` comes from ../shape-prelude.rkt, which exit-loop.sh
;; prepends; running this file on its own will not define it.
(define held
  (with-default-device dev (for/list ([_ (in-range 500)]) (randn 128 128))))
(with-default-device dev (for ([_ (in-range 3000)]) (void (randn 256 256))))
(with-default-device dev (randn 512 512))
(length held)
(finalizer-failures)
(native-memory-use)
