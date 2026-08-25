;; Body only. `dev` comes from ../shape-prelude.rkt, which exit-loop.sh
;; prepends; running this file on its own will not define it.
(define held
  (with-default-device dev (for/list ([_ (in-range 2000)]) (randn 128 128))))
(length held)
(finalizer-failures)
