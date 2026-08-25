;; Body only. `dev` comes from ../shape-prelude.rkt, which exit-loop.sh
;; prepends; running this file on its own will not define it.
(with-default-device dev
  (for ([_ (in-range 5000)]) (void (randn 256 256))))
(finalizer-failures)
(native-memory-use)
