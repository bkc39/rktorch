;; Body only. `dev` comes from ../shape-prelude.rkt, which exit-loop.sh
;; prepends; running this file on its own will not define it.
(with-default-device dev (randn 512 512))
(with-default-device dev (randn 300 300))
(with-default-device dev (randn 64 64 64))
(with-default-device dev (arange 0 100000))
(finalizer-failures)
