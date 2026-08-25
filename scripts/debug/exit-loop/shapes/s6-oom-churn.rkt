(define held
  (with-default-device dev (for/list ([_ (in-range 300)]) (randn 128 128))))
(define ooms 0)
(define others 0)
(with-default-device dev
  (for ([_ (in-range 150)])
    (with-handlers ([exn:fail:rktorch:oom? (lambda (_) (set! ooms (add1 ooms)))]
                    [exn:fail? (lambda (_) (set! others (add1 others)))])
      (void (zeros 1152921504606846976)))
    (void (matmul (randn 256 256) (randn 256 256)))))
(printf "ooms=~a other=~a\n" ooms others)
(unless (and (= ooms 150) (zero? others))
  (error 's6-oom-churn "expected 150 typed OOMs and 0 generic, got ~a/~a" ooms others))
(length held)
(finalizer-failures)
(native-memory-use)
