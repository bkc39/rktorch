#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" * device randn)
           (only-in "../nn.rkt" Linear Sequential mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" check-exn/contract)
           (only-in "private/comparable.rkt" host-bytes layer-bytes)
           (only-in "private/trainer-kit.rkt" call-with-logged-warnings toy-loader))

  (define observed (box #f))

  (define-trainer Shaped (net scale enc)
    (define-init (hidden #:scale [s 1.0] #:encoder [encoder (Linear 2 2)])
      (set! net (Sequential (Linear 2 hidden) (Linear hidden 1)))
      (set! scale s)
      (set! enc encoder))
    (define-hparams scale)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-forward (x) (* (net (enc x)) scale))
    (define-training-step (x y) (mse-loss (net (enc x)) y))
    (define-hook (on-load-checkpoint _ckpt)
      (set-box! observed (list (run-stage run) (run-epoch run) (run-global-step run)
                               (run-optimizers run)))))

  (define-trainer Wider (net scale enc head)
    (define-init (hidden #:scale [s 1.0] #:encoder [encoder (Linear 2 2)])
      (set! net (Sequential (Linear 2 hidden) (Linear hidden 1)))
      (set! scale s)
      (set! enc encoder)
      (set! head (Linear 1 1)))
    (define-hparams scale)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net (enc x)) y)))

  (define (trained dir)
    (define path (build-path dir "shaped.ckpt"))
    (define tr (Shaped 3 #:scale 2.0))
    (set-hparam! tr 'scale 0.5)
    (void (fit tr (toy-loader 4) #:max-epochs 2 #:logger #f
               #:callbacks (list (lambda-callback
                                  #:on-train-end (lambda (r) (save-checkpoint! r path))))))
    (values tr path))

  (define probe (randn 5 2))

  (define (an-encoder) (hasheq 'encoder (Linear 2 2)))

  (test-case "T3-L22, LO-12: the loaded trainer answers as the saved one"
    (with-temporary-directory (dir #:template "load-~a")
      (define-values (tr path) (trained dir))
      (set-box! observed #f)
      (define back (load-from-checkpoint Shaped path #:init-args (an-encoder)))
      (check-equal? (host-bytes (back probe)) (host-bytes (tr probe)) "forward")
      (check-equal? (trainer-hparams back) (trainer-hparams tr))
      (check-equal? (trainer-hparams back) (hasheq 'scale 0.5) "the hparams, not the init's 2.0")
      (check-equal? (layer-bytes back) (layer-bytes tr))
      (define tree (read-checkpoint path))
      (check-equal? (unbox observed)
                    (list #f (hash-ref tree "epoch") (hash-ref tree "global_step") '())
                    "on-load-checkpoint sees a run that has not started")))

  (test-case "T3-L22: keyword and positional init args are reapplied, init-args replace them"
    (with-temporary-directory (dir #:template "init-~a")
      (define-values (_tr path) (trained dir))
      (define seen (box #f))
      (define (recording hidden #:scale s #:encoder e)
        (set-box! seen (list hidden s))
        (Shaped hidden #:scale s #:encoder e))
      (void (load-from-checkpoint recording path #:init-args (an-encoder)))
      (check-equal? (unbox seen) '(3 2.0))
      (check-exn #rx"size|shape"
                 (lambda () (load-from-checkpoint Shaped path
                                                  #:init-args (hash-set (an-encoder) 'hidden 4)))
                 "a replaced width no longer fits the file")))

  (test-case "T3-L22: a layer argument comes only from #:init-args"
    (with-temporary-directory (dir #:template "layer-~a")
      (define-values (_tr path) (trained dir))
      (check-exn #rx"encoder" (lambda () (load-from-checkpoint Shaped path)))
      (check-not-exn (lambda () (load-from-checkpoint Shaped path #:init-args (an-encoder))))
      (check-equal? (hash-ref (read-checkpoint path) "init_dropped") '("encoder"))))

  (test-case "T3-L22: #:device places the trainer, #:strict? governs a mismatch"
    (with-temporary-directory (dir #:template "strict-~a")
      (define-values (_tr path) (trained dir))
      (define cpu (device 'cpu))
      (define back (load-from-checkpoint Shaped path #:device cpu #:init-args (an-encoder)))
      (for ([p (in-list (parameters back))])
        (check-equal? (device p) cpu))
      (check-exn #rx"head" (lambda () (load-from-checkpoint Wider path #:init-args (an-encoder))))
      (check-not-exn
       (lambda () (load-from-checkpoint Wider path #:strict? #f #:init-args (an-encoder))))))

  (define-trainer Unscaled (net enc)
    (define-init (hidden #:scale [_s 1.0] #:encoder [encoder (Linear 2 2)])
      (set! net (Sequential (Linear 2 hidden) (Linear hidden 1)))
      (set! enc encoder))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net (enc x)) y)))

  (test-case "a saved hparam the trainer lacks is skipped with one warning, as Lightning's"
    (with-temporary-directory (dir #:template "hparam-~a")
      (define-values (_tr path) (trained dir))
      (for ([strict? (in-list '(#t #f))])
        (define-values (back warnings)
          (call-with-logged-warnings
           (lambda ()
             (load-from-checkpoint Unscaled path #:strict? strict? #:init-args (an-encoder)))))
        (check-equal? (trainer-hparams back) (hasheq))
        (check-equal? (length warnings) 1 (format "strict? ~a" strict?))
        (check-regexp-match #rx"scale.*shaped[.]ckpt" (car warnings)))))

  (define-trainer Spread (net)
    (define-init (width . more) (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (test-case "a rest formal's #:init-args value must be a list"
    (with-temporary-directory (dir #:template "rest-~a")
      (define path (build-path dir "rest.ckpt"))
      (void (fit (Spread 1 'a) (toy-loader 4) #:max-epochs 1 #:logger #f
                 #:callbacks (list (lambda-callback
                                    #:on-train-end (lambda (r) (save-checkpoint! r path))))))
      (check-exn/contract #rx"load-from-checkpoint:.*list.*formal: 'more"
                          (lambda ()
                            (load-from-checkpoint Spread path #:init-args (hasheq 'more 5))))
      (check-not-exn
       (lambda () (load-from-checkpoint Spread path #:init-args (hasheq 'more '(1 2)))))))

  (define given (box #f))

  (define-trainer Old (net)
    (define-init (width [depth 1] #:rate [r 0.1] #:tag [tag 'a] . more)
      (set-box! given (list width depth more))
      (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Narrow (net)
    (define-init (width #:rate [r 0.1]) (set-box! given (list width r)) (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Plain (net)
    (define-init (width) (set-box! given (list width)) (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Renamed (net)
    (define-init (width #:lr r) (set-box! given (list width r)) (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Taller (net)
    (define-init (width height) (set! net (Linear 2 (* width height))))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Keyed (net)
    (define-init (width #:depth [d 1] #:rate [r 0.1])
      (set-box! given (list width d r))
      (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer Optional (net)
    (define-init (width [depth 1]) (set-box! given (list width depth)) (set! net (Linear 2 width)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define (saved-from tr dir)
    (define path (build-path dir "old.ckpt"))
    (void (fit tr (toy-loader 2) #:max-epochs 1 #:logger #f
               #:callbacks (list (lambda-callback
                                  #:on-train-end (lambda (r) (save-checkpoint! r path))))))
    path)

  (define (loaded make-trainer path #:init-args [init-args (hasheq)])
    (set-box! given #f)
    (define-values (_tr warnings)
      (call-with-logged-warnings
       (lambda () (load-from-checkpoint make-trainer path #:init-args init-args))))
    (values (unbox given) warnings))

  (test-case "saved init arguments the constructor does not take are dropped with one warning"
    (with-temporary-directory (dir #:template "narrow-~a")
      (define path (saved-from (Old 1 2 #:rate 1/3 #:tag 'b 'x 2.5) dir))
      (define-values (args warnings) (loaded Narrow path))
      (check-equal? args '(1 1/3))
      (check-equal? (length warnings) 1)
      (check-regexp-match #rx"takes no positional depth, keyword #:tag, rest more saved in .*old[.]ckpt"
                          (car warnings))))

  (test-case "a rest formal the constructor does not take is dropped"
    (with-temporary-directory (dir #:template "rest-drop-~a")
      (define path (saved-from (Spread 1 'a 'b) dir))
      (define-values (args warnings) (loaded Plain path))
      (check-equal? args '(1))
      (check-equal? (length warnings) 1)
      (check-regexp-match #rx"constructor takes no rest more saved in .*old[.]ckpt" (car warnings))))

  (test-case "a saved rest formal's values do not fill optional positionals, as Lightning's *args"
    (with-temporary-directory (dir #:template "rest-optional-~a")
      (define path (saved-from (Spread 1 5) dir))
      (define-values (args warnings) (loaded Optional path))
      (check-equal? args '(1 1))
      (check-equal? (length warnings) 1)
      (check-regexp-match #rx"takes no rest more saved in" (car warnings))))

  (test-case "saved rest values never reach a constructor that takes a rest formal, as Lightning's"
    (with-temporary-directory (dir #:template "rest-rest-~a")
      (define path (saved-from (Spread 1 5) dir))
      (define-values (args warnings) (loaded Old path))
      (check-equal? args '(1 1 ()))
      (check-equal? (length warnings) 1)
      (check-regexp-match #rx"takes no rest more saved in" (car warnings))
      (define-values (given-args none) (loaded Old path #:init-args (hasheq 'more '(5))))
      (check-equal? given-args '(1 5 ()) "an #:init-args rest value follows the saved positionals")
      (check-equal? none '())))

  (test-case "a positional that became a keyword keeps its default; the warning says positional"
    (with-temporary-directory (dir #:template "became-keyword-~a")
      (define path (saved-from (Old 1 3 #:rate 1/3) dir))
      (define-values (args warnings) (loaded Keyed path))
      (check-equal? args '(1 1 1/3))
      (check-equal? (length warnings) 1)
      (check-regexp-match #rx"takes no positional depth, keyword #:tag saved in" (car warnings))))

  (test-case "a positional count the constructor's arity leaves out raises the named error"
    (with-temporary-directory (dir #:template "gap-~a")
      (define path (saved-from (Taller 1 2) dir))
      (check-exn/contract #rx"load-from-checkpoint:.*wrap.*saved: 2.*old[.]ckpt"
                          (lambda ()
                            (load-from-checkpoint (case-lambda [(w) (Plain w)] [(w _a _b) (Plain w)])
                                                  path)))))

  (test-case "a renamed keyword: the old one is dropped, the new one comes from #:init-args"
    (with-temporary-directory (dir #:template "renamed-~a")
      (define path (saved-from (Narrow 1 #:rate 1/3) dir))
      (check-exn/contract #rx"load-from-checkpoint:.*formal: '#:lr.*old[.]ckpt"
                          (lambda () (load-from-checkpoint Renamed path)))
      (define-values (args warnings) (loaded Renamed path #:init-args (hasheq 'lr 0.2)))
      (check-equal? args '(1 0.2))
      (check-regexp-match #rx"#:rate" (car warnings))
      (check-exn/contract #rx"load-from-checkpoint:.*positional.*wrap.*saved: 1.*old[.]ckpt"
                          (lambda () (load-from-checkpoint Taller path)))))

  (define (saved-with dir name edit)
    (define path (build-path dir name))
    (void (fit (Plain 1) (toy-loader 2) #:max-epochs 1 #:logger #f
               #:callbacks (list (lambda-callback
                                  #:on-save-checkpoint (lambda (_r ckpt) (edit ckpt))
                                  #:on-train-end (lambda (r) (save-checkpoint! r path))))))
    path)

  (test-case "hparams a hook removed load as empty, as Lightning's .get(..., {})"
    (with-temporary-directory (dir #:template "removed-~a")
      (define path
        (saved-with dir "bare.ckpt"
                    (lambda (ckpt)
                      (hash-remove! ckpt "hyper_parameters")
                      (hash-remove! ckpt "trainer_hparams"))))
      (check-exn/contract #rx"load-from-checkpoint:.*formal: 'width.*bare[.]ckpt"
                          (lambda () (load-from-checkpoint Plain path)))
      (define back (load-from-checkpoint Plain path #:init-args (hasheq 'width 1)))
      (check-equal? (trainer-hparams back) (hasheq))))

  (test-case "missing or malformed init formals raise naming the entry and the path"
    (with-temporary-directory (dir #:template "formals-~a")
      (for ([edit (list (lambda (ckpt) (hash-remove! ckpt "init_formals"))
                        (lambda (ckpt) (hash-set! ckpt "init_formals" 5))
                        (lambda (ckpt) (hash-set! ckpt "init_formals" '(("keyword" "rate" "r"))))
                        (lambda (ckpt) (hash-set! ckpt "epoch" "one")))]
            [i (in-naturals)])
        (define path (saved-with dir (format "f~a.ckpt" i) edit))
        (check-exn/contract
         #rx"^load-from-checkpoint: the checkpoint's entry is missing.*f[0-9][.]ckpt"
         (lambda () (load-from-checkpoint Plain path))
         (format "edit ~a" i))))))
