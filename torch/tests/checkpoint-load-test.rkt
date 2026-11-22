#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" * device randn)
           (only-in "../nn.rkt" Linear Sequential mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
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
           (lambda () (load-from-checkpoint Unscaled path #:strict? strict? #:init-args (an-encoder)))))
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
      (check-exn (lambda (e) (and (exn:fail:contract? e)
                                  (regexp-match? #rx"load-from-checkpoint:.*list.*formal: 'more"
                                                 (exn-message e))))
                 (lambda () (load-from-checkpoint Spread path #:init-args (hasheq 'more 5))))
      (check-not-exn
       (lambda () (load-from-checkpoint Spread path #:init-args (hasheq 'more '(1 2))))))))
