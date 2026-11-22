#lang racket/base

(module+ test
  (require rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/trainer-kit.rkt" apply-keywords toy-loader))

  (define blames-this-test? (blames-caller? "checkpoint-contract-test"))

  (define-trainer Tiny (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-test-step (x y) (mse-loss (net x) y))
    (define-predict-step (x _y) (net x)))

  (define not-implemented? #rx"#:ckpt-path '(last|best): not implemented until T3 \\(#386\\)")

  (test-case "T3-L21: a #:ckpt-path that is no path, 'last or 'best blames the caller"
    (check-exn blames-this-test?
               (lambda () (fit (Tiny) (toy-loader 4) #:max-epochs 1 #:ckpt-path 3)))
    (check-exn blames-this-test? (lambda () (test (Tiny) (toy-loader 4) #:ckpt-path 'latest)))
    (check-exn blames-this-test? (lambda () (predict (Tiny) (toy-loader 4) #:ckpt-path #t))))

  (test-case "#:ckpt-path accepts 'last and 'best, which are not implemented yet"
    (define (unsupported? e)
      (and (exn:fail:unsupported? e) (regexp-match? not-implemented? (exn-message e))))
    (for ([path (list 'last 'best)])
      (check-exn unsupported?
                 (lambda () (fit (Tiny) (toy-loader 4) #:max-epochs 1 #:logger #f
                                 #:ckpt-path path)))
      (check-exn unsupported?
                 (lambda () (test (Tiny) (toy-loader 4) #:logger #f #:ckpt-path path)))
      (check-exn unsupported?
                 (lambda () (predict (Tiny) (toy-loader 4) #:ckpt-path path)))))

  (test-case "T3-L21: save-checkpoint!, read-checkpoint and load-from-checkpoint blame the caller"
    (check-exn blames-this-test? (lambda () (save-checkpoint! (Tiny) "a.ckpt")))
    (check-exn blames-this-test? (lambda () (read-checkpoint 'a.ckpt)))
    (check-exn blames-this-test? (lambda () (load-from-checkpoint 'Tiny "a.ckpt")))
    (check-exn blames-this-test?
               (lambda () (load-from-checkpoint Tiny "a.ckpt" #:init-args (make-hasheq))))
    (check-exn blames-this-test? (lambda () (load-from-checkpoint Tiny "a.ckpt" #:device 'gpu))))

  (test-case "T3-L21: model-checkpoint's keywords and preconditions blame the caller"
    (check-exn exn:fail:contract? (lambda () (model-checkpoint)) "#:dirpath is required")
    (check-exn blames-this-test?
               (lambda () (model-checkpoint #:dirpath "ck" #:every-n-train-steps 2
                                            #:every-n-epochs 1)))
    (check-exn #rx"mutually exclusive"
               (lambda () (model-checkpoint #:dirpath "ck" #:every-n-train-steps 2
                                            #:every-n-epochs 1)))
    (check-exn blames-this-test? (lambda () (model-checkpoint #:dirpath "ck" #:save-top-k 2)))
    (check-exn #rx"without #:monitor"
               (lambda () (model-checkpoint #:dirpath "ck" #:monitor #f #:save-top-k 2)))
    (check-exn blames-this-test? (lambda () (model-checkpoint #:dirpath "ck" #:save-top-k -2)))
    (check-exn blames-this-test? (lambda () (model-checkpoint #:dirpath "ck" #:filename "{x")))
    (check-exn blames-this-test? (lambda () (model-checkpoint #:dirpath "ck" #:mode 'median)))
    (check-exn blames-this-test? (lambda () (model-checkpoint #:dirpath 3)))
    (check-exn blames-this-test? (lambda () (on-exception-checkpoint #:dirpath "ck" #:filename "")))
    (check-exn blames-this-test? (lambda () (model-checkpoint-best-model-path (Tiny)))))

  (test-case "the triggers do not conflict when one of them is 0 or #f"
    (for ([kws (list '((#:every-n-train-steps . 0) (#:every-n-epochs . 1))
                     '((#:every-n-train-steps . 2) (#:every-n-epochs . 0))
                     '((#:every-n-train-steps . #f) (#:every-n-epochs . 3)))])
      (check-exn exn:fail:unsupported?
                 (lambda () (apply-keywords model-checkpoint (cons '(#:dirpath . "ck") kws)))
                 (format "~s" kws)))
    (check-exn exn:fail:unsupported?
               (lambda () (model-checkpoint #:dirpath "ck" #:monitor 'val-loss #:save-top-k 3))))

  (test-case "the checkpoint callbacks raise exn:fail:unsupported until T3"
    (for ([call (list (lambda () (on-exception-checkpoint #:dirpath "ck")))])
      (check-exn (lambda (e) (and (exn:fail:unsupported? e)
                                  (regexp-match? #rx"not implemented until T3 \\(#386\\)"
                                                 (exn-message e))))
                 call))))
