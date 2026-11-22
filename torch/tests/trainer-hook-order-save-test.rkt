#lang racket/base

(module+ test
  (require rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/hook-order-kit.rkt" check-events fit-scenario)
           (only-in "private/lightning-names.rkt" expected-events)
           (only-in "private/trainer-kit.rkt" scenario))

  (define-callback saver (dir weights-only?)
    #:kind checkpoint
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run
                        (build-path dir (format "epoch=~a-step=~a.ckpt"
                                                (run-epoch run) (run-global-step run)))
                        #:weights-only? weights-only?)))

  (test-case "T3-L16, T3 LO-9: the save scenario"
    (with-temporary-directory (dir #:template "save-order-~a")
      (define sc (scenario))
      (fit-scenario sc #:callbacks (list (saver dir #f)))
      (check-events 'save sc)))

  (test-case "T3 LO-9: a weights-only save runs no callback state-dict or on-save-checkpoint"
    (with-temporary-directory (dir #:template "weights-order-~a")
      (define sc (scenario))
      (fit-scenario sc #:callbacks (list (saver dir #t)))
      (check-events 'save-weights-only sc
                    #:expected
                    (filter (lambda (e) (not (and (eq? (car e) 'callback)
                                                  (memq (cadr e) '(state-dict on-save-checkpoint)))))
                            (expected-events 'save)))))

  (test-case "T3 LO-9: save-checkpoint! inside on-save-checkpoint raises"
    (with-temporary-directory (dir #:template "nested-~a")
      (define nested
        (lambda-callback #:on-save-checkpoint
                         (lambda (r _ckpt) (save-checkpoint! r (build-path dir "inner.ckpt")))))
      (define sc (scenario))
      (check-exn #rx"save-checkpoint!"
                 (lambda () (fit-scenario sc #:callbacks (list nested (saver dir #f)))))
      (check-false (file-exists? (build-path dir "inner.ckpt"))))))
