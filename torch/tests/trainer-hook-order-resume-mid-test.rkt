#lang racket/base

(module+ test
  (require rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "private/hook-order-kit.rkt" check-events fit-scenario step-saver)
           (only-in "private/trainer-kit.rkt" scenario))

  (test-case "T3-L16, T3 LO-13: the save-resume-mid scenario, with no epoch start on resume"
    (with-temporary-directory (dir #:template "mid-order-~a")
      (fit-scenario (scenario) #:callbacks (list (step-saver dir 2)))
      (define sc (scenario))
      (fit-scenario sc #:ckpt-path (build-path dir "epoch=0-step=2.ckpt")
                    #:callbacks (list (step-saver dir 2)))
      (check-events 'save-resume-mid sc))))
