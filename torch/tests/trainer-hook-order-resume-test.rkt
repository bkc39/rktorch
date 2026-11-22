#lang racket/base

(module+ test
  (require rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/hook-order-kit.rkt" check-events epoch-saver fit-scenario)
           (only-in "private/trainer-kit.rkt" recorder Recorded scenario toy-loader))

  (define (saved-epochs dir)
    (fit-scenario (scenario) #:callbacks (list (epoch-saver dir)))
    (build-path dir "epoch=1-step=6.ckpt"))

  (test-case "T3-L16, T3 LO-10: the save-resume scenario"
    (with-temporary-directory (dir #:template "resume-order-~a")
      (define path (saved-epochs dir))
      (define sc (scenario))
      (fit-scenario sc #:max-epochs 4 #:ckpt-path path #:callbacks (list (epoch-saver dir)))
      (check-events 'save-resume sc)))

  (test-case "T3-L16, T3 LO-11: the test-ckpt scenario"
    (with-temporary-directory (dir #:template "test-order-~a")
      (define path (saved-epochs dir))
      (define sc (scenario))
      (void (test (Recorded sc) (toy-loader 4) #:logger #f #:ckpt-path path
                  #:callbacks (list (recorder sc))))
      (check-events 'test-ckpt sc))))
