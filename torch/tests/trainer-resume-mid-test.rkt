#lang racket/base

(module+ test
  (require rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/resume-harness.rkt"
                    check-resume fixture-fit-args fixture-world world-callbacks world-train
                    world-trainer world-val))

  (test-case "T3-L3: a split at step 3, mid-epoch, equals straight"
    (check-resume fixture-world #:split-step 3 #:total 4 #:fit-args fixture-fit-args))

  (test-case "T3-L3: a split at a validation-end save equals straight"
    (check-resume fixture-world #:split 2 #:total 4
                  #:fit-args (cons '(#:check-val-every-n-epoch . 2) fixture-fit-args)))

  (test-case "T3-L4: a mid-epoch checkpoint with a #:workers loader raises before any hook"
    (with-temporary-directory (dir #:template "workers-~a")
      (define path (build-path dir "mid.ckpt"))
      (define w (fixture-world dir))
      (void (fit (world-trainer w) (world-train w) #:val (world-val w) #:max-epochs 1 #:logger #f
                 #:callbacks (cons (lambda-callback
                                    #:on-train-batch-end
                                    (lambda (r . _)
                                      (when (= (run-global-step r) 3) (save-checkpoint! r path))))
                                   (world-callbacks w))))
      (define resumed (fixture-world dir #:workers 2))
      (define seen (box '()))
      (check-exn #rx"mid-epoch.*workers|workers.*mid-epoch"
                 (lambda ()
                   (fit (world-trainer resumed) (world-train resumed) #:max-epochs 2 #:logger #f
                        #:ckpt-path path
                        #:callbacks (list (lambda-callback
                                           #:on-fit-start (lambda (_r) (set-box! seen '(hook))))))))
      (check-equal? (unbox seen) '()))))
