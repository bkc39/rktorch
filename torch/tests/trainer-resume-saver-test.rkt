#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" generator-state set-generator-state!)
           "../train.rkt"
           (only-in "private/resume-harness.rkt"
                    check-resume fixture-fit-args training-world world world-callbacks
                    world-generators))

  (define-callback epoch-end-saver (dir)
    #:kind checkpoint
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (build-path dir (format "epoch=~a.ckpt" (run-epoch run))))))

  (define (last-epoch-file _saver r dir)
    (build-path dir (format "epoch=~a.ckpt" (sub1 (run-epoch r)))))

  (define (generator-keeper g)
    (lambda-callback #:state-key "train-generator"
                     #:state-dict (lambda (_r) (generator-state g))
                     #:load-state! (lambda (_r state) (set-generator-state! g state))))

  (define (kept-generator-world dir)
    (define w (training-world dir #:generator? #t))
    (struct-copy world w
                 [callbacks (append (world-callbacks w)
                                    (map generator-keeper (world-generators w)))]))

  (define (check-saver-resume make-world splits)
    (check-resume make-world #:split splits #:total 4 #:fit-args fixture-fit-args
                  #:saver epoch-end-saver #:resume-path last-epoch-file))

  (test-case "T3-L1: 2 epochs, resume, 2 more equals 4 straight, the loader on the default stream"
    (check-saver-resume training-world '(2)))

  (test-case "T3-L1: 2 epochs, resume, 2 more equals 4 straight, the loader on its own generator"
    (check-saver-resume kept-generator-world '(2)))

  (test-case "T3-L2: 1+1+1+1 resumes equal 4 straight"
    (check-saver-resume training-world '(1 2 3))
    (check-saver-resume kept-generator-world '(1 2 3))))
