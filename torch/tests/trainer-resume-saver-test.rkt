#lang racket/base

(module+ test
  (require rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" generator-state make-generator randn set-generator-state!)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt"
           (only-in "private/resume-harness.rkt"
                    check-resume fixture-fit-args training-world world world-callbacks
                    world-generators))

  (define-trainer Tagged (net tag)
    (define-init () (set! net (Linear 2 1)) (set! tag 0))
    (define-hparams tag)
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-train-epoch-start)
      (when (zero? (run-epoch run))
        (set! tag 1))))

  (define ((tagged-world callbacks) _dir)
    (define g (make-generator 5))
    (world (Tagged)
           (dataloader (tensor-dataset (randn 4 2 #:generator g) (randn 4 1 #:generator g))
                       #:batch-size 2)
           #f callbacks '() (box '())))

  (define tag-dropper
    (lambda-callback #:on-save-checkpoint
                     (lambda (_r tree)
                       (hash-update! tree "trainer_hparams" (lambda (h) (hash-remove h "tag"))))))

  (define (law-failures thunk)
    (define failures '())
    (define (note! e)
      (set! failures
            (cons (for/first ([info (in-list (exn:test:check-stack e))]
                              #:when (eq? (check-info-name info) 'message))
                    (check-info-value info))
                  failures)))
    (parameterize ([current-check-around
                    (lambda (check) (with-handlers ([exn:test:check? note!]) (check)))])
      (thunk))
    (reverse failures))

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
    (check-saver-resume kept-generator-world '(1 2 3)))

  (test-case "the resume law compares the hyperparameters, which training need not read"
    (define (tagged-resume callbacks)
      (law-failures
       (lambda ()
         (check-resume (tagged-world callbacks) #:split '(2) #:total 4
                       #:saver epoch-end-saver #:resume-path last-epoch-file))))
    (check-equal? (tagged-resume '()) '())
    (check-equal? (tagged-resume (list tag-dropper)) '("resume law: hparams"))))
