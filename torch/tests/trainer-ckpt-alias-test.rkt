#lang racket/base

(module+ test
  (require (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" Small)
           (only-in "private/trainer-kit.rkt" call-with-logged-warnings toy-loader))

  (define (resumed-from mc epochs path #:more [more '()])
    (define step (box #f))
    (define start (box #f))
    (void (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs epochs #:logger #f
               #:ckpt-path path
               #:callbacks (append (list (lambda-callback
                                          #:on-load-checkpoint
                                          (lambda (_r ckpt)
                                            (set-box! step (hash-ref ckpt "global_step")))
                                          #:on-train-start
                                          (lambda (r) (set-box! start (run-epoch r))))
                                         mc)
                                   more)))
    (list (unbox step) (unbox start)))

  (test-case "T3-L24: 'best resumes from the model-checkpoint's best path"
    (with-temporary-directory (dir #:template "best-~a")
      (define mc (model-checkpoint #:dirpath dir #:monitor 'val-loss))
      (void (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 3 #:logger #f
                 #:callbacks (list mc)))
      (define best (read-checkpoint (model-checkpoint-best-model-path mc)))
      (check-equal? (car (resumed-from mc 4 'best)) (hash-ref best "global_step"))))

  (test-case "T3-L24: 'best with no model-checkpoint, or two, raises"
    (with-temporary-directory (dir #:template "best-count-~a")
      (check-exn #rx"model-checkpoint"
                 (lambda ()
                   (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f #:ckpt-path 'best)))
      (check-exn #rx"model-checkpoint"
                 (lambda ()
                   (fit (Small) (toy-loader 4) #:max-epochs 1 #:logger #f #:ckpt-path 'best
                        #:callbacks (list (model-checkpoint #:dirpath (build-path dir "a"))
                                          (model-checkpoint #:dirpath (build-path dir "b"))))))))

  (test-case "T3-L24: 'last resumes from the last file, and with none warns and starts over"
    (with-temporary-directory (dir #:template "last-~a")
      (define mc (model-checkpoint #:dirpath dir #:save-last #t))
      (void (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 2 #:logger #f
                 #:callbacks (list mc)))
      (check-equal? (resumed-from mc 3 'last) '(4 2))
      (define empty (build-path dir "empty"))
      (define-values (seen warnings)
        (call-with-logged-warnings
         (lambda ()
           (parameterize ([current-error-port (open-output-nowhere)])
             (resumed-from (model-checkpoint #:dirpath empty) 1 'last)))))
      (check-equal? seen '(#f 0) "no checkpoint loaded, training from epoch 0")
      (check-true (for/or ([w (in-list warnings)]) (regexp-match? #rx"last" w))))))
