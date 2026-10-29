#lang racket/base

(module+ test
  (require (only-in json string->jsexpr)
           (only-in racket/file
                    delete-directory/files file->lines file->string file->value
                    make-temporary-directory)
           (only-in racket/string string-prefix?)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" ones)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           "../train.rkt")

  (define-trainer Line (net rate)
    (define-init (r) (set! net (Linear 1 1)) (set! rate r))
    (define-hparams rate)
    (define-optimizers (sgd (parameters net) #:lr rate))
    (define-training-step (x y)
      (define loss (mse-loss (net x) y))
      (log! 'loss loss #:on-step? #t #:on-epoch? #f)
      loss))

  (define (fit-logged logger)
    (fit (Line 0.5) (dataloader (tensor-dataset (ones 2 1) (ones 2 1)) #:batch-size 1)
         #:max-epochs 1 #:log-every-n-steps 1 #:logger logger))

  (define (call-with-scratch proc)
    (define dir (make-temporary-directory))
    (dynamic-wind void (lambda () (proc dir)) (lambda () (delete-directory/files dir))))

  (define hparams (hasheq 'r 0.5 'rate 0.5))

  (test-case "stdout-logger prints the hparams, each step's metrics and the status"
    (define out (open-output-string))
    (fit-logged (stdout-logger #:out out))
    (define lines (regexp-split #rx"\n" (get-output-string out)))
    (check-equal? (car lines) "hparams r=0.5 rate=0.5")
    (check-regexp-match #rx"^step 0 epoch=0 loss=[0-9.e-]+$" (cadr lines))
    (check-regexp-match #rx"^step 1 epoch=0 loss=[0-9.e-]+$" (caddr lines))
    (check-equal? (cdddr lines) '("success" "")))

  (test-case "jsonl-logger writes a JSON line per read and the hparams file"
    (call-with-scratch
     (lambda (dir)
       (define metrics (build-path dir "metrics.jsonl"))
       (define (steps) (map (lambda (line) (hash-ref (string->jsexpr line) 'step))
                            (file->lines metrics)))
       (fit-logged (jsonl-logger dir))
       (check-equal? (steps) '(0 1))
       (check-true (andmap (lambda (line) (real? (hash-ref (string->jsexpr line) 'loss)))
                           (file->lines metrics)))
       (check-equal? (file->value (build-path dir "hparams.rktd")) hparams)
       (check-exn exn:fail:filesystem:exists? (lambda () (jsonl-logger dir)))
       (fit-logged (jsonl-logger dir #:exists 'append))
       (check-equal? (steps) '(0 1 0 1))
       (fit-logged (jsonl-logger dir #:exists 'replace))
       (check-equal? (steps) '(0 1)))))

  (test-case "jsonl-logger writes NaN and infinities as strings"
    (call-with-scratch
     (lambda (dir)
       (logger-log-metrics! (jsonl-logger dir) (hasheq 'x +nan.0 'y -inf.0) 3)
       (check-equal? (map string->jsexpr (file->lines (build-path dir "metrics.jsonl")))
                     (list (hasheq 'step 3 'x "+nan.0" 'y "-inf.0"))))))

  (test-case "csv-logger writes its header once, whatever #:exists says"
    (call-with-scratch
     (lambda (dir)
       (define metrics (build-path dir "metrics.csv"))
       (fit-logged (csv-logger dir))
       (fit-logged (csv-logger dir #:exists 'append))
       (define lines (file->lines metrics))
       (check-equal? (car lines) "step,key,value")
       (check-equal? (length lines) 9)
       (check-equal? (length (filter (lambda (l) (string-prefix? l "step,")) lines)) 1)
       (check-equal? (file->value (build-path dir "hparams.rktd")) hparams)
       (check-exn exn:fail:filesystem:exists? (lambda () (csv-logger dir)))
       (fit-logged (csv-logger dir #:exists 'replace))
       (check-equal? (length (file->lines metrics)) 5))))

  (test-case "csv-logger quotes a key holding a delimiter, a quote or a newline"
    (call-with-scratch
     (lambda (dir)
       (logger-log-metrics! (csv-logger dir)
                            (hasheq '|acc,top-5| 0.9 '|say "hi"| 1.0 (string->symbol "a\nb") 2.0)
                            3)
       (check-equal? (file->string (build-path dir "metrics.csv"))
                     (string-append "step,key,value\n"
                                    "3,\"a\nb\",2.0\n"
                                    "3,\"acc,top-5\",0.9\n"
                                    "3,\"say \"\"hi\"\"\",1.0\n")))))

  (struct unprintable ()
    #:property prop:custom-write (lambda (v out mode) (error 'unprintable "cannot print")))

  (test-case "a file logger closes its port when writing escapes"
    (call-with-scratch
     (lambda (dir)
       (define owner (make-custodian))
       (parameterize ([current-custodian owner])
         (check-exn #rx"cannot print"
                    (lambda ()
                      (logger-log-hyperparams! (jsonl-logger dir) (hasheq 'x (unprintable))))))
       (check-equal? (filter port? (custodian-managed-list owner (current-custodian))) '())))))
