#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/runtime-path #%datum define-runtime-path))

(provide (struct-out example)
         examples
         twins
         find-examples
         example-env
         example-runner)

(define-runtime-path examples-dir "../../examples/test")
(define-runtime-path workloads-dir "workloads")
(define-runtime-path python-dir "python")

(struct example (name short full epochs-var warmup optimizer) #:transparent)

(define two-epochs-of-128 '(("EPOCHS" . 2) ("BATCH" . 128)))

(define examples
  (list (example "05-mnist" '(("EPOCHS" . 3)) '(("EPOCHS" . 3)) "EPOCHS" 0 'first)
        (example "06-gpt" '(("STEPS" . 300)) '(("STEPS" . 2000)) #f 50 'first)
        (example "08-diffusion" '(("EPOCHS" . 2)) '(("EPOCHS" . 10)) "EPOCHS" 0 'first)
        (example "09-resnet" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "10-dcgan" '(("EPOCHS" . 3)) '(("EPOCHS" . 5)) "EPOCHS" 0 'first)
        (example "11-vae" '(("EPOCHS" . 3)) '(("EPOCHS" . 10)) "EPOCHS" 0 'first)
        (example "12-char-rnn" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "13-translation" '(("EPOCHS" . 3)) '(("EPOCHS" . 30)) "EPOCHS" 0 'first)
        (example "15-finetune"
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 3))
                 '(("FEATURE_EPOCHS" . 0) ("FINETUNE_EPOCHS" . 15))
                 "FINETUNE_EPOCHS" 0 'last)
        (example "16-style-transfer" '(("STEPS" . 100)) '(("STEPS" . 1000)) #f 20 'first)
        (example "loader" '(("EPOCHS" . 2)) '(("EPOCHS" . 5)) "EPOCHS" 0 'first)
        (example "17-workers" '(("EPOCHS" . 2)) '(("EPOCHS" . 3)) "EPOCHS" 0 'first)
        (example "cifar-loader" two-epochs-of-128 two-epochs-of-128 "EPOCHS" 0 'first)
        (example "cifar-resnet" two-epochs-of-128 '(("EPOCHS" . 3) ("BATCH" . 128))
                 "EPOCHS" 0 'first)))

(define (twin name-of-example script #:per-param? [per-param? #t])
  (define ex (for/first ([ex (in-list examples)] #:when (equal? (example-name ex) name-of-example))
               ex))
  (define per-param '(("OPTIMIZER" . "per-param")))
  (define name (string-append "py-" (example-name ex)))
  (cons (cons (struct-copy example ex [name name]) script)
        (if per-param?
            (list (cons (struct-copy example ex
                                     [name (string-append name "-per-param")]
                                     [short (append (example-short ex) per-param)]
                                     [full (append (example-full ex) per-param)])
                        script))
            '())))

(define twin-runners
  (append (twin "05-mnist" "mnist.py")
          (twin "09-resnet" "resnet.py")
          (twin "17-workers" "workers.py")
          (twin "cifar-loader" "cifar_loader.py" #:per-param? #f)
          (twin "cifar-resnet" "cifar_resnet.py" #:per-param? #f)))

(define twins (map car twin-runners))

(define runners
  (for/fold ([h (hash "loader" (build-path workloads-dir "loader.rkt")
                      "17-workers" (build-path workloads-dir "workers.rkt")
                      "cifar-loader" (build-path workloads-dir "cifar-loader.rkt")
                      "cifar-resnet" (build-path workloads-dir "cifar-resnet.rkt"))])
            ([t (in-list twin-runners)])
    (hash-set h (example-name (car t)) (build-path python-dir (cdr t)))))

(define (find-examples names)
  (for/list ([n (in-list names)])
    (or (for/first ([ex (in-sequences (in-list examples) (in-list twins))]
                    #:when (regexp-match? (regexp (string-append "^" (regexp-quote n)))
                                          (example-name ex)))
          ex)
        (raise-user-error 'e2e "no example named ~a" n))))

(define (example-env ex scale)
  (if (eq? scale 'full) (example-full ex) (example-short ex)))

(define (example-runner ex)
  (hash-ref runners (example-name ex)
            (lambda () (build-path examples-dir (string-append (example-name ex) ".rkt")))))

(module+ test
  (require rackunit)

  (test-case "examples are found by their number"
    (check-equal? (map example-name (find-examples '("05" "16")))
                  '("05-mnist" "16-style-transfer"))
    (check-exn #rx"no example named 99" (lambda () (find-examples '("99"))))
    (check-true (file-exists? (example-runner (car (find-examples '("loader")))))
                "a workload's runner is found in its own directory")
    (check-equal? (example-env (car examples) 'full) '(("EPOCHS" . 3)))
    (check-equal? (cdr (assoc "STEPS" (example-env (cadr examples) 'short))) 300)
    (check-equal? (map example-name (find-examples '("py-05" "17")))
                  '("py-05-mnist" "17-workers"))
    (for ([ex (in-list (append examples twins))])
      (check-true (file-exists? (example-runner ex)) (example-name ex))))

  (test-case "a twin has a per-param form unless it trains no optimizer of its own"
    (check-equal? (map example-name (find-examples '("py-09-resnet-per" "py-cifar")))
                  '("py-09-resnet-per-param" "py-cifar-loader"))
    (check-exn #rx"no example named py-cifar-loader-per"
               (lambda () (find-examples '("py-cifar-loader-per"))))))
