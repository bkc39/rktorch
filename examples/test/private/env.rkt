#lang racket/base

(provide env-number)

(define (env-number name default)
  (define supplied (getenv name))
  (define n (and supplied (string->number supplied)))
  (cond
    [(not supplied) default]
    [(real? n) n]
    [else
     (raise-user-error 'env-number "~a is not a real number: ~s" name supplied)]))

(module+ test
  (require (only-in rackunit check-equal? check-exn))
  (define (read-env name supplied default)
    (parameterize ([current-environment-variables (make-environment-variables)])
      (when supplied
        (putenv name supplied))
      (env-number name default)))
  (check-equal? (read-env "EPOCHS" #f 3) 3)
  (check-equal? (read-env "LIMIT" #f #f) #f)
  (check-equal? (read-env "EPOCHS" "0" 3) 0)
  (check-equal? (read-env "EPOCHS" "0" #f) 0)
  (check-equal? (read-env "EPOCHS" "12" 3) 12)
  (check-equal? (read-env "TEMPERATURE" "0.5" 0.8) 0.5)
  (for ([bad (in-list '(("EPOCHS" . "three")
                        ("EPOCHS" . "")
                        ("SEED" . " 4")
                        ("TEMPERATURE" . "1+2i")))])
    (define expected
      (format "env-number: ~a is not a real number: ~s" (car bad) (cdr bad)))
    (check-exn (lambda (e)
                 (and (exn:fail:user? e) (equal? (exn-message e) expected)))
               (lambda () (read-env (car bad) (cdr bad) 3))
               expected)))
