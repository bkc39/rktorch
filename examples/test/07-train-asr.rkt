#lang racket/base

(require (for-syntax racket/base)
         (only-in racket/match match-define)
         (only-in racket/runtime-path define-runtime-path)
         rackunit
         (only-in torch/private/util with-temporary-directory)
         (only-in "../../scripts/train-asr.rkt" env-number))

(define-runtime-path train-asr "../../scripts/train-asr.rkt")

(define (with-env settings thunk)
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([setting (in-list settings)])
    (match-define (cons name value) setting)
    (environment-variables-set! env (string->bytes/utf-8 name)
                                (and value (string->bytes/utf-8 value))))
  (parameterize ([current-environment-variables env])
    (thunk)))

(define ((env-violation? rx) e)
  (and (exn:fail? e) (regexp-match? rx (exn-message e))))

(test-case "env-number reads a number, or the default when the variable is unset"
  (with-env '(("EPOCHS" . #f))
    (lambda ()
      (check-equal? (env-number "EPOCHS" 20) 20)
      (check-equal? (env-number "EPOCHS" #f) #f)))
  (with-env '(("EPOCHS" . "3") ("WORKERS" . "0"))
    (lambda ()
      (check-equal? (env-number "EPOCHS" 20) 3)
      (check-equal? (env-number "WORKERS" 0 exact-nonnegative-integer?) 0))))

(test-case "a bad value is an error naming its variable"
  (for ([bad (in-list (list (list "EPOCHS" "0" exact-positive-integer?
                                  #rx"promised: EPOCHS:exact-positive-integer[?]")
                            (list "EPOCHS" "1.5" exact-positive-integer?
                                  #rx"promised: EPOCHS:exact-positive-integer[?]")
                            (list "WORKERS" "-1" exact-nonnegative-integer?
                                  #rx"promised: WORKERS:natural[?]")
                            (list "EPOCHS" "three" exact-positive-integer?
                                  #rx"EPOCHS is not a number: three")))])
    (match-define (list name value accept/c rx) bad)
    (with-env (list (cons name value))
      (lambda ()
        (check-exn (env-violation? rx) (lambda () (env-number name 1 accept/c)))))))

(test-case "a bad value stops the script before the corpus download"
  (with-temporary-directory (cache)
    (with-env (list (cons "EPOCHS" "0") (cons "RKTORCH_AUDIO_DIR" (path->string cache)))
      (lambda ()
        (check-exn (env-violation? #rx"promised: EPOCHS:exact-positive-integer[?]")
                   (lambda () (dynamic-require `(submod ,train-asr main) #f)))))
    (check-equal? (directory-list cache) '())))
