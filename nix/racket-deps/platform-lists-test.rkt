#lang racket/base
(require (only-in racket/match == match match-define)
         (only-in syntax/modread with-module-reading-parameterization))

(define (module-forms file)
  (match (with-module-reading-parameterization
           (λ () (call-with-input-file file read)))
    [`(module ,_ ,_ (#%module-begin ,forms ...)) forms]
    [`(module ,_ ,_ ,forms ...) forms]))

(define (definition file name)
  (or (for/or ([form (in-list (module-forms file))])
        (match form
          [`(define ,(== name) ,value) value]
          [_ #f]))
      (raise-user-error 'platform-lists "~a defines no ~a" file name)))

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define (dep-platform dep)
  (match (and (pair? dep) (member '#:platform (cdr dep)))
    [(list* _ platform _) platform]
    [_ #f]))

(match-define (vector racket-deps native-location info) (current-command-line-arguments))

(define own
  (match (definition racket-deps 'own-platform-packages)
    [`(quote ,names) #:when (andmap string? names) names]))

(define located
  (match (definition native-location 'platform-packages)
    [`(hash ,pairs ...)
     (let loop ([pairs pairs] [table (hash)])
       (match pairs
         ['() table]
         [(list* (? string? platform) (? string? name) more)
          (loop more (hash-set table platform name))]))]))

(unless (equal? (sort own string<?) (sort (hash-values located) string<?))
  (raise-user-error 'platform-lists
                    "own-platform-packages in ~a is ~s, but platform-packages in ~a names ~s"
                    racket-deps own native-location (hash-values located)))

(define deps
  (match (definition info 'deps)
    [`(quote ,deps) deps]))

(for ([dep (in-list deps)]
      #:when (member (dep-name dep) own))
  (define platform (dep-platform dep))
  (unless (equal? (hash-ref located platform #f) (dep-name dep))
    (raise-user-error 'platform-lists
                      "~a declares ~a with #:platform ~s, which platform-packages in ~a maps to ~s"
                      info
                      (dep-name dep)
                      platform
                      native-location
                      (hash-ref located platform #f))))
