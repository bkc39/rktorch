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

(match-define (vector racket-deps native-location) (current-command-line-arguments))

(define own
  (match (definition racket-deps 'own-platform-packages)
    [`(quote ,(list (? string? names) ...)) names]))

(define located
  (match (definition native-location 'platform-packages)
    [`(hash ,pairs ...)
     (let loop ([pairs pairs])
       (match pairs
         ['() '()]
         [(list* (? string?) (? string? name) more) (cons name (loop more))]))]))

(unless (equal? (sort own string<?) (sort located string<?))
  (raise-user-error 'platform-lists
                    "own-platform-packages in ~a is ~s, but platform-packages in ~a names ~s"
                    racket-deps own native-location located))
