#lang racket/base

(require (only-in racket/contract/combinator exn:fail:contract:blame?))

(provide blamed blames-caller? contract-violation?)

(define (blamed file)
  (regexp (string-append "blaming: [^\n]*" (regexp-quote file))))

(define ((contract-violation? blame rx) e)
  (and (exn:fail:contract? e)
       (regexp-match? blame (exn-message e))
       (regexp-match? rx (exn-message e))))

(define ((blames-caller? module-name) e)
  (and (exn:fail:contract:blame? e)
       (regexp-match? (blamed module-name) (exn-message e))))
