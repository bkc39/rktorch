#lang racket/base

(provide blamed contract-violation?)

(define (blamed file)
  (regexp (string-append "blaming: [^\n]*" (regexp-quote file))))

(define ((contract-violation? blame rx) e)
  (and (exn:fail:contract? e)
       (regexp-match? blame (exn-message e))
       (regexp-match? rx (exn-message e))))
