#lang racket/base

(require (only-in racket/contract/combinator exn:fail:contract:blame?))

(provide blames-caller?)

(define ((blames-caller? module-name) e)
  (and (exn:fail:contract:blame? e)
       (regexp-match? (regexp (string-append "blaming: [^\n]*"
                                             (regexp-quote module-name)))
                      (exn-message e))))
