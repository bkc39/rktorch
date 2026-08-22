#lang racket/base

(require (only-in racket/generic define/generic)
         (only-in racket/list append-map range)
         (only-in "module.rkt"
                  gen:module
                  module-forward
                  module-parameters
                  module-named-parameters
                  module-buffers
                  module-set-training!
                  module-training?))

(provide Sequential
         sequential?)

(struct Sequential% (modules)
  #:reflection-name 'Sequential
  #:property prop:procedure
  (lambda (self . inputs) (apply module-forward self inputs))
  #:methods gen:module
  ;; bare method names here are the enclosing methods; recursion must use
  ;; the define/generic aliases
  [(define/generic gen-forward module-forward)
   (define/generic gen-parameters module-parameters)
   (define/generic gen-named module-named-parameters)
   (define/generic gen-buffers module-buffers)
   (define/generic gen-set-training! module-set-training!)
   (define/generic gen-training? module-training?)
   (define (module-forward self . inputs)
     (apply (lambda (x)
              (for/fold ([acc x])
                        ([m (in-list (Sequential%-modules self))])
                (gen-forward m acc)))
            inputs))
   (define (module-parameters self)
     (append-map gen-parameters (Sequential%-modules self)))
   (define (module-named-parameters self prefix)
     (define ms (Sequential%-modules self))
     (append-map
      (lambda (i m)
        (gen-named m (string-append prefix (number->string i) ".")))
      (range (length ms)) ms))
   (define (module-buffers self)
     (append-map gen-buffers (Sequential%-modules self)))
   (define (module-set-training! self training?)
     (for ([m (in-list (Sequential%-modules self))])
       (gen-set-training! m training?)))
   (define (module-training? self)
     (andmap gen-training? (Sequential%-modules self)))])

(define (Sequential . modules)
  (Sequential% modules))

(define sequential? Sequential%?)
