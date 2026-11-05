#lang racket/base

(require (only-in racket/match match))

(provide check-hashable)

(define (check-hashable who k table)
  (when (unhashable? k table)
    (raise-arguments-error
     who
     (string-append "an equal?-based hash's key reaches that hash or a cycle with a mutable"
                    " hash on it, which equal-hash-code cannot hash"))))

(define (parts v)
  (match v
    [(cons a d) (list a d)]
    [(mcons a d) (list a d)]
    [(box x) (list x)]
    [(? vector?) (vector->list v)]
    [(? hash?) (for*/list ([(k x) (in-hash v)] [part (in-list (list k x))]) part)]
    [_ '()]))

(define (unhashable? k table)
  (define state (make-hasheq))
  (define cyclic? #f)
  (define mutable-hash? #f)
  (define (reached?)
    (let visit ([v k])
      (match (hash-ref state v #f)
        ['open (set! cyclic? #t) #f]
        ['closed #f]
        [_
         (when (and (hash? v) (not (immutable? v)))
           (set! mutable-hash? #t))
         (hash-set! state v 'open)
         (begin0 (or (eq? v table) (ormap visit (parts v)))
                 (hash-set! state v 'closed))])))
  (and (or (pair? k) (mpair? k) (box? k) (vector? k) (hash? k))
       (or (reached?) (and cyclic? mutable-hash?))))
