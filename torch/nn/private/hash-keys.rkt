#lang racket/base

(require (only-in racket/dict in-dict in-dict-keys)
         (only-in racket/match match))

(provide check-hashable
         hash-parts
         rekeyed!)

(define (check-hashable who k table)
  (when (unhashable? k table)
    (raise-arguments-error
     who
     (string-append "an equal?-based hash's key reaches that hash or a cycle with a mutable"
                    " hash on it, which equal-hash-code cannot hash"))))

(define (rekeyed! who table entries)
  (for ([k (in-dict-keys entries)])
    (check-hashable who k table))
  (hash-clear! table)
  (for ([(k x) (in-dict entries)])
    (hash-set! table k x))
  (unless (= (hash-count table) (length entries))
    (raise-arguments-error
     who
     (string-append "two keys of an equal?-based hash copied to equal? keys,"
                    " so the copy would hold fewer entries than the original"))))

(define (parts v)
  (match v
    [(cons a d) (list a d)]
    [(mcons a d) (list a d)]
    [(box x) (list x)]
    [(? vector?) (vector->list v)]
    [(? hash?) (hash-parts v)]
    [(? struct?) (cdr (vector->list (struct->vector v)))]
    [_ '()]))

(define (hash-parts h)
  (for*/list ([(k x) (in-hash h)] [part (in-list (list k x))])
    part))

(define (reachable roots)
  (let visit ([todo roots] [seen (hasheq)])
    (match todo
      ['() seen]
      [(cons v more)
       (if (hash-has-key? seen v)
           (visit more seen)
           (visit (append (parts v) more) (hash-set seen v #t)))])))

(define (unhashable? k table)
  (define from-k (reachable (list k)))
  (or (hash-has-key? from-k table)
      (for/or ([v (in-hash-keys from-k)]
               #:when (and (hash? v) (not (immutable? v))))
        (hash-has-key? (reachable (parts v)) v))))
