#lang racket/base

(require (only-in racket/match == match match-lambda))

(provide check-hashable
         collapsed!
         filled!
         filling!
         hash-parts
         key-check
         rekeyed!)

(struct key-check (hashable filling)
  #:constructor-name build-key-check
  #:omit-define-syntaxes)

(define (key-check)
  (build-key-check (make-hasheq) (make-hasheq)))

(define (filling! kc v)
  (hash-set! (key-check-filling kc) v #t))

(define (filled! kc v)
  (hash-remove! (key-check-filling kc) v))

(define (rekeyed! who kc table entries)
  (for-each (match-lambda [(cons k _) (check-hashable who kc k table)]) entries)
  (hash-clear! table)
  (for-each (match-lambda [(cons k x) (hash-set! table k x)]) entries)
  (collapsed! who (length entries) table))

(define (collapsed! who expected table)
  (unless (= (hash-count table) expected)
    (raise-arguments-error
     who
     (string-append "two keys of an equal?-based hash copied to equal? keys,"
                    " so the copy would hold fewer entries than the original"))))

(define (hash-parts h)
  (for*/list ([(k x) (in-hash h)] [part (in-list (list k x))])
    part))

(define (keys-hashed? h)
  (and (hash? h) (hash-equal? h) (not (immutable? h))))

(define (compound? v)
  (or (pair? v) (mpair? v) (box? v) (vector? v) (hash? v) (struct? v)))

(define (edges v)
  (filter compound?
          (match v
            [(cons a d) (list a d)]
            [(mcons a d) (list a d)]
            [(box x) (list x)]
            [(? vector?) (vector->list v)]
            [(? keys-hashed?) (hash-parts v)]
            [(? hash?) (hash-values v)]
            [(? struct?) (match (vector->list (struct->vector v)) [(list* _ fields) fields])])))

(struct search (who kc table index low settled component [stack #:mutable])
  #:constructor-name build-search
  #:omit-define-syntaxes)

(define (search who kc table)
  (build-search who kc table (make-hasheq) (make-hasheq) (make-hasheq) (make-hasheq) '()))

(define (check-hashable who kc k table)
  (when (compound? k)
    (reached (search who kc table) k))
  (void))

(define (reached s v)
  (cond
    [(eq? v (search-table s))
     (raise-arguments-error
      (search-who s)
      "an equal?-based hash's key reaches that hash, which equal-hash-code cannot hash")]
    [(hash-ref (key-check-hashable (search-kc s)) v #f) #t]
    [(hash-ref (search-component s) v #f) => (lambda (settled) (eq? settled 'settled))]
    [(hash-ref (search-index s) v #f) #t]
    [else (visited s v)]))

(define (visited s v)
  (define i (hash-count (search-index s)))
  (define low (search-low s))
  (hash-set! (search-index s) v i)
  (hash-set! low v i)
  (set-search-stack! s (cons v (search-stack s)))
  (define settled?
    (for/fold ([settled? (not (hash-ref (key-check-filling (search-kc s)) v #f))])
              ([w (in-list (edges v))])
      (define w-settled? (reached s w))
      (when (on-stack? s w)
        (hash-set! low v (min (hash-ref low v) (hash-ref low w))))
      (and w-settled? settled?)))
  (hash-set! (search-settled s) v settled?)
  (when (= (hash-ref low v) i)
    (popped! s v))
  (eq? (hash-ref (search-component s) v 'settled) 'settled))

(define (on-stack? s w)
  (and (hash-ref (search-index s) w #f) (not (hash-ref (search-component s) w #f))))

(define (popped! s v)
  (define-values (members more)
    (let pop ([stack (search-stack s)] [members '()])
      (match stack
        [(cons (== v eq?) more) (values (cons v members) more)]
        [(cons m more) (pop more (cons m members))])))
  (set-search-stack! s more)
  (define in-component (for/hasheq ([m (in-list members)]) (values m #t)))
  (when (for*/or ([m (in-list members)] #:when (keys-hashed? m) [k (in-hash-keys m)])
          (hash-ref in-component k #f))
    (raise-arguments-error
     (search-who s)
     (string-append "an equal?-based hash's key reaches a mutable equal?-based hash"
                    " that its own keys reach, which equal-hash-code cannot hash")))
  (define settled? (for/and ([m (in-list members)]) (hash-ref (search-settled s) m)))
  (for ([m (in-list members)])
    (hash-set! (search-component s) m (if settled? 'settled 'unsettled))
    (when settled?
      (hash-set! (key-check-hashable (search-kc s)) m #t))))
