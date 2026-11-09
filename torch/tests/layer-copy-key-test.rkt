#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" copied-field holder)
           (only-in "private/within-memory.rkt" memory-bounded-test within-memory))

  (define unhashable #rx"^layer-copy: an equal[?]-based hash's key reaches ")
  (define own-table #rx"^layer-copy: an equal[?]-based hash's key reaches that hash")
  (define part-built
    #rx"^layer-copy: an immutable equal[?]-based hash was copied while one of its keys was still")

  (define (keyed-by-holder-of make-holder)
    (define table (make-hash))
    (define-values (key close!) (make-holder))
    (hash-set! table key 1)
    (close! table)
    table)

  (define (vector-holder)
    (define v (vector #f))
    (values v (lambda (table) (vector-set! v 0 table))))

  (define (value-holder)
    (define g (make-hash))
    (values (list g) (lambda (table) (hash-set! g 'back table))))

  (define (self-valued)
    (define g (make-hash))
    (hash-set! g 'self g)
    g)

  (define (vector-valued)
    (define h (make-hash))
    (hash-set! h 'k (vector h))
    h)

  (define (eq-keyed-cycle)
    (define e (make-hasheq))
    (hash-set! e (vector e) 1)
    e)

  (memory-bounded-test "a key that reaches its table directly or through another hash raises"
    (for ([make-holder (in-list (list vector-holder value-holder))])
      (check-exn own-table (lambda () (copied-field (holder (keyed-by-holder-of make-holder)))))))

  (memory-bounded-test "a key reaching a hash whose cycle runs through values or eq? keys copies"
    (for ([make (in-list (list self-valued vector-valued eq-keyed-cycle))])
      (define inner (make))
      (define table (make-hash (list (cons (list inner) 'x))))
      (define table2 (copied-field (holder table)))
      (match-define (list (and key2 (list inner2))) (hash-keys table2))
      (check-false (eq? inner2 inner))
      (check-equal? (hash-count inner2) 1)
      (check-eq? (hash-ref table2 key2 #f) 'x)))

  (define (box-tree n)
    (cond
      [(= n 1) (box n)]
      [else (cons (box-tree (quotient n 2)) (box-tree (- n (quotient n 2))))]))

  (test-case "a thousand keys over one shared tree of twenty thousand boxes check in one pass"
    (within-memory
     512
     (lambda ()
       (define shared (box-tree 20000))
       (define table (make-hash (for/list ([i (in-range 1000)]) (cons (vector i shared) i))))
       (check-equal? (hash-count (copied-field (holder table))) 1000))
     #:seconds 10))

  (memory-bounded-test "an immutable equal? hash whose keys are equal? only while part-built raises"
    (for ([wrap (in-list (list values list))])
      (define k1 (vector #f 'a))
      (define k2 (vector #f 'b))
      (define frozen (hash (wrap k1) 1 (wrap k2) 2))
      (vector-set! k1 0 frozen)
      (vector-set! k2 0 frozen)
      (check-false (equal? k1 k2))
      (check-exn part-built (lambda () (copied-field (holder k1))))))

  (memory-bounded-test "a mutable equal? hash keyed by a pair on its cycle keeps every entry"
    (define table (make-hash))
    (define p (cons (holder table) 'a))
    (hash-set! table p 1)
    (hash-set! table (cons #f #f) 2)
    (define p2 (copied-field (holder p)))
    (match-define (cons layer2 'a) p2)
    (define table2 (layer2))
    (check-equal? (hash-count table2) (hash-count table))
    (check-eq? (hash-ref table2 p2 #f) 1)
    (check-eq? (hash-ref table2 (cons #f #f) #f) 2))

  (memory-bounded-test "two pair keys still being copied keep both entries"
    (define table (make-hash))
    (define q2 (cons (holder table) 'b))
    (define q1 (cons q2 'a))
    (hash-set! table q1 1)
    (hash-set! table q2 2)
    (define q1-copy (copied-field (holder q1)))
    (match-define (cons (and q2-copy (cons layer2 'b)) 'a) q1-copy)
    (define table2 (layer2))
    (check-equal? (hash-count table2) (hash-count table))
    (check-eq? (hash-ref table2 q1-copy #f) 1)
    (check-eq? (hash-ref table2 q2-copy #f) 2))

  (memory-bounded-test "two mutable keys equal? while part-built keep both entries"
    (define table (make-hash))
    (define k2 (vector (holder table) 'b))
    (define k1 (vector k2 'a))
    (hash-set! table k1 1)
    (hash-set! table k2 2)
    (define k1-copy (copied-field (holder k1)))
    (match-define (vector (and k2-copy (vector layer2 'b)) 'a) k1-copy)
    (define table2 (layer2))
    (check-equal? (hash-count table2) (hash-count table))
    (check-eq? (hash-ref table2 k1-copy #f) 1)
    (check-eq? (hash-ref table2 k2-copy #f) 2))

  (memory-bounded-test "an equal-always? hash keyed by a pair on its cycle finds the copied key"
    (define table (make-hashalw))
    (define p (cons (holder table) 'a))
    (hash-set! table p 1)
    (define p2 (copied-field (holder p)))
    (match-define (cons layer2 'a) p2)
    (check-equal? (hash-keys (layer2)) (list p2))
    (check-eq? (hash-ref (layer2) p2 #f) 1)
    (define back (box #f))
    (define q (cons (holder back) 'a))
    (define frozen (hashalw q 1))
    (set-box! back frozen)
    (define q2 (copied-field (holder q)))
    (match-define (cons back-layer2 'a) q2)
    (define frozen2 (unbox (back-layer2)))
    (check-true (immutable? frozen2))
    (check-true (hash-equal-always? frozen2))
    (check-eq? (hash-ref frozen2 q2 #f) 1))

  (memory-bounded-test "keys copied to equal? keys raise instead of merging entries"
    (define one (holder 0))
    (define table (make-hash (list (cons (cons (holder 1) 'k) 1) (cons (cons (holder 2) 'k) 2))))
    (check-exn #rx"^layer-rebuild: two keys of an equal[?]-based hash copied to equal[?] keys"
               (lambda () (layer-rebuild (holder table) (lambda (_name _c) one) values))))

  (memory-bounded-test "keys of an eq?, eqv? or equal-always? hash replaced by one layer raise"
    (define one (holder 0))
    (for ([kind (in-list (list (list "eq" make-hasheq make-immutable-hasheq)
                               (list "eqv" make-hasheqv make-immutable-hasheqv)
                               (list "equal-always" make-hashalw make-immutable-hashalw)))])
      (match-define (list same makes ...) kind)
      (define message
        (regexp (format "^layer-rebuild: two keys of an ~a[?]-based hash copied to ~a[?] keys"
                        same same)))
      (for ([make (in-list makes)])
        (define table (make (list (cons (holder 1) 'a) (cons (holder 2) 'b))))
        (check-exn message
                   (lambda () (layer-rebuild (holder table) (lambda (_name _c) one) values))))))

  (memory-bounded-test "a key holding a mutable hash beside a cycle without one copies"
    (define v (vector #f))
    (vector-set! v 0 v)
    (define key (list (make-hash '((a . 1))) v))
    (define table (make-hash (list (cons key 'x))))
    (define table2 (copied-field (holder table)))
    (match-define (list key2) (hash-keys table2))
    (check-false (eq? key2 key))
    (check-eq? (hash-ref table2 key2 #f) 'x)
    (match-define (list inner2 v2) key2)
    (check-eq? (vector-ref v2 0) v2)
    (check-equal? (hash-ref inner2 'a) 1))

  (struct named-key (name [table #:mutable])
    #:transparent
    #:property prop:equal+hash
    (list (lambda (a b recur) (recur (named-key-name a) (named-key-name b)))
          (lambda (a recur) (recur (named-key-name a)))
          (lambda (a recur) (recur (named-key-name a)))))

  (memory-bounded-test "a transparent key whose own hash ignores its table copies"
    (define table (make-hash))
    (define key (named-key "w" #f))
    (hash-set! table key 1)
    (set-named-key-table! key table)
    (define table2 (copied-field (holder table)))
    (check-equal? (hash-count table2) 1)
    (check-eq? (hash-ref table2 key #f) 1))

  (struct prefab-key ([table #:mutable]) #:prefab)
  (struct open-key ([table #:mutable]) #:transparent)

  (test-case "a struct key reaching its table through its fields raises, not hangs"
    (within-memory
     256
     (lambda ()
       (for ([make (in-list (list prefab-key open-key))]
             [close! (in-list (list set-prefab-key-table! set-open-key-table!))])
         (define table (make-hash))
         (define key (make #f))
         (hash-set! table key 1)
         (close! key table)
         (check-exn unhashable (lambda () (layer-copy (holder table))))))
     #:seconds 10)))
