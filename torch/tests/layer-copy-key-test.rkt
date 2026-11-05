#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" copied-field holder)
           (only-in "private/within-memory.rkt" memory-bounded-test within-memory))

  (define unhashable #rx"^layer-copy: an equal[?]-based hash's key reaches that hash")

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
