#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/within.rkt" bounded-test))

  (define-layer holder (slot)
    #:init (v)
    (set! slot v)
    #:forward ()
    slot)

  (define-layer twin (a b)
    #:init (x y)
    (set! a x)
    (set! b y)
    #:forward ()
    (list a b))

  (define ((refusal field) e)
    (and (exn:fail:contract? e)
         (regexp-match? (pregexp (string-append "^layer-copy: .*field: \""
                                                field "\""))
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define (check-refused field m)
    (check-exn (refusal field) (lambda () (layer-copy m))))

  (define (copied-field m)
    ((layer-copy m)))

  (define (self-box)
    (define b (box #f))
    (set-box! b b)
    b)

  (define (self-vector)
    (define v (vector #f))
    (vector-set! v 0 v)
    v)

  (define (self-hash)
    (define h (make-hash))
    (hash-set! h 'me h)
    h)

  (define (self-keyed-hash)
    (define h (make-hasheq))
    (hash-set! h (box h) 1)
    h)

  (define (reader-graph-cycle)
    (define p (make-placeholder #f))
    (placeholder-set! p (list 1 p))
    (make-reader-graph p))

  (bounded-test "a self-referential box is refused naming its field"
    (check-refused "slot" (holder (self-box))))

  (bounded-test "a self-referential vector is refused naming its field"
    (check-refused "slot" (holder (self-vector))))

  (bounded-test "a hash holding itself as a value is refused"
    (check-refused "slot" (holder (self-hash))))

  (bounded-test "a hash whose key holds the hash is refused"
    (check-refused "slot" (holder (self-keyed-hash))))

  (bounded-test "a box filled later with an ancestor layer is refused"
    (define b (box #f))
    (define outer (twin (holder b) 0))
    (set-box! b outer)
    (check-refused "slot" outer))

  (bounded-test "a layer set into a vector in its own field is refused"
    (define v (vector #f))
    (define m (holder v))
    (vector-set! v 0 m)
    (check-refused "slot" m))

  (bounded-test "an immutable make-reader-graph cycle is refused"
    (check-refused "slot" (holder (reader-graph-cycle))))

  (bounded-test "a box and a list that hold each other are refused from either"
    (define t (ones 1))
    (define b (box #f))
    (define l (list t b))
    (set-box! b l)
    (check-refused "a" (twin b 0))
    (check-refused "b" (twin 0 l)))

  (bounded-test "two fields sharing a box share one fresh box in the copy"
    (define b (box 1))
    (match-define (list a2 b2) (copied-field (twin b b)))
    (check-eq? a2 b2)
    (check-false (eq? a2 b))
    (check-equal? (unbox a2) 1))

  (bounded-test "a parent and its child sharing a box still share it"
    (define b (box 1))
    (match-define (list inner b2) (~> (holder b) (twin b) copied-field))
    (check-eq? (inner) b2)
    (check-false (eq? b2 b)))

  (bounded-test "vector and box keys of an equal? hash are found in the copy"
    (define h (make-hash (list (cons (vector 1 2) 'v) (cons (box 3) 'b))))
    (define c (copied-field (holder h)))
    (define vk (findf vector? (hash-keys c)))
    (define bk (findf box? (hash-keys c)))
    (check-false (memq vk (hash-keys h)))
    (check-false (memq bk (hash-keys h)))
    (check-eq? (hash-ref c vk) 'v)
    (check-eq? (hash-ref c bk) 'b)
    (check-eq? (hash-ref c (vector 1 2)) 'v)
    (check-eq? (hash-ref c (box 3)) 'b))

  (bounded-test "a key sharing a sub-container with its value is copied"
    (define shared (vector 1))
    (define h (make-hash (list (cons (list shared) shared))))
    (define c (copied-field (holder h)))
    (match-define (list k) (hash-keys c))
    (check-eq? (car k) (hash-ref c k))
    (check-false (eq? (car k) shared)))

  (define entries (list (cons 'k (box 1))))

  (bounded-test "a copied hash keeps its comparison"
    (for ([h (in-list (list (make-hash entries) (make-hasheqv entries)
                            (make-hasheq entries) (make-hashalw entries)
                            (hash 'k (box 1)) (hasheqv 'k (box 1))
                            (hasheq 'k (box 1)) (hashalw 'k (box 1))))])
      (define c (copied-field (holder h)))
      (check-false (eq? c h))
      (check-false (eq? (hash-ref c 'k) (hash-ref h 'k)))
      (check-equal? (immutable? c) (immutable? h))
      (check-equal? (hash-equal? c) (hash-equal? h))
      (check-equal? (hash-eqv? c) (hash-eqv? h))
      (check-equal? (hash-eq? c) (hash-eq? h))
      (check-equal? (hash-equal-always? c) (hash-equal-always? h))))

  (bounded-test "a copied hash keeps its weakness"
    (for ([h (in-list (list (make-weak-hash entries)
                            (make-ephemeron-hasheq entries)
                            (make-hash entries)))])
      (define c (copied-field (holder h)))
      (check-false (eq? c h))
      (check-equal? (hash-weak? c) (hash-weak? h))
      (check-equal? (hash-ephemeron? c) (hash-ephemeron? h))))

  (bounded-test "tensor and layer keys map to the copies the fields hold"
    (define w (Parameter (ones 2)))
    (define inner (holder 0))
    (define table (make-hasheq (list (cons w 'w) (cons inner 'inner))))
    (match-define (list fields table2) (~> (twin w inner) (twin table) copied-field))
    (match-define (list w2 inner2) (fields))
    (check-false (eq? w2 w))
    (check-false (eq? inner2 inner))
    (check-eq? (hash-ref table2 w2 #f) 'w)
    (check-eq? (hash-ref table2 inner2 #f) 'inner)
    (check-false (hash-ref table2 w #f)))

  (bounded-test "a layer in a hash value is copied, not shared"
    (define inner (holder 0))
    (match-define (list inner2 h2)
      (copied-field (twin inner (make-hasheq (list (cons 'k inner))))))
    (check-eq? (hash-ref h2 'k) inner2)
    (check-false (eq? inner2 inner)))

  (bounded-test "an immutable box stays immutable"
    (define c (~> (box-immutable 1) holder copied-field))
    (check-true (immutable? c))
    (check-equal? (unbox c) 1))

  (bounded-test "a direct layer-rebuild keeps sharing within the call"
    (define b (box 1))
    (define rebuilt (layer-rebuild (twin b b) (lambda (_name c) c) values))
    (match-define (list a2 b2) (rebuilt))
    (check-eq? a2 b2)
    (check-false (eq? a2 b)))

  (bounded-test "a direct layer-rebuild refuses a cycle naming its field"
    (check-exn #rx"^layer-rebuild: .*field: \"slot\""
               (lambda ()
                 (layer-rebuild (holder (self-box)) (lambda (_name c) c) values))))

  (bounded-test "a direct layer-rebuild inside a layer-copy keeps its own memo"
    (define rebuild-directly layer-rebuild)
    (struct pair-of (inner)
      #:methods gen:layer
      [(define (layer-forward self . _inputs) (pair-of-inner self))
       (define (layer-named-children self)
         (list (cons "inner" (pair-of-inner self))))
       (define (layer-rebuild self child _tensor)
         (define inner (pair-of-inner self))
         (set! slots (list (inner)
                           ((child "inner" inner))
                           ((rebuild-directly inner (lambda (_name c) c) values))))
         (pair-of (child "inner" inner)))])
    (define slots '())
    (~> (box 1) holder pair-of layer-copy)
    (match-define (list original copied direct) slots)
    (check-false (eq? copied original))
    (check-false (eq? direct original))
    (check-false (eq? direct copied)))

  (bounded-test "a hand-written layer-rebuild sees a child copy fully built"
    (struct wrapper (inner)
      #:methods gen:layer
      [(define (layer-forward self . _inputs) (wrapper-inner self))
       (define (layer-named-children self)
         (list (cons "inner" (wrapper-inner self))))
       (define (layer-rebuild self child _tensor)
         (define copy (child "inner" (wrapper-inner self)))
         (set! seen (unbox (copy)))
         (wrapper copy))])
    (define seen #f)
    (define c (~> (vector 1 (box 2)) box holder wrapper layer-copy))
    (check-equal? seen (vector 1 (box 2)))
    (check-equal? (unbox ((c))) (vector 1 (box 2)))))
