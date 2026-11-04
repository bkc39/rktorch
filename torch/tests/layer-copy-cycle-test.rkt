#lang racket/base

(module+ test
  (require (only-in racket/generic define/generic)
           (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/within.rkt" bounded-test)
           (only-in "private/within-memory.rkt" memory-bounded-test))

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

  (memory-bounded-test "a self-referential box is refused naming its field"
    (check-refused "slot" (holder (self-box))))

  (memory-bounded-test "a self-referential vector is refused naming its field"
    (check-refused "slot" (holder (self-vector))))

  (memory-bounded-test "a hash holding itself as a value is refused"
    (check-refused "slot" (holder (self-hash))))

  (memory-bounded-test "a hash whose key holds the hash is refused"
    (check-refused "slot" (holder (self-keyed-hash))))

  (memory-bounded-test "a box filled later with an ancestor layer is refused"
    (define b (box #f))
    (define outer (twin (holder b) 0))
    (set-box! b outer)
    (check-refused "slot" outer))

  (memory-bounded-test "a layer set into a vector in its own field is refused"
    (define v (vector #f))
    (define m (holder v))
    (vector-set! v 0 m)
    (check-refused "slot" m))

  (memory-bounded-test "an immutable make-reader-graph cycle is refused"
    (check-refused "slot" (holder (reader-graph-cycle))))

  (memory-bounded-test "a box and a list that hold each other are refused from either"
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

  (memory-bounded-test "a direct layer-rebuild refuses a cycle naming its field"
    (check-exn #rx"^layer-rebuild: .*field: \"slot\""
               (lambda ()
                 (layer-rebuild (holder (self-box)) (lambda (_name c) c) values))))

  (struct looped ([inner #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (looped-inner self))
     (define (layer-rebuild self child _tensor)
       (looped (child "inner" (looped-inner self))))])

  (memory-bounded-test "a cycle only rebuilds can see is refused while copying"
    (define w (looped #f))
    (set-looped-inner! w (holder w))
    (check-refused "slot" w))

  (struct named ([back #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (named-back self))
     (define (layer-named-children self)
       (list (cons "back" (named-back self))))])

  (memory-bounded-test "a cycle through layer-named-children is refused naming it"
    (define n (named #f))
    (define h (holder (list n)))
    (set-named-back! n h)
    (check-refused "back" h))

  (define rebuild-directly layer-rebuild)

  (struct pair-of (inner seen)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (pair-of-seen self))
     (define (layer-named-children self)
       (list (cons "inner" (pair-of-inner self))))
     (define (layer-rebuild self child _tensor)
       (define inner (pair-of-inner self))
       (pair-of (child "inner" inner)
                (list (inner)
                      ((child "inner" inner))
                      ((rebuild-directly inner (lambda (_name c) c) values)))))])

  (bounded-test "a direct layer-rebuild inside a layer-copy keeps its own memo"
    (match-define (list original copied direct)
      (~> (box 1) holder (pair-of '()) layer-copy pair-of-seen))
    (check-false (eq? copied original))
    (check-false (eq? direct original))
    (check-false (eq? direct copied)))

  (struct via (inner direct)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (via-direct self))
     (define (layer-named-children self)
       (list (cons "inner" (via-inner self))))
     (define (layer-rebuild self child tensor)
       (define inner (via-inner self))
       (via (child "inner" inner) (rebuild-directly inner child tensor)))])

  (bounded-test "a direct layer-rebuild given layer-copy's child shares its layers and containers"
    (define leaf (holder 0))
    (define c (~> (twin leaf (box 1)) (via #f) layer-copy))
    (match-define (list leaf-copy box-copy) ((via-inner c)))
    (match-define (list leaf-direct box-direct) ((c)))
    (check-false (eq? leaf-copy leaf))
    (check-eq? leaf-direct leaf-copy)
    (check-eq? box-direct box-copy))

  (struct deleg (inner)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (deleg-inner self))
     (define (layer-rebuild self child tensor)
       (~> (deleg-inner self) (rebuild-directly child tensor) deleg))])

  (bounded-test "a container shared through a delegating layer stays shared"
    (define b (box 1))
    (match-define (list d2 h2) (copied-field (twin (deleg (holder b)) (holder b))))
    (check-eq? ((deleg-inner d2)) (h2))
    (check-false (eq? (h2) b)))

  (bounded-test "a hash key shared through a delegating layer finds the other copy"
    (define b (box 1))
    (match-define (list d2 h2)
      (~> (list (cons b 'x)) make-hasheq holder deleg (twin (holder b)) copied-field))
    (check-eq? (hash-ref ((deleg-inner d2)) (h2) #f) 'x))

  (struct generic-deleg (inner)
    #:methods gen:layer
    [(define/generic raw-rebuild layer-rebuild)
     (define (layer-forward self . _inputs) (generic-deleg-inner self))
     (define (layer-rebuild self child tensor)
       (~> self generic-deleg-inner (raw-rebuild child tensor) generic-deleg))])

  (bounded-test "a define/generic rebuild of a define-layer layer copies it"
    (define b (box 1))
    (define c (~> (holder b) generic-deleg layer-copy generic-deleg-inner))
    (check-false (eq? (c) b))
    (check-equal? (c) (box 1))
    (define d (~> (holder b) generic-deleg (layer-rebuild (lambda (_name c) c) values)))
    (check-false (eq? ((generic-deleg-inner d)) b))
    (check-equal? ((generic-deleg-inner d)) (box 1))
    (define h (holder (box 1)))
    (match-define (list d2 h2) (~> h generic-deleg (twin h) copied-field))
    (check-eq? (generic-deleg-inner d2) h2)
    (check-false (eq? h2 h)))

  (struct leaf (v)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (leaf-v self))
     (define (layer-rebuild self _child _tensor)
       (leaf (leaf-v self)))])

  (bounded-test "#466: a define/generic rebuild of a hand-written layer copies it apart"
    (define l (leaf 1))
    (match-define (list d2 l2) (~> l generic-deleg (twin l) copied-field))
    (check-false (eq? (generic-deleg-inner d2) l2))
    (check-false (eq? l2 l)))

  (struct fallback (f)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (fallback-f self))
     (define (layer-rebuild self child _tensor)
       (define f (fallback-f self))
       (fallback (with-handlers ([exn:fail:contract? (lambda (_e) f)])
                   (child "a" f))))])

  (bounded-test "a refused child caught by a rebuild does not read as a cycle"
    (define f
      (procedure->Layer (lambda (x) x) #:parameters (list (cons "w" (Parameter (ones 1))))))
    (check-exn #rx"^layer-copy: a procedure->Layer layer with parameters"
               (lambda () (layer-copy (twin (fallback f) f)))))

  (bounded-test "a weak or ephemeron hash keeps its keys and copies its values"
    (define k (box 1))
    (for ([make (in-list (list make-weak-hasheq make-ephemeron-hasheq))])
      (define v (box 2))
      (define c (~> (list (cons k v)) make holder copied-field))
      (collect-garbage 'major)
      (check-equal? (hash-ref c k #f) (box 2))
      (check-false (eq? (hash-ref c k #f) v))))

  (struct wrapper (inner seen)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (wrapper-inner self))
     (define (layer-named-children self)
       (list (cons "inner" (wrapper-inner self))))
     (define (layer-rebuild self child _tensor)
       (define copy (child "inner" (wrapper-inner self)))
       (wrapper copy (unbox (copy))))])

  (bounded-test "a hand-written layer-rebuild sees a child copy fully built"
    (define c (~> (vector 1 (box 2)) box holder (wrapper #f) layer-copy))
    (check-equal? (wrapper-seen c) (vector 1 (box 2)))
    (check-equal? (unbox ((c))) (vector 1 (box 2))))

  (bounded-test "a layer reached through a delegating rebuild and a field is copied once"
    (for ([h (in-list (list (holder (box 1)) (~> 1 box holder deleg)))])
      (match-define (list d2 h2) (~> (deleg h) (twin h) copied-field))
      (check-eq? (deleg-inner d2) h2)
      (check-false (eq? h2 h))
      (match-define (list h3 d3) (copied-field (twin h (deleg h))))
      (check-eq? (deleg-inner d3) h3)
      (check-false (eq? h3 h))))

  (struct back ([outer #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (back-outer self))
     (define (layer-rebuild self child tensor)
       (~> (back-outer self) (rebuild-directly child tensor) back))])

  (memory-bounded-test "a cycle closed through a delegating rebuild is refused naming its field"
    (define w (back #f))
    (define h (holder w))
    (set-back-outer! w h)
    (check-refused "slot" h))

  (memory-bounded-test "mutually delegating hand-written layers are refused"
    (define b1 (back #f))
    (set-back-outer! b1 (back b1))
    (check-exn #rx"^layer-copy: a hand-written layer-rebuild reaches the layer itself"
               (lambda () (layer-copy b1))))

  (bounded-test "a weak table stays keyed by the original, not the field's copy"
    (define k (box 1))
    (match-define (list k2 t2) (copied-field (twin k (make-weak-hasheq (list (cons k 'v))))))
    (check-false (eq? k2 k))
    (check-eq? (hash-ref t2 k #f) 'v)
    (check-false (hash-ref t2 k2 #f)))

  (memory-bounded-test "a weak hash whose key holds the hash copies, sharing the key"
    (for ([make (in-list (list make-weak-hasheq make-ephemeron-hasheq))])
      (define h (make))
      (define k (box h))
      (hash-set! h k 1)
      (match-define (list h2 k2) (copied-field (holder (list h k))))
      (check-false (eq? h2 h))
      (check-equal? (hash-count h2) 1)
      (check-eq? (hash-ref h2 k #f) 1)
      (check-false (eq? k2 k))
      (check-eq? (unbox k2) h2)))

  (memory-bounded-test "a weak hash holding itself as a value is refused"
    (for ([make (in-list (list make-weak-hasheq make-ephemeron-hasheq))])
      (define h (make))
      (hash-set! h 'me h)
      (check-refused "slot" (holder h))))

  (define stash #f)

  (struct stasher (inner)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (stasher-inner self))
     (define (layer-rebuild self child _tensor)
       (set! stash child)
       (stasher (child "inner" (stasher-inner self))))])

  (define (check-closed b)
    (define closed #rx"^layer-copy: a child callback used after layer-copy returned")
    (check-exn closed (lambda () (stash "inner" (holder b))))
    (check-exn closed (lambda () (layer-rebuild (holder b) stash values))))

  (bounded-test "a child callback kept past layer-copy refuses later use"
    (define b (box 1))
    (~> (holder b) stasher layer-copy)
    (check-closed b)
    (define f
      (procedure->Layer (lambda (x) x) #:parameters (list (cons "w" (Parameter (ones 1))))))
    (set! stash #f)
    (check-exn #rx"^layer-copy: a procedure->Layer" (lambda () (layer-copy (stasher f))))
    (check-true (procedure? stash))
    (check-closed b))

  (struct escaper (leave inner)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (escaper-inner self))
     (define (layer-rebuild self child _tensor)
       (set! stash child)
       ((escaper-leave self) 'left))])

  (bounded-test "a child callback is closed when a rebuild escapes layer-copy"
    (define b (box 1))
    (set! stash #f)
    (check-eq? (let/ec leave (layer-copy (escaper leave (holder b)))) 'left)
    (check-true (procedure? stash))
    (check-closed b)))
