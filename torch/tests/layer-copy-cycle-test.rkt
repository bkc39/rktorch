#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" copied-field holder rebuild-directly twin)
           (only-in "private/within-memory.rkt" memory-bounded-test))

  (define (self-box)
    (define b (box #f))
    (set-box! b b)
    b)

  (define (self-vector)
    (define v (vector #f))
    (vector-set! v 0 v)
    v)

  (define (self-valued make)
    (define h (make))
    (hash-set! h 'me h)
    h)

  (define (reader-graph-cycle)
    (define p (make-placeholder #f))
    (placeholder-set! p (list 1 p))
    (make-reader-graph p))

  (define reentered #rx"^layer-copy: a cycle re-enters a hand-written layer-rebuild")

  (memory-bounded-test "a self-referential box copies to a fresh self-referential box"
    (define b (self-box))
    (define c (copied-field (holder b)))
    (check-false (eq? c b))
    (check-eq? (unbox c) c))

  (memory-bounded-test "a self-referential vector copies to a fresh one"
    (define v (self-vector))
    (define c (copied-field (holder v)))
    (check-false (eq? c v))
    (check-eq? (vector-ref c 0) c))

  (memory-bounded-test "a hash holding itself as a value copies, holding its copy"
    (for ([make (in-list (list make-hash make-hasheq make-weak-hasheq make-ephemeron-hasheq))])
      (define h (self-valued make))
      (define c (copied-field (holder h)))
      (check-false (eq? c h))
      (check-eq? (hash-ref c 'me) c)
      (check-equal? (hash-weak? c) (hash-weak? h))
      (check-equal? (hash-equal? c) (hash-equal? h))))

  (memory-bounded-test "an eq? hash keyed by itself copies, keyed by its copy"
    (define h (make-hasheq))
    (hash-set! h h 1)
    (define c (copied-field (holder h)))
    (check-false (eq? c h))
    (check-equal? (hash-keys c) (list c))
    (check-eq? (hash-ref c c) 1))

  (memory-bounded-test "a hash keyed by a box holding the hash copies, the key holding the copy"
    (define h (make-hasheq))
    (hash-set! h (box h) 1)
    (define c (copied-field (holder h)))
    (match-define (list k) (hash-keys c))
    (check-false (eq? c h))
    (check-eq? (unbox k) c)
    (check-eq? (hash-ref c k) 1))

  (memory-bounded-test "an equal? hash keyed by a box holding the hash copies alike"
    (define k (box #f))
    (define h (hash k 1))
    (set-box! k h)
    (define c (copied-field (holder h)))
    (match-define (list k2) (hash-keys c))
    (check-false (eq? k2 k))
    (check-eq? (unbox k2) c)
    (check-true (immutable? c))
    (check-equal? (hash-has-key? c k2) (hash-has-key? h k)))

  (memory-bounded-test "a layer whose plain field holds the layer copies to its copy"
    (define v (vector #f))
    (define m (holder v))
    (vector-set! v 0 m)
    (define c (layer-copy m))
    (check-false (eq? c m))
    (check-eq? (vector-ref (c) 0) c))

  (memory-bounded-test "a box filled later with an ancestor layer holds the ancestor's copy"
    (define b (box #f))
    (define outer (twin (holder b) 0))
    (set-box! b outer)
    (define c (layer-copy outer))
    (match-define (list inner _) (c))
    (check-false (eq? inner (car (outer))))
    (check-eq? (unbox (inner)) c))

  (memory-bounded-test "a child's back-pointer to its parent points at the parent's copy"
    (define up (vector #f))
    (define kid (holder up))
    (define parent (twin kid (Parameter (ones 2))))
    (vector-set! up 0 parent)
    (for ([copy (in-list (list (layer-copy parent)
                               (vector-ref ((layer-copy kid)) 0)))])
      (match-define (list kid2 w2) (copy))
      (check-false (eq? copy parent))
      (check-eq? (vector-ref (kid2) 0) copy)
      (check-equal? (children copy) (list kid2))
      (check-equal? (parameters copy) (list w2))
      (check-false (eq? w2 (cadr (parent))))))

  (memory-bounded-test "an immutable make-reader-graph cycle copies to a fresh cycle"
    (define l (reader-graph-cycle))
    (define c (copied-field (holder l)))
    (check-false (eq? c l))
    (check-pred pair? c)
    (check-equal? (car c) 1)
    (check-eq? (cadr c) c))

  (define (vector-cycle extra)
    (define p (make-placeholder #f))
    (placeholder-set! p (vector-immutable (hasheq 'v p) p extra))
    (make-reader-graph p))

  (memory-bounded-test "an immutable cycle of a vector and a hash copies whole"
    (define l (vector-cycle (box 1)))
    (define c (copied-field (holder l)))
    (check-false (eq? c l))
    (check-true (immutable? c))
    (check-true (immutable? (vector-ref c 0)))
    (check-eq? (hash-ref (vector-ref c 0) 'v) c)
    (check-eq? (vector-ref c 1) c)
    (check-false (eq? (vector-ref c 2) (vector-ref l 2)))
    (check-equal? (vector-ref c 2) (box 1)))

  (memory-bounded-test "an immutable vector cycle that holds another cycle raises"
    (define l (vector-cycle (self-box)))
    (check-exn #rx"^layer-copy: a cycle of immutable vectors, boxes or hashes reaches another cycle"
               (lambda () (layer-copy (holder l)))))

  (memory-bounded-test "a box and a list that hold each other copy from either"
    (define t (ones 1))
    (define b (box #f))
    (define l (list t b))
    (set-box! b l)
    (for ([m (in-list (list (twin b 0) (twin 0 l)))])
      (match-define (list x y) ((layer-copy m)))
      (define b2 (if (box? x) x (cadr y)))
      (match-define (list t2 b3) (unbox b2))
      (check-eq? b3 b2)
      (check-false (eq? b2 b))
      (check-false (eq? t2 t))))

  (memory-bounded-test "an equal? hash keyed by a box on a cycle is found by the key's copy"
    (define back (box #f))
    (define table (make-hash))
    (define key (box (holder back)))
    (hash-set! table key 'x)
    (set-box! back table)
    (match-define (list key2 table2) (copied-field (twin key table)))
    (check-false (eq? key2 key))
    (check-eq? (unbox ((unbox key2))) table2)
    (check-eq? (hash-ref table2 key2 #f) 'x))

  (memory-bounded-test "an immutable equal? hash copied while its key is half built raises"
    (define back (box #f))
    (define key (vector 7 (holder back)))
    (define table (hash key 1))
    (set-box! back table)
    (check-exn #rx"^layer-copy: an immutable equal[?]-based hash was copied while one of its keys"
               (lambda () (layer-copy (holder key))))
    (match-define (list layer2 _) (~> (vector-ref key 1) (twin 0) copied-field))
    (define table2 (unbox (layer2)))
    (match-define (list key2) (hash-keys table2))
    (check-eq? (vector-ref key2 1) layer2)
    (check-eq? (hash-ref table2 key2 #f) 1))

  (memory-bounded-test "an equal? hash keyed by a box that came to hold the hash raises"
    (define table (make-hash))
    (define key (box #f))
    (hash-set! table key 1)
    (set-box! key table)
    (check-exn #rx"^layer-copy: an equal[?]-based hash's key reaches that hash"
               (lambda () (layer-copy (holder table))))
    (check-exn #rx"^layer-copy: an equal[?]-based hash's key reaches that hash"
               (lambda () (layer-copy (holder key)))))

  (memory-bounded-test "an equal? hash key reaching the hash through an mpair or a hash raises"
    (define (keyed make-key close!)
      (define table (make-hash))
      (define-values (key slot) (make-key))
      (hash-set! table key 1)
      (close! slot table)
      table)
    (define through-mpair
      (keyed (lambda () (define p (mcons #f #f)) (values p p)) set-mcar!))
    (define through-hash
      (keyed (lambda () (define b (box #f)) (values (hash 'in b) b)) set-box!))
    (for ([table (list through-mpair through-hash)])
      (check-exn #rx"^layer-copy: an equal[?]-based hash's key reaches that hash"
                 (lambda () (layer-copy (holder table))))))

  (memory-bounded-test "an equal? hash key sharing a part twice is not a cycle"
    (define part (box 1))
    (define table (make-hash (list (cons (list part part) 'v))))
    (define table2 (copied-field (holder table)))
    (match-define (list (list a b)) (hash-keys table2))
    (check-false (eq? a part))
    (check-eq? a b)
    (check-eq? (hash-ref table2 (list (box 1) (box 1)) #f) 'v))

  (memory-bounded-test "a direct layer-rebuild with plain callbacks copies a cycle"
    (define c ((layer-rebuild (holder (self-box)) (lambda (_name c) c) values)))
    (check-eq? (unbox c) c))

  (struct looped ([inner #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (looped-inner self))
     (define (layer-rebuild self child _tensor)
       (looped (child "inner" (looped-inner self))))])

  (memory-bounded-test "a cycle re-entering a hand-written rebuild raises, naming the layer"
    (define w (looped #f))
    (set-looped-inner! w (holder w))
    (check-exn reentered (lambda () (layer-copy w))))

  (memory-bounded-test "the same cycle entered at its define-layer layer copies"
    (define w (looped #f))
    (define h (holder w))
    (set-looped-inner! w h)
    (define h2 (layer-copy h))
    (check-false (eq? h2 h))
    (check-pred looped? (h2))
    (check-eq? (looped-inner (h2)) h2))

  (struct named ([back #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (named-back self))
     (define (layer-named-children self)
       (list (cons "back" (named-back self))))])

  (memory-bounded-test "a hand-written layer with no layer-rebuild on a cycle is refused"
    (define n (named #f))
    (define h (holder (list n)))
    (set-named-back! n h)
    (check-exn #rx"^layer-copy: a hand-written gen:layer has no layer-rebuild method"
               (lambda () (layer-copy h))))

  (struct back ([outer #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (back-outer self))
     (define (layer-rebuild self child tensor)
       (~> (back-outer self) (rebuild-directly child tensor) back))])

  (memory-bounded-test "a cycle closed through a delegating rebuild copies from the define-layer side"
    (define w (back #f))
    (define h (holder w))
    (set-back-outer! w h)
    (define h2 (layer-copy h))
    (check-eq? (back-outer (h2)) h2)
    (check-exn reentered (lambda () (layer-copy w))))

  (memory-bounded-test "mutually delegating hand-written layers raise"
    (define b1 (back #f))
    (set-back-outer! b1 (back b1))
    (check-exn reentered (lambda () (layer-copy b1))))

  (struct cb ([other #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (cb-other self))
     (define (layer-rebuild self child _tensor)
       (cb (child "x" (cb-other self))))])

  (memory-bounded-test "hand-written layers delegating through child raise"
    (define c1 (cb #f))
    (set-cb-other! c1 (cb c1))
    (check-exn reentered (lambda () (layer-copy c1))))

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
      (check-eq? (unbox k2) h2))))
