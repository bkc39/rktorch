#lang racket/base

(module+ test
  (require (only-in racket/generic define/generic)
           (only-in racket/match match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/copy-layers.rkt"
                    copied-field deleg deleg-inner holder keep set-deleg-inner! twin)
           (only-in "private/within-memory.rkt" memory-bounded-test))

  (define (self-box)
    (define b (box #f))
    (set-box! b b)
    b)

  (define (self-vector)
    (define v (vector #f))
    (vector-set! v 0 v)
    v)

  (define (self-mpair)
    (define p (mcons 1 #f))
    (set-mcdr! p p)
    p)

  (define ((self-valued make))
    (define h (make))
    (hash-set! h 'me h)
    h)

  (define (me h)
    (hash-ref h 'me))

  (define (first-slot v)
    (vector-ref v 0))

  (struct cycle-case (name make back))

  (define self-cycles
    (list (cycle-case "box" self-box unbox)
          (cycle-case "vector" self-vector first-slot)
          (cycle-case "mpair" self-mpair mcdr)
          (cycle-case "hash" (self-valued make-hash) me)
          (cycle-case "hasheq" (self-valued make-hasheq) me)
          (cycle-case "weak hasheq" (self-valued make-weak-hasheq) me)
          (cycle-case "ephemeron hasheq" (self-valued make-ephemeron-hasheq) me)))

  (define (reader-graph make)
    (define p (make-placeholder #f))
    (placeholder-set! p (make p))
    (make-reader-graph p))

  (define (listed p) (list 1 p))
  (define (paired p) (cons 1 p))
  (define (vector-and-hash p) (vector-immutable (hasheq 'v p) p))
  (define (vector-and-mutable-vector p) (vector-immutable p (vector 1)))
  (define (vector-and-pair-cycle p) (vector-immutable (reader-graph paired) p))

  (define immutable-cycles
    `(("a list" ,listed)
      ("a pair" ,paired)
      ("a vector and a hash" ,vector-and-hash)
      ("a vector holding a mutable vector" ,vector-and-mutable-vector)
      ("a vector holding a pair cycle" ,vector-and-pair-cycle)))

  (define reentered #rx"^layer-copy: a cycle re-enters a hand-written layer-rebuild")

  (memory-bounded-test "a self-referential mutable container copies to a fresh one holding itself"
    (for ([c (in-list self-cycles)])
      (match-define (cycle-case name make back) c)
      (define v (make))
      (define v2 (copied-field (holder v)))
      (check-false (eq? v2 v) name)
      (check-eq? (back v2) v2 name)
      (when (hash? v)
        (check-equal? (hash-weak? v2) (hash-weak? v) name)
        (check-equal? (hash-ephemeron? v2) (hash-ephemeron? v) name)
        (check-equal? (hash-equal? v2) (hash-equal? v) name))))

  (memory-bounded-test "a cycle of immutable values alone raises, naming the field"
    (for ([c (in-list immutable-cycles)])
      (match-define (list name make) c)
      (check-exn #rx"^layer-copy: a cycle of immutable values cannot be copied\n  field: \"slot\""
                 (lambda () (layer-copy (holder (reader-graph make))))
                 name)))

  (memory-bounded-test "an immutable cycle closed through a mutable box copies, as a tuple does"
    (define b (box #f))
    (define v (vector-immutable (hasheq 'v b) b (self-box) (vector-immutable 1 'a)))
    (set-box! b v)
    (for ([m (in-list (list (holder v) (holder b) (twin b v) (twin v b)))])
      (define v2
        (match (copied-field m)
          [(? box? b2) (unbox b2)]
          [(or (list (? box?) v2) (list v2 (? box?))) v2]
          [v2 v2]))
      (match-define (vector table2 b2 inner2 shared2) v2)
      (check-true (immutable? v2))
      (check-false (eq? v2 v))
      (check-false (eq? b2 b))
      (check-eq? (unbox b2) v2)
      (check-eq? (hash-ref table2 'v) b2)
      (check-false (eq? inner2 (vector-ref v 2)))
      (check-eq? (unbox inner2) inner2)
      (check-eq? shared2 (vector-ref v 3))))

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
    (check-equal? (hash-values c) '(1)))

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
    (match-define (list original-inner _) (outer))
    (check-false (eq? inner original-inner))
    (check-eq? (unbox (inner)) c))

  (memory-bounded-test "a child's back-pointer to its parent points at the parent's copy"
    (define up (vector #f))
    (define kid (holder up))
    (define parent (twin kid (Parameter (ones 2))))
    (vector-set! up 0 parent)
    (match-define (list _ w) (parent))
    (for ([copy (in-list (list (layer-copy parent)
                               (vector-ref ((layer-copy kid)) 0)))])
      (match-define (list kid2 w2) (copy))
      (check-false (eq? copy parent))
      (check-eq? (vector-ref (kid2) 0) copy)
      (check-equal? (children copy) (list kid2))
      (check-equal? (parameters copy) (list w2))
      (check-false (eq? w2 w))))

  (memory-bounded-test "a box and a list that hold each other copy from either"
    (define t (ones 1))
    (define b (box #f))
    (define l (list t b))
    (set-box! b l)
    (for ([m (in-list (list (twin b 0) (twin 0 l)))])
      (define b2
        (match ((layer-copy m))
          [(list (? box? b2) _) b2]
          [(list _ (list _ b2)) b2]))
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
     (define (layer-rebuild-method self child _tensor)
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

  (memory-bounded-test "a hand-written layer with no layer-rebuild-method on a cycle is refused"
    (define n (named #f))
    (define h (holder (list n)))
    (set-named-back! n h)
    (check-exn #rx"^layer-copy: a hand-written gen:layer has no layer-rebuild-method"
               (lambda () (layer-copy h))))

  (memory-bounded-test "a cycle through a delegating rebuild copies from the define-layer side"
    (define d (deleg #f 'plain))
    (define h (holder d))
    (set-deleg-inner! d h)
    (define h2 (layer-copy h))
    (check-eq? (deleg-inner (h2)) h2)
    (check-exn reentered (lambda () (layer-copy d))))

  (memory-bounded-test "#466: layers delegating through the plain name raise, naming the layer"
    (define d (deleg #f 'plain))
    (set-deleg-inner! d (deleg d 'plain))
    (check-exn #rx"^layer-copy: a cycle re-enters [^\n]*\n  layer: #<procedure:deleg>"
               (lambda () (layer-copy d)))
    (check-exn #rx"^layer-rebuild: a cycle re-enters [^\n]*\n  layer: #<procedure:deleg>"
               (lambda () (layer-rebuild d keep values))))

  (memory-bounded-test "layers delegating with callbacks of their own raise, naming the layer"
    (define d (deleg #f 'fixed))
    (set-deleg-inner! d (deleg d 'fixed))
    (check-exn #rx"^layer-rebuild: a cycle re-enters [^\n]*\n  layer: #<procedure:deleg>"
               (lambda () (layer-rebuild d keep values))))

  (define (count-rebuild! count)
    (set-box! count (add1 (unbox count)))
    (when (> (unbox count) 64)
      (raise 'unbounded)))

  (define (unbounded? e)
    (eq? e 'unbounded))

  (struct prompted ([inner #:mutable] count)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (prompted-inner self))
     (define (layer-rebuild-method self _child _tensor)
       (match-define (prompted inner count) self)
       (count-rebuild! count)
       (prompted (call-with-continuation-prompt (lambda () (layer-rebuild inner keep values)))
                 count))])

  (memory-bounded-test "a cycle of direct rebuilds behind prompts raises, naming the layer"
    (define p (prompted #f (box 0)))
    (set-prompted-inner! p (prompted p (prompted-count p)))
    (check-exn #rx"^layer-rebuild: a cycle re-enters [^\n]*\n  layer: #<procedure:prompted>"
               (lambda ()
                 (with-handlers ([unbounded? (lambda (_e) (fail "the cycle recursed"))])
                   (layer-rebuild p keep values)))))

  (struct raw-looped ([inner #:mutable] count)
    #:methods gen:layer
    [(define/generic raw layer-rebuild-method)
     (define (layer-forward self . _inputs) (raw-looped-inner self))
     (define (layer-rebuild-method self child tensor)
       (match-define (raw-looped inner count) self)
       (count-rebuild! count)
       (raw-looped (raw inner child tensor) count))])

  (memory-bounded-test "a define/generic cycle of hand-written layers recurses until stopped"
    (define r (raw-looped #f (box 0)))
    (set-raw-looped-inner! r (raw-looped r (raw-looped-count r)))
    (check-eq? (with-handlers ([unbounded? values]) (layer-copy r)) 'unbounded)
    (check-equal? (unbox (raw-looped-count r)) 65))

  (struct cb ([other #:mutable])
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (cb-other self))
     (define (layer-rebuild-method self child _tensor)
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
      (check-eq? (unbox k2) h2)))

  (memory-bounded-test "an immutable value over no mutable value or layer is shared, as a tuple is"
    (define pure (list 1 (vector-immutable 'a "s") (box-immutable 2) (hash 'k (list 3)) (hasheq)))
    (define over (list 1 (vector-immutable (box 2))))
    (match-define (list pure2 over2) (copied-field (twin pure over)))
    (check-eq? pure2 pure)
    (check-false (eq? over2 over))
    (match-define (list 1 (vector b2)) over2)
    (match-define (list _ (vector b)) over)
    (check-false (eq? b2 b))
    (check-equal? b2 (box 2)))

  (memory-bounded-test "a long list closed through a box is a list in the copy"
    (define b (box #f))
    (define p (list b (holder b)))
    (define long (append (build-list 256 values) p))
    (set-box! b long)
    (define p2 (copied-field (holder p)))
    (match-define (list b2 holder2) p2)
    (define long2 (unbox b2))
    (check-true (list? long2))
    (check-equal? (length long2) 258)
    (check-eq? (list-tail long2 256) p2)
    (check-eq? (holder2) b2)))
