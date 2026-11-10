#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/copy-layers.rkt" copied-field deleg deleg-inner holder keep twin)
           (only-in "private/within.rkt" bounded-test))

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
    (match-define (list (and k (list part))) (hash-keys c))
    (check-eq? part (hash-ref c k))
    (check-false (eq? part shared)))

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

  (struct pair-of (inner seen)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (pair-of-seen self))
     (define (layer-named-children self)
       (list (cons "inner" (pair-of-inner self))))
     (define (layer-rebuild-method self child _tensor)
       (define inner (pair-of-inner self))
       (pair-of (child "inner" inner)
                (list (inner)
                      ((child "inner" inner))
                      ((layer-rebuild inner (lambda (_name c) c) values)))))])

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
     (define (layer-rebuild-method self child tensor)
       (define inner (via-inner self))
       (via (child "inner" inner) (layer-rebuild inner child tensor)))])

  (bounded-test "a direct layer-rebuild given layer-copy's child shares its layers and containers"
    (define leaf (holder 0))
    (define c (~> (twin leaf (box 1)) (via #f) layer-copy))
    (match-define (list leaf-copy box-copy) ((via-inner c)))
    (match-define (list leaf-direct box-direct) ((c)))
    (check-false (eq? leaf-copy leaf))
    (check-eq? leaf-direct leaf-copy)
    (check-eq? box-direct box-copy))

  (bounded-test "a container shared through a delegating layer stays shared"
    (define b (box 1))
    (match-define (list d2 h2) (copied-field (twin (deleg (holder b) 'plain) (holder b))))
    (check-eq? ((deleg-inner d2)) (h2))
    (check-false (eq? (h2) b)))

  (bounded-test "a hash key shared through a delegating layer finds the other copy"
    (define b (box 1))
    (match-define (list d2 h2)
      (~> (list (cons b 'x)) make-hasheq holder (deleg 'plain) (twin (holder b)) copied-field))
    (check-eq? (hash-ref ((deleg-inner d2)) (h2) #f) 'x))

  (bounded-test "a define/generic rebuild of a define-layer layer copies it"
    (define b (box 1))
    (define c (~> (holder b) (deleg 'generic) layer-copy deleg-inner))
    (check-false (eq? (c) b))
    (check-equal? (c) (box 1))
    (define d (~> (holder b) (deleg 'generic) (layer-rebuild keep values)))
    (check-false (eq? ((deleg-inner d)) b))
    (check-equal? ((deleg-inner d)) (box 1))
    (define h (holder (box 1)))
    (match-define (list d2 h2) (~> h (deleg 'generic) (twin h) copied-field))
    (check-eq? (deleg-inner d2) h2)
    (check-false (eq? h2 h)))

  (struct leaf (v)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (leaf-v self))
     (define (layer-rebuild-method self _child _tensor)
       (leaf (leaf-v self)))])

  (bounded-test "a hand-written layer reached twice through the plain name is copied once"
    (define l (leaf 1))
    (match-define (list d2 l2) (~> l (deleg 'plain) (twin l) copied-field))
    (check-eq? (deleg-inner d2) l2)
    (check-false (eq? l2 l)))

  (bounded-test "a define/generic rebuild of a hand-written layer copies it apart, as __deepcopy__"
    (define l (leaf 1))
    (match-define (list d2 l2) (~> l (deleg 'generic) (twin l) copied-field))
    (check-false (eq? (deleg-inner d2) l2)))

  (struct both (a b)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (list (both-a self) (both-b self)))
     (define (layer-rebuild-method self child _tensor)
       (both (layer-rebuild (both-a self) child values)
             (layer-rebuild (both-b self) child values)))])

  (bounded-test "a hand-written layer reached twice by one direct rebuild is rebuilt once"
    (define l (leaf 1))
    (define r (layer-rebuild (both l l) keep values))
    (check-eq? (both-a r) (both-b r))
    (check-false (eq? (both-a r) l)))

  (define (deep _name c)
    (layer-rebuild c deep values))

  (bounded-test "a child that rebuilds directly keeps a layer reached twice shared"
    (define h (holder 0))
    (match-define (list a b) ((layer-rebuild (twin h h) deep values)))
    (check-eq? a b)
    (check-false (eq? a h)))

  (struct apart (a b)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (list (apart-a self) (apart-b self)))
     (define (layer-rebuild-method self child tensor)
       (apart (layer-rebuild (apart-a self) child tensor)
              (layer-rebuild (apart-b self) (lambda (name c) (child name c)) tensor)))])

  (bounded-test "a nested direct rebuild on other callbacks keeps its own memo"
    (define l (leaf 1))
    (define r (layer-rebuild (apart l l) keep values))
    (check-false (eq? (apart-a r) (apart-b r))))

  (define (in-thread thunk)
    (define result (box #f))
    (thread-wait (thread (lambda () (set-box! result (thunk)))))
    (unbox result))

  (struct fenced (a b run)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (list (fenced-a self) (fenced-b self)))
     (define (layer-rebuild-method self child tensor)
       (match-define (fenced a b run) self)
       (fenced (layer-rebuild a child tensor) (run (lambda () (layer-rebuild b child tensor))) run))])

  (bounded-test "a nested direct rebuild behind a prompt joins the one it is inside"
    (define l (leaf 1))
    (define r (layer-rebuild (fenced l l call-with-continuation-prompt) keep values))
    (check-eq? (fenced-a r) (fenced-b r))
    (check-false (eq? (fenced-a r) l)))

  (bounded-test "a nested direct rebuild on a thread the rebuild starts keeps its own memo"
    (define l (leaf 1))
    (define r (layer-rebuild (fenced l l in-thread) keep values))
    (check-true (leaf? (fenced-b r)))
    (check-false (eq? (fenced-a r) (fenced-b r))))

  (struct outer (a d)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (outer-d self))
     (define (layer-rebuild-method self child tensor)
       (outer (layer-rebuild (outer-a self) child tensor) (layer-copy (outer-d self))))])

  (bounded-test "a layer-copy inside a direct rebuild shares nothing with it"
    (define l (leaf 1))
    (match-define (outer a2 d2) (layer-rebuild (outer l (deleg l 'fixed)) keep values))
    (check-false (eq? (deleg-inner d2) a2))
    (check-false (eq? (deleg-inner d2) l)))

  (struct fallback (f)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (fallback-f self))
     (define (layer-rebuild-method self child _tensor)
       (define f (fallback-f self))
       (fallback (with-handlers ([exn:fail:contract? (lambda (_e) f)])
                   (child "a" f))))])

  (bounded-test "a refused child caught by a rebuild does not read as a cycle"
    (define f
      (procedure->Layer (lambda (x) x) #:parameters (list (cons "w" (Parameter (ones 1))))))
    (check-exn #rx"^layer-copy: a procedure->Layer layer with parameters"
               (lambda () (layer-copy (twin (fallback f) f)))))

  (bounded-test "a refusal caught by a rebuild fails a copy left part-built"
    (define f
      (procedure->Layer (lambda (x) x) #:parameters (list (cons "w" (Parameter (ones 1))))))
    (define m (mcons #f '()))
    (set-mcar! m (cons m f))
    (check-exn #rx"^layer-copy: a rebuild raised, escaped or was killed while a copy was part-built"
               (lambda () (layer-copy (twin (fallback (holder (mcar m))) (holder m))))))

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
     (define (layer-rebuild-method self child _tensor)
       (define copy (child "inner" (wrapper-inner self)))
       (wrapper copy (unbox (copy))))])

  (bounded-test "a hand-written layer-rebuild sees a child copy fully built"
    (define c (~> (vector 1 (box 2)) box holder (wrapper #f) layer-copy))
    (check-equal? (wrapper-seen c) (vector 1 (box 2)))
    (check-equal? (unbox ((c))) (vector 1 (box 2))))

  (bounded-test "a layer reached through a delegating rebuild and a field is copied once"
    (for ([h (in-list (list (holder (box 1)) (~> 1 box holder (deleg 'plain))))])
      (match-define (list d2 h2) (~> (deleg h 'plain) (twin h) copied-field))
      (check-eq? (deleg-inner d2) h2)
      (check-false (eq? h2 h))
      (match-define (list h3 d3) (copied-field (twin h (deleg h 'plain))))
      (check-eq? (deleg-inner d3) h3)
      (check-false (eq? h3 h))))

  (bounded-test "a weak table stays keyed by the original, not the field's copy"
    (define k (box 1))
    (match-define (list k2 t2) (copied-field (twin k (make-weak-hasheq (list (cons k 'v))))))
    (check-false (eq? k2 k))
    (check-eq? (hash-ref t2 k #f) 'v)
    (check-false (hash-ref t2 k2 #f)))

  (define stash #f)

  (struct stasher (inner)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (stasher-inner self))
     (define (layer-rebuild-method self child _tensor)
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
     (define (layer-rebuild-method self child _tensor)
       (set! stash child)
       ((escaper-leave self) 'left))])

  (bounded-test "a child callback is closed when a rebuild escapes layer-copy"
    (define b (box 1))
    (set! stash #f)
    (check-eq? (let/ec leave (layer-copy (escaper leave (holder b)))) 'left)
    (check-true (procedure? stash))
    (check-closed b))

  (bounded-test "a Children value held twice is one value in the copy"
    (define kids (children-by-index (list (holder 0))))
    (match-define (list kids2 b2) (copied-field (twin kids (box kids))))
    (check-eq? (unbox b2) kids2))

  (bounded-test "an immutable equal? hash keyed by a Children value copies"
    (define kids (children-by-index (list (holder 0))))
    (check-equal? (hash-count (copied-field (holder (hash kids 1)))) 1))

  (bounded-test "a mutable string or byte string held twice is copied once"
    (for ([s (in-list (list (string-copy "ab") (bytes 1 2)))])
      (match-define (list s2 b2) (copied-field (twin s (box s))))
      (check-false (eq? s2 s))
      (check-false (immutable? s2))
      (check-equal? s2 s)
      (check-eq? (unbox b2) s2)))

  (struct settings (b) #:transparent)

  (bounded-test "immutable strings and transparent structs are shared"
    (define fixed "fixed")
    (define open-settings (settings (box 1)))
    (match-define (list s2 c2) (copied-field (twin fixed open-settings)))
    (check-eq? s2 fixed)
    (check-eq? c2 open-settings)))
