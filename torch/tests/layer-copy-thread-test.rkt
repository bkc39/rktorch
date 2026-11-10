#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
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

  (define part-built
    #rx"^layer-copy: a rebuild raised, escaped or was killed while a copy was part-built")
  (define two-threads #rx"^layer-copy: a child callback used from two threads at once")

  (define (running thunk)
    (define result (box #f))
    (values (thread (lambda () (set-box! result (with-handlers ([exn:fail? values]) (thunk)))))
            result))

  (define (copying child name x)
    (running (lambda () (child name x))))

  (struct gate (in go)
    #:methods gen:layer
    [(define (layer-forward _self . _inputs) 0)
     (define (layer-rebuild-method self _child _tensor)
       (semaphore-post (gate-in self))
       (semaphore-wait (gate-go self))
       (gate (make-semaphore 0) (make-semaphore 1)))])

  (define (closed-gate)
    (gate (make-semaphore 0) (make-semaphore 0)))

  (struct forked (inner g copies)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (forked-copies self))
     (define (layer-rebuild-method self child tensor)
       (match-define (forked inner g _) self)
       (define (rebuilt)
         (layer-rebuild inner child tensor))
       (define-values (first first-result) (running rebuilt))
       (if g (semaphore-wait (gate-in g)) (thread-wait first))
       (define-values (second second-result) (running rebuilt))
       (thread-wait second)
       (when g
         (semaphore-post (gate-go g)))
       (thread-wait first)
       (define copies (map unbox (list first-result second-result)))
       (define failure (findf exn? copies))
       (when failure
         (raise failure))
       (forked (child "inner" inner) g copies))])

  (bounded-test "one copy's child used from two threads at once is refused"
    (define g (closed-gate))
    (check-exn two-threads (lambda () (layer-copy (forked (holder g) g '())))))

  (bounded-test "a worker thread that finishes inside the rebuild joins the copy"
    (define s (holder 0))
    (define c (layer-copy (forked s #f '())))
    (match-define (list x y) (forked-copies c))
    (check-eq? x y)
    (check-eq? x (forked-inner c))
    (check-false (eq? x s)))

  (struct stalled (armed ready)
    #:methods gen:layer
    [(define (layer-forward _self . _inputs) 0)
     (define (layer-rebuild-method self _child _tensor)
       (match-define (stalled armed ready) self)
       (cond
         [(unbox armed)
          (set-box! armed #f)
          (semaphore-post ready)
          (sync never-evt)]
         [else (stalled (box #f) ready)]))])

  (struct abandoned (first second stall)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (abandoned-first self))
     (define (layer-rebuild-method self child _tensor)
       (match-define (abandoned first second stall) self)
       (define t (thread (lambda () (child "first" first))))
       (semaphore-wait (stalled-ready stall))
       (kill-thread t)
       (thread-wait t)
       (abandoned (child "first" first) (child "second" second) stall))])

  (define (itself s)
    (values s s))

  (define (held s)
    (define h (holder s))
    (values h h))

  (define (paired s)
    (define p (cons s 'x))
    (values (holder p) (holder p)))

  (define killed-cases
    `(("a hand-written layer" ,itself)
      ("a define-layer layer" ,held)
      ("a pair two layers hold" ,paired)))

  (memory-bounded-test "a worker thread killed mid-rebuild fails the copy"
    (for ([c (in-list killed-cases)])
      (match-define (list name make) c)
      (define s (stalled (box #t) (make-semaphore 0)))
      (define-values (first second) (make s))
      (check-exn part-built (lambda () (layer-copy (abandoned first second s))) name)))

  (struct both (a b g)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (both-b self))
     (define (layer-rebuild-method self child _tensor)
       (match-define (both a b g) self)
       (define-values (ta ra) (copying child "a" a))
       (semaphore-wait (gate-in g))
       (define-values (tb rb) (copying child "b" b))
       (thread-wait tb)
       (semaphore-post (gate-go g))
       (thread-wait ta)
       (both (unbox ra) (unbox rb) g))])

  (bounded-test "a second thread entering while a first is inside the copy is refused"
    (define g (closed-gate))
    (define cycle
      (let ([p (make-placeholder #f)])
        (placeholder-set! p (vector-immutable (vector-immutable p) (holder 0)))
        (make-reader-graph p)))
    (define c (layer-copy (both (holder (vector-immutable g)) (holder cycle) g)))
    (check-pred holder? (both-a c))
    (check-pred exn:fail:contract? (both-b c))
    (check-regexp-match two-threads (exn-message (both-b c))))

  (struct leaver (inner g)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (leaver-inner self))
     (define (layer-rebuild-method self child _tensor)
       (match-define (leaver inner g) self)
       (copying child "inner" inner)
       (semaphore-wait (gate-in g))
       (leaver #f g))])

  (bounded-test "a rebuild returning while its worker is inside the copy is refused"
    (define g (closed-gate))
    (check-exn two-threads (lambda () (layer-copy (leaver (holder g) g))))
    (semaphore-post (gate-go g))))
