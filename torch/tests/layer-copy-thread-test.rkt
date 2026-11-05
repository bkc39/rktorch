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

  (define rebuild-directly layer-rebuild)

  (define part-built #rx"^layer-copy: a rebuild raised, escaped or was killed while a copy was part-built")
  (define two-threads #rx"^layer-copy: a child callback used from two threads at once")

  (struct slow ()
    #:methods gen:layer
    [(define (layer-forward _self . _inputs) 0)
     (define (layer-rebuild _self _child _tensor)
       (sleep 0.1)
       (slow))])

  (struct forked (inner sequential? copies)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (forked-copies self))
     (define (layer-rebuild self child tensor)
       (define inner (forked-inner self))
       (define (worker)
         (define result (box #f))
         (define (rebuilt)
           (set-box! result (with-handlers ([exn:fail? values])
                              (rebuild-directly inner child tensor))))
         (define t (thread rebuilt))
         (when (forked-sequential? self)
           (thread-wait t))
         (lambda ()
           (thread-wait t)
           (unbox result)))
       (define copies (map (lambda (finished) (finished)) (list (worker) (worker))))
       (define failure (findf exn? copies))
       (when failure
         (raise failure))
       (forked (child "inner" inner) #t copies))])

  (bounded-test "one copy's child used from two threads at once is refused"
    (check-exn two-threads
               (lambda () (layer-copy (forked (holder (slow)) #f '())))))

  (bounded-test "a worker thread that finishes inside the rebuild joins the copy"
    (define s (holder (slow)))
    (define c (layer-copy (forked s #t '())))
    (match-define (list x y) (forked-copies c))
    (check-eq? x y)
    (check-eq? x (forked-inner c))
    (check-false (eq? x s)))

  (struct stalled (armed ready)
    #:methods gen:layer
    [(define (layer-forward _self . _inputs) 0)
     (define (layer-rebuild self _child _tensor)
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
     (define (layer-rebuild self child _tensor)
       (match-define (abandoned first second stall) self)
       (define t (thread (lambda () (child "first" first))))
       (semaphore-wait (stalled-ready stall))
       (kill-thread t)
       (thread-wait t)
       (abandoned (child "first" first) (child "second" second) stall))])

  (define killed-cases
    (list (list "a hand-written layer" (lambda (s) (values s s)))
          (list "a define-layer layer" (lambda (s) (define h (holder s)) (values h h)))
          (list "a pair two layers hold" (lambda (s) (define p (cons s 'x)) (values (holder p) (holder p))))))

  (memory-bounded-test "a worker thread killed mid-rebuild fails the copy"
    (for ([case (in-list killed-cases)])
      (match-define (list name make) case)
      (define s (stalled (box #t) (make-semaphore 0)))
      (define-values (first second) (make s))
      (check-exn part-built (lambda () (layer-copy (abandoned first second s))) name)))

  (struct gate (in go)
    #:methods gen:layer
    [(define (layer-forward _self . _inputs) 0)
     (define (layer-rebuild self _child _tensor)
       (semaphore-post (gate-in self))
       (semaphore-wait (gate-go self))
       (gate (make-semaphore 0) (make-semaphore 1)))])

  (define (closed-gate)
    (gate (make-semaphore 0) (make-semaphore 0)))

  (define (copying child name x)
    (define result (box #f))
    (values (thread (lambda () (set-box! result (with-handlers ([exn:fail? values]) (child name x)))))
            result))

  (struct both (a b g)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (both-b self))
     (define (layer-rebuild self child _tensor)
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
     (define (layer-rebuild self child _tensor)
       (match-define (leaver inner g) self)
       (copying child "inner" inner)
       (semaphore-wait (gate-in g))
       (leaver #f g))])

  (bounded-test "a rebuild returning while its worker is inside the copy is refused"
    (define g (closed-gate))
    (check-exn two-threads (lambda () (layer-copy (leaver (holder g) g))))
    (semaphore-post (gate-go g))))
