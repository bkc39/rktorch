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
    (check-exn #rx"^layer-copy: a child callback used from two threads at once"
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

  (struct abandoned (inner)
    #:methods gen:layer
    [(define (layer-forward self . _inputs) (abandoned-inner self))
     (define (layer-rebuild self child _tensor)
       (define inner (abandoned-inner self))
       (define t (thread (lambda () (child "inner" inner))))
       (semaphore-wait (stalled-ready inner))
       (kill-thread t)
       (thread-wait t)
       (abandoned (child "inner" inner)))])

  (memory-bounded-test "a worker thread killed mid-rebuild leaves the layer free to rebuild"
    (define s (stalled (box #t) (make-semaphore 0)))
    (define inner (~> s abandoned layer-copy abandoned-inner))
    (check-pred stalled? inner)
    (check-false (eq? inner s))))
