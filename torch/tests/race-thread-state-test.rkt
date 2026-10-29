#lang racket/base

(module+ test
  (require (only-in ffi/unsafe _fun _int64)
           (only-in ffi/unsafe/vm vm-eval)
           (only-in ffi/vector _s64vector s64vector)
           (only-in racket/match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    * @ copy! dtype ones requires-grad? with-autocast
                    with-no-grad zeros)
           (only-in "../foreign/error.rkt" check-handle)
           (only-in "../foreign/raw/linalg.rkt" tr-mm/raw)
           (only-in "../foreign/raw/memory.rkt" oom-retry)
           (only-in "../foreign/raw/syntax.rkt" _Tensor _Tensor/null define-torch)
           (only-in "private/race-harness.rkt"
                    check-known-failure join labelled make-gate release-gate!
                    spawn wait-for-arrival while-inside with-parked with-race-hook
                    worker-thread))

  (define os-thread-id (vm-eval 'get-thread-id))

  (define (registering-thread-id run)
    (define seen (box #f))
    (define (note-thread! name _subject)
      (when (and (eq? name 'handle-registered) ((labelled 'op) #f))
        (set-box! seen (os-thread-id))))
    (with-race-hook note-thread!
      (run (lambda () (void (ones 2 2)))))
    (unbox seen))

  (define ((on-thread pool) thunk)
    (join (spawn thunk #:pool pool #:label 'op)))

  (test-case "an op registers its handle on the calling thread's own OS thread"
    (define main-id (os-thread-id))
    (check-equal? (registering-thread-id (on-thread #f)) main-id "a coroutine thread")
    (check-not-equal? (registering-thread-id (on-thread 'own)) main-id
                      "a parallel thread's op hopped to main's OS thread")
    (check-not-false (registering-thread-id (on-thread 'own)) "no op registered"))

  (define (failure-message thunk)
    (with-handlers ([exn:fail? exn-message])
      (thunk)
      #f))

  (define ((failing-mm rows inner))
    (check-handle 'mm (tr-mm/raw (zeros rows inner) (zeros (add1 inner) 2))))

  (define (kind pool) (or pool 'ordinary))

  (define (clobbered-failure pool)
    (define g (make-gate 'failure-reading #:when (labelled 'a)))
    (with-parked [a (spawn (lambda () (failure-message (failing-mm 2 3)))
                           #:pool pool #:label 'a)
                  #:at g]
      (check-eq? (wait-for-arrival g) 'parked)
      (define other
        (join (spawn (lambda () (failure-message (failing-mm 2 5)))
                     #:pool pool #:label 'b)))
      (release-gate! g)
      (define own (join a))
      (check-regexp-match #rx"2x5 and 6x2" other)
      own))

  (test-case "a failure's message survives another thread's failure"
    (for ([pool (in-list '(#f own))])
      (define own (clobbered-failure pool))
      (define description (format "the parked thread reads its own error (~a)" (kind pool)))
      (define holds? (regexp-match? #rx"2x3 and 4x2" own))
      (if pool
          (check-true holds? description)
          (check-known-failure "#194" description holds?))))

  ;; Bare bindings: the library's own wraps would retry inside, at the same
  ;; race point. The kind is the shim's, which it keeps per OS thread.
  (define-torch zeros/bare
    (_fun (dims : (_s64vector i)) (ndim : _int64) -> _Tensor/null)
    #:c-id tr_zeros)

  (define-torch mm/bare
    (_fun _Tensor _Tensor -> _Tensor/null)
    #:c-id tr_mm)

  (define (out-of-memory)
    (zeros/bare (s64vector (expt 2 22) (expt 2 22)) 2))

  (define ((mismatched-mm a b))
    (mm/bare a b))

  (test-case "a parallel thread's failing mm raises its own message while main fails too"
    (define worker
      (spawn (lambda () (failure-message (failing-mm 2 3))) #:pool 'own))
    (let loop ([n 0])
      (unless (or (> n 200) (thread-dead? (worker-thread worker)))
        (failure-message (failing-mm 2 5))
        (loop (add1 n))))
    (check-regexp-match #rx"2x3 and 4x2" (join worker)))

  (define (clobbered-oom-kind pool)
    (define retried? (box #f))
    (define a-left (zeros 2 3))
    (define a-right (zeros 5 2))
    (define (retrying fail)
      ((oom-retry #:collect! (lambda () (set-box! retried? #t))) fail))
    (define oom-call (retrying out-of-memory))
    (define g
      (make-gate 'oom-kind-reading
                 #:when (lambda (subject) (eq? subject out-of-memory))))
    (with-parked [a (spawn oom-call #:pool pool #:label 'a) #:at g]
      (wait-for-arrival g)
      (join (spawn (retrying (mismatched-mm a-left a-right)) #:pool pool #:label 'b))
      (release-gate! g)
      (join a)
      (unbox retried?)))

  (test-case "an OOM's kind survives another thread's failure"
    (for ([pool (in-list '(#f own))])
      (define description (format "the parked thread retries its own OOM (~a)" (kind pool)))
      (define holds? (clobbered-oom-kind pool))
      (if pool
          (check-true holds? description)
          (check-known-failure "#194" description holds?))))

  (define ((failing-copy width))
    (copy! (zeros 2 2) (zeros width width)))

  (test-case "a failing in-place call on a parallel thread reports its own error"
    (check-regexp-match #rx"tensor b [(]5[)]" (failure-message (failing-copy 5)))
    (define own
      (join (spawn (lambda () (failure-message (failing-copy 3))) #:pool 'own)))
    (check-regexp-match #rx"^copy!: " own)
    (check-regexp-match #rx"tensor b [(]3[)]" own
                        "the error read after a plain call on a parallel thread is that call's"))

  (define x (ones 2 2 #:requires-grad? #t))

  (define (grad-tracked?)
    (requires-grad? (* x 2.0)))

  (define (matmul-dtype)
    (dtype (@ (ones 4 4) (ones 4 4))))

  (define (in-worker pool thunk)
    (join (spawn thunk #:pool pool)))

  (define (no-grad thunk) (with-no-grad (thunk)))
  (define (autocast thunk) (with-autocast #:device 'cpu (thunk)))

  (struct mode (name enter observe outside inside))

  (define modes
    (list (mode 'grad no-grad grad-tracked? #t #f)
          (mode 'autocast autocast matmul-dtype 'float32 'bfloat16)))

  (struct outcome (description holds? known-failure?))

  (define (mode-leaks pool m)
    (match-define (mode name enter observe outside inside) m)
    (define own? (eq? pool 'own))
    (list
     (outcome (format "a worker's ~a mode stays out of the trainer (~a)"
                      name (kind pool))
              (equal? outside
                      (while-inside enter (lambda (_w) (observe)) #:pool pool))
              (not own?))
     (outcome (format "the trainer's ~a mode stays out of a worker (~a)"
                      name (kind pool))
              (equal? outside (enter (lambda () (in-worker pool observe))))
              (not own?))
     (outcome (format "a worker's ~a mode reaches its own ops (~a)"
                      name (kind pool))
              (equal? inside (in-worker pool (lambda () (enter observe))))
              #f)))

  (test-case "grad mode and autocast stay with the thread that set them"
    (check-true (grad-tracked?))
    (check-eq? (matmul-dtype) 'float32)
    (for* ([pool (in-list '(#f own))]
           [m (in-list modes)]
           [o (in-list (mode-leaks pool m))])
      (match-define (outcome description holds? known-failure?) o)
      (cond
        [known-failure? (check-known-failure "#194" description holds?)]
        [else (check-true holds? description)]))
    (check-true (grad-tracked?) "a case left grad mode off")
    (check-eq? (matmul-dtype) 'float32 "a case left autocast on")))
