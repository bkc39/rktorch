#lang racket/base

(module+ test
  (require (only-in ffi/unsafe/atomic call-as-atomic)
           (only-in racket/list count)
           (only-in racket/match match struct*)
           rackunit
           (only-in "../foreign.rkt" conv2d cpu-device cuda-device native-memory-use randn zeros)
           (only-in (submod "../foreign.rkt" unsafe) to!)
           (only-in "../foreign/raw/collector.rkt" collect-and-wait! finalizer-runs)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" within))

  (struct gc-info (mode pre-amount pre-admin-amount code-amount
                   post-amount post-admin-amount
                   start-process-time end-process-time
                   start-time end-time)
    #:prefab)

  (define calls 200)
  (define result-bytes (* 128 16 32 32 4))

  (define (cpu-bytes)
    (cond [(assoc (cpu-device) (native-memory-use)) => cdr]
          [else 0]))

  (define (event-mode event)
    (match event
      [(vector _ _ (struct* gc-info ([mode mode])) _) mode]
      [_ #f]))

  (define (drain-modes receiver)
    (for/list ([event (in-producer (lambda () (sync/timeout 0 receiver)) #f)])
      (event-mode event)))

  (define (major? mode) (memq mode '(major major-incremental)))
  (define (minor? mode) (memq mode '(minor minor-incremental)))

  (define (check-conv-loop-releases)
    (define x (randn 128 3 32 32))
    (define w (randn 16 3 3 3))
    (define b (randn 16))
    (define receiver (make-log-receiver (current-logger) 'debug 'GC))
    (for ([_ (in-range 3)]) (collect-and-wait!))
    (drain-modes receiver)
    (define base (cpu-bytes))
    (define runs-before (finalizer-runs))
    (for ([_ (in-range calls)])
      (conv2d x w #:bias b #:padding 1))
    (define modes (drain-modes receiver))
    (define majors (count major? modes))
    (define minors (count minor? modes))
    (define held (- (cpu-bytes) base))
    (define released (- (finalizer-runs) runs-before))
    (define (summary what)
      (format "~a: ~a MiB still charged, ~a of ~a released, ~a majors, ~a minors"
              what (quotient held 1048576) released calls majors minors))
    (check-true (> minors 0) (summary "the receiver saw no collection"))
    (check-true (<= held (* 4 result-bytes)) (summary "dead results stayed charged"))
    (check-true (>= released (- calls 4)) (summary "results were not released"))
    (check-true (<= majors 1) (summary "the backlog ran majors")))

  (test-case "a loop of 8 MiB convs releases each result within a collection (#377)"
    (within 120 check-conv-loop-releases))

  (define (ran-during? op)
    (define ran? (box #f))
    (thread (lambda () (set-box! ran? #t)))
    (op)
    (unbox ran?))

  (define (yields-after-a-parallel-turn?)
    (collect-garbage 'minor)
    (thread-wait (thread (lambda () (zeros 1)) #:pool 'own))
    (ran-during? (lambda () (zeros 1))))

  (test-case "a parallel thread's turn after a collection leaves this thread its own (#377)"
    (within 60
            (lambda ()
              (define trials 20)
              (define yielded (count values (for/list ([_ (in-range trials)])
                                              (yields-after-a-parallel-turn?))))
              (check-equal? yielded trials
                            (format "this thread yielded after ~a of ~a collections"
                                    yielded trials)))))

  (test-case "after a collection a tensor op lets a waiting green thread run"
    (within 60
            (lambda ()
              (define x (randn 128 3 32 32))
              (define w (randn 16 3 3 3))
              (define done (box 0))
              (define ran-after (box #f))
              (collect-garbage 'minor)
              (thread (lambda () (set-box! ran-after (unbox done))))
              (for ([i (in-range 50)] #:break (unbox ran-after))
                (conv2d x w #:padding 1)
                (set-box! done (add1 i)))
              (check-equal? (unbox ran-after) 0
                            "the waiting thread did not run during the first op"))))

  (test-case "a break taken in to!'s re-accounting leaves the move's own error"
    (within 60
            (lambda ()
              (define t (zeros 4))
              (define mover (current-thread))
              (define (break-in-reaccounting name _subject)
                (when (and (eq? name 'failure-reading) (eq? (current-thread) mover))
                  (collect-garbage 'minor)
                  (thread (lambda () (break-thread mover)))))
              (define raised (box #f))
              (with-handlers ([exn:break? void])
                (with-handlers ([(lambda (_) #t) (lambda (e) (set-box! raised e))])
                  (with-race-hook break-in-reaccounting
                    (to! t (cuda-device 63))))
                (sleep 0))
              (check-false (exn:break? (unbox raised))
                           "the break replaced the failed move's error")
              (check-pred exn:fail? (unbox raised)
                          "the move to a missing device did not fail"))))

  (test-case "an op inside an atomic section after a collection does not yield"
    (within 60
            (lambda ()
              (check-not-exn
               (lambda ()
                 (call-as-atomic
                  (lambda ()
                    (collect-garbage 'minor)
                    (zeros 1))))
               "the op slept inside an atomic section")))))
