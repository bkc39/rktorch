#lang racket/base

(module+ test
  (require rackunit
           "../private/lock.rkt")

  (define (finishes? thunk)
    (and (sync/timeout 10 (thread thunk)) #t))

  (test-case "with-lock answers its body's values"
    (check-equal? (with-lock (make-lock) (+ 1 2)) 3)
    (check-equal? (call-with-values (lambda () (with-lock (make-lock) (values 1 2)))
                                    list)
                  '(1 2)))

  (define (take-turns pool rounds pause)
    (define lock (make-lock))
    (define count 0)
    (define workers
      (for/list ([_ (in-range 8)])
        (thread
         (lambda ()
           (for ([_ (in-range rounds)])
             (with-lock lock
               (define seen count)
               (pause)
               (set! count (add1 seen)))))
         #:pool pool)))
    (for-each thread-wait workers)
    (check-equal? count (* 8 rounds)))

  (test-case "holders take turns"
    (take-turns #f 50 (lambda () (sleep 0))))

  (test-case "parallel holders take turns"
    (take-turns 'own 4000 (lambda () (for ([_ (in-range 1000)]) (void)))))

  (define (hold lock)
    (define held (make-semaphore 0))
    (define holder
      (thread (lambda ()
                (with-lock lock
                  (semaphore-post held)
                  (sync never-evt)))))
    (semaphore-wait held)
    holder)

  (test-case "a waiter proceeds once the holder releases"
    (define lock (make-lock))
    (define go (make-semaphore 0))
    (define held (make-semaphore 0))
    (define holder
      (thread (lambda ()
                (with-lock lock
                  (semaphore-post held)
                  (semaphore-wait go)))))
    (semaphore-wait held)
    (define waiter (thread (lambda () (with-lock lock 'mine))))
    (check-false (sync/timeout 0.05 waiter))
    (semaphore-post go)
    (check-not-false (sync/timeout 10 waiter))
    (thread-wait holder))

  (test-case "an escape from the body releases the lock"
    (define lock (make-lock))
    (check-exn #rx"from the body"
               (lambda () (with-lock lock (error 'test "from the body"))))
    (check-true (finishes? (lambda () (with-lock lock 'again)))))

  (test-case "a killed holder does not keep the lock"
    (define lock (make-lock))
    (define holder (hold lock))
    (define waiter (thread (lambda () (with-lock lock 'mine))))
    (check-false (sync/timeout 0.05 waiter))
    (kill-thread holder)
    (check-not-false (sync/timeout 10 waiter))
    (check-true (finishes? (lambda () (with-lock lock 'again)))))

  (test-case "two waiters on a killed holder both get the lock in turn"
    (define lock (make-lock))
    (define holder (hold lock))
    (define inside (box 0))
    (define overlaps (box 0))
    (define (waiter)
      (thread (lambda ()
                (with-lock lock
                  (set-box! inside (add1 (unbox inside)))
                  (when (> (unbox inside) 1)
                    (set-box! overlaps (add1 (unbox overlaps))))
                  (sleep 0.02)
                  (set-box! inside (sub1 (unbox inside)))))))
    (define waiters (list (waiter) (waiter)))
    (check-false (sync/timeout 0.05 (apply choice-evt waiters)))
    (kill-thread holder)
    (for ([w (in-list waiters)])
      (check-not-false (sync/timeout 10 w)))
    (check-equal? (unbox overlaps) 0)
    (check-true (finishes? (lambda () (with-lock lock 'again)))))

  (test-case "a suspended holder keeps the lock"
    (define lock (make-lock))
    (define holder (hold lock))
    (thread-suspend holder)
    (define waiter (thread (lambda () (with-lock lock 'mine))))
    (check-false (sync/timeout 0.2 waiter))
    (thread-resume holder)
    (check-false (sync/timeout 0.05 waiter))
    (kill-thread holder)
    (check-not-false (sync/timeout 10 waiter)))

  (test-case "a waiter can be broken while it waits"
    (define lock (make-lock))
    (define holder (hold lock))
    (define outcome (box #f))
    (define waiter
      (thread (lambda ()
                (with-handlers ([exn:break? (lambda (_e) (set-box! outcome 'broken))])
                  (with-lock lock (set-box! outcome 'entered))))))
    (check-false (sync/timeout 0.05 waiter))
    (break-thread waiter)
    (check-not-false (sync/timeout 10 waiter))
    (check-equal? (unbox outcome) 'broken)
    (kill-thread holder)
    (check-true (finishes? (lambda () (with-lock lock 'again)))))

  (test-case "a waiter with breaks disabled is not broken"
    (define lock (make-lock))
    (define holder (hold lock))
    (define waiter
      (thread (lambda ()
                (parameterize-break #f
                  (with-lock lock 'entered)))))
    (check-false (sync/timeout 0.05 waiter))
    (break-thread waiter)
    (check-false (sync/timeout 0.05 waiter))
    (kill-thread holder)
    (check-not-false (sync/timeout 10 waiter)))

  (test-case "the body runs with the caller's breaks"
    (check-true (with-lock (make-lock) (break-enabled)))
    (check-false (parameterize-break #f (with-lock (make-lock) (break-enabled)))))

  (test-case "taking a lock the current thread holds is an error"
    (define lock (make-lock))
    (check-exn #rx"with-lock: the current thread already holds this lock"
               (lambda () (with-lock lock (with-lock lock 'inner))))
    (check-true (finishes? (lambda () (with-lock lock 'again))))))
