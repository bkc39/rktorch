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

  (test-case "holders take turns"
    (define lock (make-lock))
    (define count 0)
    (define workers
      (for/list ([_ (in-range 8)])
        (thread
         (lambda ()
           (for ([_ (in-range 50)])
             (with-lock lock
               (define seen count)
               (sleep 0)
               (set! count (add1 seen))))))))
    (for-each thread-wait workers)
    (check-equal? count 400))

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
    (define held (make-semaphore 0))
    (define holder
      (thread (lambda ()
                (with-lock lock
                  (semaphore-post held)
                  (sync never-evt)))))
    (semaphore-wait held)
    (define waiter (thread (lambda () (with-lock lock 'mine))))
    (check-false (sync/timeout 0.05 waiter))
    (kill-thread holder)
    (check-not-false (sync/timeout 10 waiter))
    (check-true (finishes? (lambda () (with-lock lock 'again))))))
