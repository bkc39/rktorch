#lang racket/base

(require (for-syntax racket/base)
         syntax/parse/define)

(provide call-with-lock
         make-lock
         with-lock)

(struct holder (thread released))

(define (make-lock)
  (box #f))

(define (acquire! lock)
  (define me (holder (current-thread) (make-semaphore 0)))
  (let retry ()
    (define held (unbox lock))
    (define free? (or (not held) (thread-dead? (holder-thread held))))
    (cond
      [(and free? (box-cas! lock held me)) me]
      [free? (retry)]
      [else
       (sync (semaphore-peek-evt (holder-released held))
             (thread-dead-evt (holder-thread held)))
       (retry)])))

(define (release! lock me)
  (set-box! lock #f)
  (semaphore-post (holder-released me)))

;; a killed holder never releases, so a waiter takes the lock over once it is dead
(define (call-with-lock lock thunk)
  (define me #f)
  (dynamic-wind (lambda () (set! me (acquire! lock)))
                thunk
                (lambda () (release! lock me))))

(define-syntax-parse-rule (with-lock lock:expr body:expr ...+)
  (call-with-lock lock (lambda () body ...)))
