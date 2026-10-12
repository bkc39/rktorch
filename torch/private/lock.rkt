#lang racket/base

(require (for-syntax racket/base)
         syntax/parse/define)

(provide call-with-lock
         make-lock
         with-lock)

(struct holder (thread released))

(define (make-lock)
  (box #f))

(define (acquire! lock breaks?)
  (define me (holder (current-thread) (make-semaphore 0)))
  (define wait (if breaks? sync/enable-break sync))
  (let retry ()
    (define held (unbox lock))
    (define free? (or (not held) (thread-dead? (holder-thread held))))
    (cond
      [(and free? (box-cas! lock held me)) (memory-order-acquire) me]
      [free? (retry)]
      [(eq? (holder-thread held) (current-thread))
       (error 'with-lock "the current thread already holds this lock")]
      [else
       (wait (semaphore-peek-evt (holder-released held))
             (thread-dead-evt (holder-thread held)))
       (retry)])))

(define (release! lock me)
  (memory-order-release)
  (set-box! lock #f)
  (semaphore-post (holder-released me)))

;; a killed holder never releases, so a waiter takes the lock over once it is dead
(define (call-with-lock lock thunk)
  (define breaks? (break-enabled))
  (parameterize-break #f
    (define me (acquire! lock breaks?))
    (dynamic-wind void
                  (lambda () (parameterize-break breaks? (thunk)))
                  (lambda () (release! lock me)))))

(define-syntax-parse-rule (with-lock lock:expr body:expr ...+)
  (call-with-lock lock (lambda () body ...)))
