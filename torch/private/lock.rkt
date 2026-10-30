#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr))
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign/raw/cas.rkt" cas-claim! cas-claim-when!))

(provide call-with-lock
         make-lock
         with-lock)

(struct holder (thread released))

(define (make-lock)
  (box #f))

(define (free? held)
  (or (not held) (thread-dead? (holder-thread held))))

(define (acquire! lock breaks?)
  (define me (holder (current-thread) (make-semaphore 0)))
  (define wait (if breaks? sync/enable-break sync))
  (let retry ()
    (cond
      [(cas-claim-when! lock free? me #:site 'lock) (memory-order-acquire) me]
      [else
       (define held (unbox lock))
       (cond
         [(free? held) (retry)]
         [(eq? (holder-thread held) (current-thread))
          (error 'with-lock "the current thread already holds this lock")]
         [else
          (wait (semaphore-peek-evt (holder-released held))
                (thread-dead-evt (holder-thread held)))
          (retry)])])))

(define (release! lock me)
  (memory-order-release)
  (cas-claim! lock me #f #:site 'lock)
  (semaphore-post (holder-released me)))

;; a killed holder never releases, so a waiter takes the lock over once it is dead
(define (call-with-lock lock thunk)
  (define breaks? (break-enabled))
  (parameterize-break #f
    (define me (acquire! lock breaks?))
    (dynamic-wind void
                  (lambda ()
                    (call-with-continuation-barrier
                     (lambda () (parameterize-break breaks? (thunk)))))
                  (lambda () (release! lock me)))))

(define-syntax-parse-rule (with-lock lock:expr body:expr ...+)
  (call-with-lock lock (lambda () body ...)))

(module+ private
  (provide acquire! release!))
