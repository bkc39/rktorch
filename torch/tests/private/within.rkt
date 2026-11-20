#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/function const)
         (only-in rackunit fail test-case)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "wedge.rkt" call-with-process-deadline))

(provide bounded-test
         shut-down-body!
         within)

(define (threads-under c super)
  (for*/list ([v (in-list (custodian-managed-list c super))]
              [t (in-list (cond
                            [(thread? v) (list v)]
                            [(custodian? v) (threads-under v super)]
                            [else '()]))])
    t))

;; custodian-shutdown-all can hang the place while the custodian holds a
;; running parallel thread (#345), so threads are killed first. kill-thread
;; on a parallel thread inside a foreign call blocks the whole place until
;; the call returns, so every stop runs under a process deadline.
(define (kill-threads-under! c super #:except [spared #f])
  (let kill-all ([rounds 10])
    (define ts (remq spared (threads-under c super)))
    (for-each kill-thread ts)
    (for ([t (in-list ts)]) (sync/timeout 5 (thread-dead-evt t)))
    (when (and (pair? ts) (positive? rounds))
      (kill-all (sub1 rounds)))))

(define (shut-down! c super)
  (kill-threads-under! c super)
  (custodian-shutdown-all c))

(define (shut-down-body! c super
                         #:what [what "stopping a body's threads"]
                         #:seconds [seconds 60])
  (call-with-process-deadline seconds (lambda () (shut-down! c super)) #:what what))

(define (unwind-body! body c super)
  (kill-threads-under! c super #:except body)
  (break-thread body)
  (sync/timeout 10 (thread-dead-evt body))
  (shut-down! c super))

(define (within seconds thunk #:unwind-seconds [unwind-seconds 60])
  (define super (current-custodian))
  (define body-custodian (make-custodian))
  (define outcome #f)
  (define t
    (parameterize ([current-custodian body-custodian])
      (thread (lambda ()
                (set! outcome
                      (with-handlers ([(const #t) (lambda (v) (lambda () (raise v)))])
                        (call-with-values thunk
                                          (lambda vs (lambda () (apply values vs))))))))))
  (cond
    [(sync/timeout seconds t) (outcome)]
    [else
     (call-with-process-deadline
      unwind-seconds
      (lambda () (unwind-body! t body-custodian super))
      #:what (format "stopping the threads of a body with no result within ~a s" seconds))
     (fail (format "no result within ~a s" seconds))]))

(define-syntax-parse-rule (bounded-test name:expr body:expr ...+)
  (test-case name (within 60 (lambda () body ...))))
