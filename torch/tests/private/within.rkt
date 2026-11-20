#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/function const)
         (only-in rackunit fail test-case)
         (only-in syntax/parse/define define-syntax-parse-rule))

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
;; running parallel thread (#345); kill-thread does not, so threads go first.
(define (kill-threads-under! c super #:except [spared #f])
  (let kill-all ([rounds 10])
    (define ts (remq spared (threads-under c super)))
    (for-each kill-thread ts)
    (for ([t (in-list ts)]) (sync/timeout 5 (thread-dead-evt t)))
    (when (and (pair? ts) (positive? rounds))
      (kill-all (sub1 rounds)))))

(define (shut-down-body! c super)
  (kill-threads-under! c super)
  (custodian-shutdown-all c))

(define (unwind-body! body c super)
  (kill-threads-under! c super #:except body)
  (break-thread body)
  (sync/timeout 10 (thread-dead-evt body))
  (shut-down-body! c super))

(define (within seconds thunk)
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
     (unwind-body! t body-custodian super)
     (fail (format "no result within ~a s" seconds))]))

(define-syntax-parse-rule (bounded-test name:expr body:expr ...+)
  (test-case name (within 60 (lambda () body ...))))
