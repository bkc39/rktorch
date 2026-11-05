#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre ...+ expr))
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide claimed
         claims
         claims-who
         copied
         finished!
         recorded-count
         remember!
         settled!
         with-closed
         with-fill)

(struct claims (who done open threads starter lock [state #:auto #:mutable])
  #:auto-value 'live
  #:constructor-name build-claims
  #:omit-define-syntaxes)

(define (claims who)
  (build-claims who (make-hasheq) (make-hasheq) (make-hasheq) (current-thread) (make-semaphore 1)))

(define-syntax-parse-rule (with-closed c:expr body:expr ...+)
  (let ([cs c])
    (dynamic-wind void (lambda () body ...) (lambda () (close! cs)))))

(define-syntax-parse-rule (with-claim c:expr v:expr body:expr ...+)
  (let ([cs c] [x v])
    (dynamic-wind void (lambda () body ...) (lambda () (release! cs x)))))

(define-syntax-parse-rule (with-fill c:expr body:expr ...+)
  (let ([cs c] [filled? (box #f)])
    (dynamic-wind void
                  (lambda () (begin0 (let () body ...) (set-box! filled? #t)))
                  (lambda () (unless (unbox filled?) (set-claims-state! cs 'failed))))))

(define (locked c thunk)
  (call-with-semaphore (claims-lock c) thunk))

(define (close! c)
  (set-claims-state! c 'closed)
  (hash-clear! (claims-done c))
  (hash-clear! (claims-open c)))

(define (check-live! c)
  (match (claims-state c)
    ['live (void)]
    ['failed
     (raise-arguments-error
      (claims-who c)
      (string-append "a rebuild raised, escaped or was killed while a copy was part-built,"
                     " so nothing more can be copied"))]
    ['closed
     (raise-arguments-error (claims-who c) "a child callback used after layer-copy returned")]))

(define (abandon! c)
  (set-claims-state! c 'failed)
  (check-live! c))

(define (check-owner! c)
  (define me (current-thread))
  (define others
    (for/list ([t (in-hash-keys (claims-threads c))] #:unless (eq? t me))
      t))
  (check-live! c)
  (cond
    [(ormap thread-dead? others) (abandon! c)]
    [(not (member others (list '() (list (claims-starter c)))))
     (raise-arguments-error (claims-who c) "a child callback used from two threads at once")]))

(define (claim! c v)
  (check-owner! c)
  (define open (claims-open c))
  (define copy (hash-ref (claims-done c) v #f))
  (cond
    [copy (cons 'done copy)]
    [(hash-ref open v #f) 'open]
    [else
     (hash-set! open v #t)
     (hash-update! (claims-threads c) (current-thread) add1 0)
     'claimed]))

(define (release! c v)
  (define threads (claims-threads c))
  (define me (current-thread))
  (define (released)
    (hash-remove! (claims-open c) v)
    (hash-update! threads me sub1 1)
    (when (zero? (hash-ref threads me))
      (hash-remove! threads me)))
  (locked c released))

(define (claimed c v build)
  (match (locked c (lambda () (claim! c v)))
    [(cons 'done copy) copy]
    ['open
     (raise-arguments-error (claims-who c)
                            "a cycle re-enters a hand-written layer-rebuild before it returns"
                            "layer" v)]
    ['claimed (with-claim c v (build))]))

(define (copied c v default)
  (check-live! c)
  (hash-ref (claims-done c) v default))

(define (recorded-count c)
  (hash-count (claims-done c)))

(define (remember! c v copy)
  (hash-set! (claims-done c) v copy)
  copy)

(define (settled! c v copy)
  (locked c (lambda () (check-owner! c)))
  (remember! c v copy))

(define (finished! c)
  (locked c (lambda () (check-owner! c))))
