#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre ...+ expr))
         (only-in racket/match == match)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide claimed
         claims
         claims-who
         copied
         finished!
         locked
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
  (build-claims who (make-hasheq) (make-hasheq) (make-hasheq) (current-thread)
                (make-semaphore 1)))

(define (call-with-exit body on-exit)
  (define returned? #f)
  (dynamic-wind void
                (lambda () (begin0 (body) (set! returned? #t)))
                (lambda () (on-exit returned?))))

(define-syntax-parse-rule (with-closed c:expr body:expr ...+)
  (let ([cs c])
    (call-with-exit (lambda () body ...) (lambda (_returned?) (close! cs)))))

(define-syntax-parse-rule (with-claim c:expr v:expr body:expr ...+)
  (let ([cs c] [x v])
    (call-with-exit (lambda () body ...) (lambda (_returned?) (release! cs x)))))

(define-syntax-parse-rule (with-fill c:expr body:expr ...+)
  (let ([cs c])
    (call-with-exit (lambda () body ...)
                    (lambda (returned?) (unless returned? (set-claims-state! cs 'failed))))))

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
  (check-live! c)
  (match (for/list ([t (in-hash-keys (claims-threads c))] #:unless (eq? t me)) t)
    [(list _ ... (? thread-dead?) _ ...) (abandon! c)]
    ['() (void)]
    [(list (== (claims-starter c) eq?)) (void)]
    [_ (raise-arguments-error (claims-who c) "a child callback used from two threads at once")]))

(define (alone? c)
  (define threads (claims-threads c))
  (and (match (hash-count threads)
         [0 #t]
         [1 (hash-has-key? threads (current-thread))]
         [_ #f])
       (eq? (claims-state c) 'live)))

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
  (define done (hash-ref (claims-done c) v #f))
  (or (and done (alone? c) done)
      (match (locked c (lambda () (claim! c v)))
        [(cons 'done copy) copy]
        ['open
         (raise-arguments-error
          (claims-who c) "a cycle re-enters a hand-written layer-rebuild-method before it returns"
          "layer" v)]
        ['claimed (with-claim c v (build))])))

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
