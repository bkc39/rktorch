#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide memo
         remembered
         remembering?
         remembering-memo
         with-remembering)

(struct memo (who done open lock)
  #:constructor-name make-memo
  #:omit-define-syntaxes)

(struct remembering ([seen #:mutable] proc)
  #:property prop:procedure
  (lambda (self name c)
    (remembering-memo self)
    ((remembering-proc self) self name c)))

(define (memo who)
  (make-memo who (make-hasheq) (make-hasheq) (make-semaphore 1)))

(define (remembering-memo r)
  (or (remembering-seen r)
      (raise-arguments-error 'layer-copy
                             "a child callback used after layer-copy returned")))

(define (call-with-remembering seen proc body)
  (define r (remembering seen proc))
  (dynamic-wind void
                (lambda () (body r))
                (lambda ()
                  (set-remembering-seen! r #f)
                  (hash-clear! (memo-done seen))
                  (hash-clear! (memo-open seen)))))

(define-syntax-parse-rule (with-remembering ([child:id seen:expr proc:expr]) body:expr ...+)
  (call-with-remembering seen proc (lambda (child) body ...)))

(define (claim seen v)
  (define open (memo-open seen))
  (define (claimed)
    (define owner (hash-ref open v #f))
    (cond
      [(hash-has-key? (memo-done seen) v) 'done]
      [(or (not owner) (thread-dead? owner))
       (hash-set! open v (current-thread))
       'claimed]
      [(eq? owner (current-thread)) 'reentered]
      [else 'elsewhere]))
  (call-with-semaphore (memo-lock seen) claimed))

(define (built seen v build)
  (define (finished)
    (define made (build))
    (hash-set! (memo-done seen) v made)
    made)
  (dynamic-wind void finished (lambda () (hash-remove! (memo-open seen) v))))

(define (remembered seen v build #:field [field #f] #:reentered [reentered #f])
  (define state (claim seen v))
  (cond
    [(eq? state 'done) (hash-ref (memo-done seen) v)]
    [(eq? state 'claimed) (built seen v build)]
    [(eq? state 'elsewhere)
     (raise-arguments-error (memo-who seen) "a child callback used from two threads at once")]
    [reentered (reentered)]
    [else
     (raise-arguments-error (memo-who seen)
                            "a field's value reaches itself, and a cycle cannot be copied"
                            "field" field)]))
