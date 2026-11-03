#lang racket/base

(provide memo
         remembered
         remembering
         remembering?
         remembering-seen)

(struct memo (who done open)
  #:constructor-name make-memo
  #:omit-define-syntaxes)

(struct remembering (seen proc)
  #:property prop:procedure (struct-field-index proc))

(define (memo who)
  (make-memo who (make-hasheq) (make-hasheq)))

(define (remembered seen v build #:field [field #f])
  (define done (memo-done seen))
  (define open (memo-open seen))
  (cond
    [(hash-has-key? done v) (hash-ref done v)]
    [(hash-has-key? open v)
     (raise-arguments-error (memo-who seen)
                            "a field's value reaches itself, and a cycle cannot be copied"
                            "field" field)]
    [else
     (define made
       (dynamic-wind (lambda () (hash-set! open v #t))
                     build
                     (lambda () (hash-remove! open v))))
     (hash-set! done v made)
     made]))
