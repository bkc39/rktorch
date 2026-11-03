#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule))

(provide memo
         remembered
         remembering?
         remembering-memo
         with-remembering)

(struct memo (who done open)
  #:constructor-name make-memo
  #:omit-define-syntaxes)

(struct remembering ([seen #:mutable] proc)
  #:property prop:procedure
  (lambda (self name c)
    (remembering-memo self)
    ((remembering-proc self) self name c)))

(define (memo who)
  (make-memo who (make-hasheq) (make-hasheq)))

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

(define (remembered seen v build #:field [field #f] #:reentered [reentered #f])
  (define done (memo-done seen))
  (define open (memo-open seen))
  (cond
    [(hash-has-key? done v) (hash-ref done v)]
    [(and (hash-has-key? open v) reentered) (reentered)]
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
