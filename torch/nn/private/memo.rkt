#lang racket/base

(require (for-syntax racket/base
                     (only-in syntax/parse/pre ...+ expr id))
         (only-in racket/match match)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "claim.rkt"
                  claimed claims claims-who copied finished! locked recorded-count remember!
                  settled! with-closed with-fill)
         (only-in "hash-keys.rkt" filled! filling! key-check))

(provide after-walk!
         allocated
         copy-of
         memo-keys
         memo-who
         rebuilt-immutable
         remembered
         remembering
         remembering-seen
         remembering?
         with-memo)

(struct memo (claims entered keys [afters #:mutable])
  #:constructor-name build-memo
  #:omit-define-syntaxes)

(define (memo who)
  (build-memo (claims who) (make-hasheq) (key-check) '()))

(struct remembering (seen proc)
  #:property prop:procedure
  (lambda (self name c)
    ((remembering-proc self) (remembering-seen self) self name c)))

(define-syntax-parse-rule (with-memo (seen:id who:expr) body:expr ...+)
  (let ([seen (memo who)])
    (with-closed (memo-claims seen)
      (begin0 (let () body ...)
              (finish! seen)))))

(define (memo-who seen)
  (claims-who (memo-claims seen)))

(define (copy-of seen v default)
  (copied (memo-claims seen) v default))

(define (after-walk! seen thunk)
  (locked (memo-claims seen)
          (lambda () (set-memo-afters! seen (cons thunk (memo-afters seen))))))

(define (finish! seen)
  (finished! (memo-claims seen))
  (for ([after (in-list (reverse (memo-afters seen)))])
    (after)))

(define (remembered seen v build)
  (define c (memo-claims seen))
  (claimed c v (lambda () (settled! c v (build)))))

(define (allocated seen v allocate fill!)
  (define c (memo-claims seen))
  (define keys (memo-keys seen))
  (define (filled)
    (define made (remember! c v (allocate)))
    (with-fill c
      (filling! keys made)
      (fill! made)
      (filled! keys made))
    made)
  (claimed c v filled))

(define (rebuilt-immutable seen field v build)
  (define since (recorded-count (memo-claims seen)))
  (define cyclic?
    (match (hash-ref (memo-entered seen) v '())
      [(cons latest _) (= latest since)]
      ['() #f]))
  (define recorded (copy-of seen v #f))
  (cond
    [recorded recorded]
    [cyclic?
     (raise-arguments-error (memo-who seen) "a cycle of immutable values cannot be copied"
                            "field" field)]
    [else (walked seen v since build)]))

(define (walked seen v since build)
  (define entered (memo-entered seen))
  (hash-update! entered v (lambda (counts) (cons since counts)) '())
  (define copy (build))
  (match (hash-ref entered v)
    [(list _) (hash-remove! entered v)]
    [(cons _ outer) (hash-set! entered v outer)])
  (or (copy-of seen v #f)
      (remember! (memo-claims seen) v copy)))
