#lang racket/base

;; Slicing tokens are recognized by DATUM, not by binding: exporting `:`
;; would collide with Typed Racket's and `_` with match's wildcard, and
;; a literal `...` token is impossible (it is the macro ellipsis, so any
;; macro whose template contained a ref form would break) — hence `..`.
;; whole-module for-syntax requires: macro-expansion exemption, as in
;; define-generated.rkt (AGENTS.md import convention)
(require (for-syntax racket/base
                     syntax/parse/pre)
         (only-in racket/contract/base -> or/c)
         (only-in racket/contract/region define/contract)
         (only-in "contracts.rkt" index-spec/c)
         (only-in "promoted.rkt"
                  ::
                  [tensor-ref uncontracted-tensor-ref]
                  [tensor-ref! uncontracted-tensor-ref!])
         (only-in "structs.rkt" tensor?))

(provide ref ref!)

;; The macro expands here rather than to promoted.rkt's binding so the
;; macro path carries the same index-spec/c blame as the facade's
;; tensor-ref.
(define/contract tensor-ref
  (-> tensor? index-spec/c ... (or/c tensor? number? boolean?))
  uncontracted-tensor-ref)

(define/contract tensor-ref!
  (-> tensor? (or/c tensor? real?) index-spec/c ... void?)
  uncontracted-tensor-ref!)

(begin-for-syntax
  (define (slice-arg stx)
    (syntax-parse stx
      [(~datum _) #'#f]
      [e:expr #'e]))

  (define (parse-spec spec who)
    (syntax-parse spec
      [(~datum :) #'(::)]
      [(~datum ..) #'(quote (... ...))]
      [(~datum _) #'#f]
      [((~datum :) arg ...)
       (define args (map slice-arg (syntax->list #'(arg ...))))
       (syntax-parse spec
         [(_ b) #`(:: #,(car args))]
         [(_ a b) #`(:: #,@args)]
         [(_ a b (~datum _)) #`(:: #,(car args) #,(cadr args))]
         [(_ a b s) #`(:: #,@args)]
         [_ (raise-syntax-error
             who "expected 1 to 3 slice arguments" spec)])]
      [((~datum :~) a) #`(:: #,(slice-arg #'a) #f)]
      [((~datum :~) a (~datum _)) #`(:: #,(slice-arg #'a) #f)]
      [((~datum :~) a s) #`(:: #,(slice-arg #'a) #f s)]
      [((~datum :~) . _)
       (raise-syntax-error
        who "expected 1 or 2 arguments (start [step])" spec)]
      [e:expr #'e])))

(define-syntax (ref stx)
  (syntax-parse stx
    [(_ t:expr spec ...)
     #`(tensor-ref t #,@(for/list ([s (in-list (syntax->list #'(spec ...)))])
                          (parse-spec s 'ref)))]))

(define-syntax (ref! stx)
  (syntax-parse stx
    [(_ t:expr spec ... v:expr)
     (define parsed
       (for/list ([s (in-list (syntax->list #'(spec ...)))])
         (parse-spec s 'ref!)))
     (define tmps (generate-temporaries parsed))
     #`(let* ([target t]
              #,@(for/list ([tmp (in-list tmps)] [e (in-list parsed)])
                   #`[#,tmp #,e])
              [val v])
         (tensor-ref! target val #,@tmps))]))
