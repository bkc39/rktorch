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
         (only-in "promoted.rkt" :: [tensor-ref uncontracted-tensor-ref])
         (only-in "structs.rkt" tensor?))

(provide ref)

;; The macro expands here rather than to promoted.rkt's binding so the
;; macro path carries the same index-spec/c blame as the facade's
;; tensor-ref.
(define/contract tensor-ref
  (-> tensor? index-spec/c ... (or/c tensor? number? boolean?))
  uncontracted-tensor-ref)

(begin-for-syntax
  (define (colon? stx)
    (eq? (syntax-e stx) ':))

  (define (bound-part parts who ctx)
    (cond
      [(null? parts) #f]
      [(null? (cdr parts)) (car parts)]
      [else (raise-syntax-error
             who "expected at most one expression between colons" ctx)]))

  (define (slice-stx spec who)
    (define-values (rev-bounds current)
      (for/fold ([bounds '()] [current '()])
                ([part (in-list (syntax->list spec))])
        (if (colon? part)
            (values (cons (reverse current) bounds) '())
            (values bounds (cons part current)))))
    (define parts
      (for/list ([p (in-list (reverse (cons (reverse current) rev-bounds)))])
        (bound-part p who spec)))
    (define (bound p) (or p #'#f))
    (case (length parts)
      [(2) #`(:: #,(bound (car parts)) #,(bound (cadr parts)))]
      [(3)
       (if (caddr parts)
           #`(:: #,(bound (car parts)) #,(bound (cadr parts))
                 #,(caddr parts))
           #`(:: #,(bound (car parts)) #,(bound (cadr parts))))]
      [else (raise-syntax-error who "too many colons in slice" spec)]))

  (define (parse-spec spec who)
    (syntax-parse spec
      [(~datum :) #'(::)]
      [(~datum ..) #'(quote (... ...))]
      [(~datum _) #'#f]
      [(part ...)
       #:when (ormap colon? (syntax->list spec))
       (slice-stx spec who)]
      [e:expr #'e])))

(define-syntax (ref stx)
  (syntax-parse stx
    [(_ t:expr spec ...)
     #`(tensor-ref t #,@(for/list ([s (in-list (syntax->list #'(spec ...)))])
                          (parse-spec s 'ref)))]))
