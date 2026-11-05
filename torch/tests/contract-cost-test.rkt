#lang racket/base

(module+ test
  (require (only-in racket/list drop)
           (only-in racket/match match match-define)
           (only-in racket/path find-relative-path)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in torch add shape tensor tensor? zeros)
           (only-in torch/foreign/structs tensor-free!)
           (only-in "private/blame.rkt" blames-caller?))

  (define-runtime-path torch-dir "..")

  (define combinators
    '(-> ->* ->i and/c between/c box/c cons/c hash/c integer-in list/c listof
      non-empty-listof not/c one-of/c or/c real-in vectorof))

  (define (builds-a-contract? expr)
    (match expr
      [(cons (? (lambda (head) (memq head combinators))) _) #t]
      [(cons head tail) (or (builds-a-contract? head) (builds-a-contract? tail))]
      [_ #f]))

  (define (dependencies? v)
    (and (list? v) (andmap symbol? v)))

  (define (dependent-clause? v)
    (match v
      [(list (? symbol? (not 'values)) (? dependencies?) _) #t]
      [_ #f]))

  (define condition-arity
    (hasheq '#:pre 2 '#:pre/desc 2 '#:post 2 '#:post/desc 2 '#:pre/name 3 '#:post/name 3))

  ;; the dependent clauses of one `->i`, its pre- and postconditions skipped
  (define (dependent-clauses parts)
    (match parts
      ['() '()]
      [(cons (? keyword? k) more)
       (dependent-clauses (drop more (hash-ref condition-arity k 0)))]
      [(cons (? dependent-clause? clause) more) (cons clause (dependent-clauses more))]
      [(cons (? list? group) more)
       (append (filter dependent-clause? group) (dependent-clauses more))]
      [(cons _ more) (dependent-clauses more)]))

  (define (arrow-i-forms v)
    (match v
      [(cons '->i parts) (cons parts (arrow-i-forms parts))]
      [(cons head tail) (append (arrow-i-forms head) (arrow-i-forms tail))]
      [_ '()]))

  (define (module-forms f)
    (match-define `(module ,_ ,_ (#%module-begin ,forms ...))
      (parameterize ([read-accept-reader #t] [read-accept-lang #t])
        (call-with-input-file f read)))
    forms)

  (define per-call-builds
    (for*/list ([f (in-directory (simplify-path torch-dir))]
                #:when (regexp-match? #rx"[.]rkt$" (path->string f))
                #:unless (regexp-match? #rx"/(tests|scribblings)/" (path->string f))
                [parts (in-list (arrow-i-forms (module-forms f)))]
                [clause (in-list (dependent-clauses parts))]
                #:when (builds-a-contract? (caddr clause)))
      (list (path->string (find-relative-path (simplify-path torch-dir) f))
            (car clause))))

  (check-equal? per-call-builds
                '(("data/loader.rkt" more)
                  ("vision/augment.rkt" flip))
                (string-append
                 "a dependent ->i clause builds a combinator on every call;"
                 " select among contracts built once at module level"))

  (test-case "the scanner finds a combinator built in a dependent clause"
    (check-equal?
     (dependent-clauses
      '(([a (or/c tensor? real?)] [b (a) (if (tensor? a) (or/c tensor? real?) tensor?)])
        #:pre/name (a b) "a precondition is not a clause" (list (x (y) z))
        (values [top tensor?] [rest (or/c #f tensor?)])))
     '([b (a) (if (tensor? a) (or/c tensor? real?) tensor?)]))
    (check-true (builds-a-contract? '(if (tensor? a) (any/c . -> . any/c) none/c))))

  (test-case "add rejects two reals in its second argument, blaming the caller"
    (define (rejects-b? e)
      (and ((blames-caller? "contract-cost-test") e)
           (regexp-match? #rx"expected: tensor[?]\n *given: 2\n *in: the b argument"
                          (exn-message e))))
    (check-exn rejects-b? (lambda () (add 1 2)))
    (check-equal? (shape (add 1 (zeros 2))) '(2)))

  (test-case "a freed tensor is no longer a tensor"
    (define t (tensor '(1.0 2.0)))
    (check-true (tensor? t))
    (tensor-free! t)
    (check-false (tensor? t))
    (check-false (tensor? 1.0))))
