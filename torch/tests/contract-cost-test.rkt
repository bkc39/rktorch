#lang racket/base

(module+ test
  (require (only-in racket/list drop)
           (only-in racket/match match match-define)
           (only-in racket/path find-relative-path)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in torch add shape tensor tensor? zeros)
           (only-in torch/foreign/structs tensor-free! tensor-impl)
           (only-in "private/blame.rkt" blames-caller?))

  (define-runtime-path torch-dir "..")

  (define combinators
    '(-> ->* ->i and/c between/c box/c cons/c hash/c integer-in list/c listof
      non-empty-listof not/c one-of/c or/c real-in vectorof))

  (define (combinator? head)
    (and (memq head combinators) #t))

  (define (helper-definitions forms)
    (for/fold ([helpers (hasheq)]) ([form (in-list forms)])
      (match form
        [`(define (,(? symbol? name) . ,_) . ,body)
         (hash-update helpers name (lambda (bodies) (cons body bodies)) '())]
        [`(define ,(? symbol? name) (lambda ,_ . ,body))
         (hash-update helpers name (lambda (bodies) (cons body bodies)) '())]
        [_ helpers])))

  ;; a form is searched element by element, so only a head is taken as a call
  (define (builds-a-contract? expr helpers [seen '()])
    (define (builds? e) (builds-a-contract? e helpers seen))
    (match expr
      [(cons (? combinator?) _) #t]
      [(cons (? symbol? head) (? list? args))
       #:when (and (hash-has-key? helpers head) (not (memq head seen)))
       (or (for/or ([body (in-list (hash-ref helpers head))])
             (builds-a-contract? body helpers (cons head seen)))
           (ormap builds? args))]
      [(? list?) (ormap builds? expr)]
      [(cons head tail) (or (builds? head) (builds? tail))]
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
      [(cons part more)
       (cond
         [(keyword? part)
          (dependent-clauses (drop more (hash-ref condition-arity part 0)))]
         [(dependent-clause? part) (cons part (dependent-clauses more))]
         [(list? part) (append (filter dependent-clause? part) (dependent-clauses more))]
         [else (dependent-clauses more)])]))

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

  (define library-files
    (for/list ([f (in-directory (simplify-path torch-dir))]
               #:when (regexp-match? #rx"[.]rkt$" (path->string f))
               #:unless (regexp-match? #rx"/(tests|scribblings)/" (path->string f)))
      f))

  (define library-forms
    (for/hash ([f (in-list library-files)])
      (values f (module-forms f))))

  (define library-helpers
    (helper-definitions (apply append (hash-values library-forms))))

  (define per-call-builds
    (for*/list ([f (in-list library-files)]
                [parts (in-list (arrow-i-forms (hash-ref library-forms f)))]
                [clause (in-list (dependent-clauses parts))]
                #:do [(match-define (list name _ expr) clause)]
                #:when (builds-a-contract? expr library-helpers))
      (list (path->string (find-relative-path (simplify-path torch-dir) f)) name)))

  (check-equal? per-call-builds
                '()
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
    (check-true (builds-a-contract? '(if (tensor? a) (any/c . -> . any/c) none/c)
                                    (hasheq))))

  (test-case "the scanner looks through the helpers a dependent clause calls"
    (define helpers
      (helper-definitions
       '((define (x/c t) (or/c t real?))
         (define (calls-x/c t) (x/c t))
         (define (looped/c t) (if t (looped/c #f) (x/c t)))
         (define (same-rank-index/c t)
           (and/c int64-tensor/c
                  (lambda (x) (= (length (shape x)) (length (shape t))))))
         (define flat/c
           (lambda (t) (flat-named-contract 'flat (lambda (v) (eq? v t))))))))
    (check-true (builds-a-contract? '(if (tensor? a) (x/c a) tensor?) helpers))
    (check-true (builds-a-contract? '(calls-x/c a) helpers))
    (check-true (builds-a-contract? '(looped/c a) helpers))
    (check-true (builds-a-contract? '(if dim (same-rank-index/c t) int64-tensor/c)
                                    helpers))
    (check-false (builds-a-contract? '(flat/c a) helpers))
    (check-false (builds-a-contract? '(if (tensor? a) x/c tensor?) helpers)))

  (test-case "add rejects two reals in its second argument, blaming the caller"
    (define (rejects-b? e)
      (and ((blames-caller? "contract-cost-test") e)
           (regexp-match? #rx"expected: tensor[?]\n *given: 2\n *in: the b argument"
                          (exn-message e))))
    (check-exn rejects-b? (lambda () (add 1 2)))
    (check-equal? (shape (add 1 (zeros 2))) '(2)))

  (test-case "a freed tensor, or one whose handle is not a pointer, is no tensor"
    (define t (tensor '(1.0 2.0)))
    (check-true (tensor? t))
    (tensor-free! t)
    (check-false (tensor? t))
    (check-false (tensor? 1.0))
    (check-false (tensor? (tensor-impl 5 '())))
    (check-false (tensor? (tensor-impl #f '())))))
