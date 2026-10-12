#lang racket/base

(require (only-in racket/treelist in-treelist treelist-length)
         (only-in resyntax/base
                  define-refactoring-rule define-refactoring-suite)
         (only-in resyntax/default-recommendations default-recommendations)
         (only-in resyntax/grimoire/syntax-path
                  syntax-label-paths syntax-path-elements)
         (only-in resyntax/grimoire/syntax-property-bundle
                  sequence->syntax-property-bundle syntax-property-entry)
         (only-in resyntax/private/analyzer make-expansion-analyzer)
         (only-in resyntax/private/syntax-neighbors syntax-original-path)
         (only-in syntax/parse/pre
                  attribute boolean define-syntax-class expr id keyword
                  number pattern str syntax-parse this-syntax ~and)
         (only-in threading ~>)
         (only-in torch ge gt le lt)
         (prefix-in torch: (only-in torch * + < <= > >=)))

(provide rktorch-rules rktorch-recommendations)

(define-syntax-class named-ordering
  #:attributes (operator)
  #:literals (ge gt le lt)
  (pattern (~and name ge)
    #:with operator (datum->syntax #'name '>= #'name)
    #:when (free-identifier=? #'operator #'torch:>=))
  (pattern (~and name gt)
    #:with operator (datum->syntax #'name '> #'name)
    #:when (free-identifier=? #'operator #'torch:>))
  (pattern (~and name le)
    #:with operator (datum->syntax #'name '<= #'name)
    #:when (free-identifier=? #'operator #'torch:<=))
  (pattern (~and name lt)
    #:with operator (datum->syntax #'name '< #'name)
    #:when (free-identifier=? #'operator #'torch:<)))

(define-refactoring-rule named-comparison-to-operator
  #:description
  "Compare with torch's operators: `>=`, `>`, `<=` and `<` dispatch to `ge`, `gt`, `le` and `lt`."
  (comparison:named-ordering a:expr b:expr)
  (comparison.operator a b))

(define core-form-heads ;; noqa
  '(quote quasiquote syntax quasisyntax unquote unquote-splicing unsyntax
    unsyntax-splicing define define-values define-syntax define-syntaxes
    define-syntax-rule define-syntax-parse-rule lambda λ case-lambda let let*
    letrec let-values let*-values letrec-values if cond case when unless begin
    begin0 set! and or for for/list for/vector for/hash for/fold for/sum
    for/and for/or for/first for/last for* for*/list match match-define
    match-let with-syntax syntax-parse syntax-case syntax-rules require
    provide struct define-struct parameterize with-handlers do delay module
    module+ module* ~> ~>> and~> and~>> lambda~> lambda~>> #%app #%datum
    #%top))

(define peer-heads '(cons list list* values vector)) ;; noqa

;; Body macros are named with-…, so a with- form's first argument is a
;; setting, not a subject to thread.
(define-syntax-class step-head
  (pattern head:id
    #:do [(define name (syntax-e #'head))]
    #:fail-when (or (memq name core-form-heads) (memq name peer-heads)) #f
    #:fail-when (regexp-match? #px"^(check-|test-|with-|define)"
                               (symbol->string name))
                #f
    #:when (identifier-binding #'head)))

(define-syntax-class commutative-operator
  (pattern head:id
    #:when (for/or ([operator (in-list (list #'+ #'* #'torch:+ #'torch:*))])
             (free-identifier=? #'head operator))))

;; A call whose other arguments compute something joins two chains, and
;; threading would hide one; a swapped operand must not run code.
(define-syntax-class atom
  #:literals (quote)
  (pattern :id)
  (pattern :number)
  (pattern :boolean)
  (pattern :str)
  (pattern :keyword)
  (pattern (quote _)))

(define-syntax-class call-chain
  #:attributes (root [step 1])
  (pattern (head:step-head inner:call-chain arg:atom ...)
    #:with root #'inner.root
    #:with (step ...) #'(inner.step ... (head arg ...)))
  (pattern (head:commutative-operator other:atom inner:call-chain)
    #:with root #'inner.root
    #:with (step ...) #'(inner.step ... (head other)))
  (pattern (head:step-head arg0 arg ...)
    #:with root this-syntax
    #:with (step ...) #'()))

(define (bare-when-unary step)
  (syntax-parse step
    [(head) #'head]
    [_ step]))

(define (threading-here? stx)
  (free-identifier=? (datum->syntax stx '~>) #'~>))

(define (syntax-parts stx)
  (let loop ([datum (syntax-e stx)])
    (cond
      [(pair? datum) (cons (car datum) (loop (cdr datum)))]
      [(syntax? datum) (loop (syntax-e datum))]
      [else '()])))

(define (path-within? path outer)
  (define elements (syntax-path-elements path))
  (define outer-elements (syntax-path-elements outer))
  (and (<= (treelist-length outer-elements) (treelist-length elements))
       (for/and ([outer-element (in-treelist outer-elements)]
                 [element (in-treelist elements)])
         (= outer-element element))))

;; A ~> or ~>> step receives the previous result from a sibling form, and a
;; keyword call binds it a few levels down, so every original form anywhere
;; inside the expansion counts, not just the call's direct arguments.
(define (written-argument-entries stx entries)
  (define-values (inner-entries inner-paths)
    (for/fold ([entries entries] [paths '()])
              ([part (in-list (syntax-parts stx))])
      (define-values (part-entries part-paths)
        (written-argument-entries part entries))
      (values part-entries (append part-paths paths))))
  (define here (syntax-original-path stx))
  (cond
    [here
     (values (cons (syntax-property-entry
                    (syntax-property stx 'expansion-path)
                    'arguments-written-here?
                    (for/and ([path (in-list inner-paths)])
                      (path-within? path here)))
                   inner-entries)
             (list here))]
    [else (values inner-entries inner-paths)]))

(define written-arguments-analyzer
  (make-expansion-analyzer
   #:name 'written-arguments-analyzer
   (lambda (expanded)
     (define-values (entries _outermost)
       (written-argument-entries (syntax-label-paths expanded 'expansion-path)
                                 '()))
     (sequence->syntax-property-bundle entries))))

;; A form the analyzer could not place carries no property, so the rule stays
;; quiet rather than rewrite a step it cannot see.
(define-refactoring-rule nested-calls-to-threading
  #:description
  "This chain of three or more calls reads better threaded with `~>`: each call takes the previous result as its first argument, or as either operand of a two-operand `+` or `*`, beside only names and literals."
  #:analyzers (list written-arguments-analyzer)
  chain:call-chain
  #:when (>= (length (attribute chain.step)) 2)
  #:when (syntax-property this-syntax 'arguments-written-here?)
  #:when (threading-here? this-syntax)
  #:with (step ...) (map bare-when-unary (attribute chain.step))
  (~> chain.root step ...))

(define-refactoring-rule threading-from-innermost-call
  #:description
  "This `~>` starts from a nested call; thread from the innermost call instead."
  #:literals (~>)
  (~> start:call-chain later ...)
  #:when (pair? (attribute start.step))
  #:with (step ...) (map bare-when-unary (attribute start.step))
  (~> start.root step ... later ...))

(define-refactoring-suite rktorch-rules
  #:rules (named-comparison-to-operator
           nested-calls-to-threading
           threading-from-innermost-call))

(define-refactoring-suite rktorch-recommendations
  #:suites (default-recommendations rktorch-rules))
