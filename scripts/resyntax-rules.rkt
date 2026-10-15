#lang racket/base

(require (only-in racket/list append* append-map filter-map)
         (only-in racket/match match-define)
         (only-in racket/treelist in-treelist treelist-length)
         (only-in resyntax/base
                  define-refactoring-rule define-refactoring-suite)
         (only-in resyntax/default-recommendations default-recommendations)
         (only-in resyntax/grimoire/syntax-path
                  child-syntax-path? syntax-label-paths syntax-path-elements
                  syntax-path-last-element syntax-path-parent)
         (only-in resyntax/grimoire/syntax-property-bundle
                  sequence->syntax-property-bundle syntax-property-entry)
         (only-in resyntax/private/analyzer make-expansion-analyzer)
         (only-in resyntax/private/syntax-neighbors syntax-original-path)
         (only-in syntax/parse/pre
                  attribute boolean define-syntax-class expr id kernel-literals
                  keyword number pattern str syntax-parse this-syntax ~and
                  ~optional ~or* ~var)
         (only-in threading ~> ~>>)
         (only-in torch
                  add device div dtype ge gt le lt matmul mul sub tensor-device
                  tensor-dtype)
         (prefix-in torch: (only-in torch * + - / < <= > >= @)))

(provide rktorch-rules rktorch-recommendations)

(define named-operators ;; noqa
  (list (list #'add #'torch:+)
        (list #'sub #'torch:-)
        (list #'mul #'torch:*)
        (list #'div #'torch:/)
        (list #'matmul #'torch:@)
        (list #'ge #'torch:>=)
        (list #'gt #'torch:>)
        (list #'le #'torch:<=)
        (list #'lt #'torch:<)))

(define long-getters ;; noqa
  (list (list #'tensor-device #'device)
        (list #'tensor-dtype #'dtype)))

(define (exported-name id)
  (match-define (list _ _ _ name _ ...) (identifier-binding id))
  name)

(define-syntax-class (renamed-in table)
  #:attributes (short)
  (pattern name:id
    #:do [(define renamed
            (for/first ([entry (in-list table)]
                        #:do [(match-define (list long short) entry)]
                        #:when (eq? (syntax-e #'name) (syntax-e long))
                        #:when (free-identifier=? #'name long)
                        #:do [(define here
                                (datum->syntax #'name (exported-name short) #'name))]
                        #:when (free-identifier=? here short))
              here))]
    #:when renamed
    #:with short renamed))

(define-refactoring-rule named-operator-to-symbol
  #:description
  "Use torch's operators: `+`, `-`, `*`, `/` and `@` dispatch to `add`, `sub`, `mul`, `div` and `matmul`, and `>=`, `>`, `<=` and `<` to `ge`, `gt`, `le` and `lt`. A `~>` step shows one operand and keeps it."
  ((~var op (renamed-in named-operators)) a:expr (~optional b:expr))
  (op.short a (~? b)))

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

(define (formal-ids formals)
  (syntax-parse formals
    [() '()]
    [rest:id (list #'rest)]
    [(id:id . rest) (cons #'id (formal-ids #'rest))]))

(define (form-non-references stx)
  (syntax-parse stx
    #:literal-sets (kernel-literals)
    [((~or* quote quote-syntax #%provide #%require #%declare define-syntaxes
            begin-for-syntax)
      . _)
     #f]
    [((~or* module module*) name language . _) (list #'name #'language)]
    [((~and head define-values) (id ...) _) (cons #'head (attribute id))]
    [((~and head (~or* let-values letrec-values)) ([(id ...) _] ...) . _)
     (cons #'head (append* (attribute id)))]
    [((~and head #%plain-lambda) formals . _) (cons #'head (formal-ids #'formals))]
    [((~and head case-lambda) [formals . _] ...)
     (cons #'head (append-map formal-ids (attribute formals)))]
    [((~and head set!) id _) (list #'head #'id)]
    [(head:id . _) (list #'head)]
    [_ '()]))

;; A call to a contracted export expands without its head, which survives
;; only in the expansion's 'origin.
(define (origin-ids stx)
  (let loop ([origin (syntax-property stx 'origin)])
    (cond
      [(identifier? origin) (list origin)]
      [(pair? origin) (append (loop (car origin)) (loop (cdr origin)))]
      [else '()])))

(struct written-node (expansion-path original-path arguments-here? references))

;; A ~> or ~>> step receives the previous result from a sibling form, and a
;; keyword call binds it a few levels down, so every original form anywhere
;; inside the expansion counts, not just the call's direct arguments.
(define (written-nodes stx nodes non-references)
  (define own-non-references (form-non-references stx))
  (define part-non-references
    (and non-references own-non-references
         (append own-non-references non-references)))
  (define-values (inner-nodes inner-paths)
    (for/fold ([nodes nodes] [paths '()])
              ([part (in-list (syntax-parts stx))])
      (define-values (part-nodes part-paths)
        (written-nodes part nodes part-non-references))
      (values part-nodes (append part-paths paths))))
  (define here (syntax-original-path stx))
  (define referenced-ids
    (cond
      [(not non-references) '()]
      [(and (identifier? stx) (not (memq stx non-references)))
       (cons stx (origin-ids stx))]
      [else (origin-ids stx)]))
  (define node
    (written-node (syntax-property stx 'expansion-path)
                  here
                  (and here
                       (for/and ([path (in-list inner-paths)])
                         (path-within? path here)))
                  (filter-map syntax-original-path referenced-ids)))
  (cond
    [here (values (cons node inner-nodes) (list here))]
    [(pair? (written-node-references node))
     (values (cons node inner-nodes) inner-paths)]
    [else (values inner-nodes inner-paths)]))

(define (written-node-entries nodes)
  (define references
    (for*/hash ([node (in-list nodes)]
                [path (in-list (written-node-references node))])
      (values path #t)))
  (define reference-parts
    (for/fold ([parts (hash)])
              ([path (in-hash-keys references)]
               #:when (child-syntax-path? path))
      (hash-update parts
                   (syntax-path-parent path)
                   (lambda (indices) (cons (syntax-path-last-element path) indices))
                   '())))
  (for*/list ([node (in-list nodes)]
              #:do [(match-define (written-node expansion-path here arguments-here? _)
                      node)]
              #:when here
              [(key value)
               (in-hash (hash 'arguments-written-here? arguments-here?
                              'variable-reference? (hash-ref references here #f)
                              'reference-parts (hash-ref reference-parts here #f)))]
              #:when value)
    (syntax-property-entry expansion-path key value)))

(define written-forms-analyzer
  (make-expansion-analyzer
   #:name 'written-forms-analyzer
   (lambda (expanded)
     (define-values (nodes _outermost)
       (written-nodes (syntax-label-paths expanded 'expansion-path) '() '()))
     (sequence->syntax-property-bundle (written-node-entries nodes)))))

(define (short-getter part)
  (syntax-parse part
    [(~var getter (renamed-in long-getters))
     (syntax-property #'getter.short 'replacement-for part)]
    [_ part]))

(define-syntax-class getter-reference
  #:attributes (replacement)
  (pattern (~var getter (renamed-in long-getters))
    #:when (syntax-property this-syntax 'variable-reference?)
    #:attr replacement (short-getter this-syntax))
  (pattern (part ...)
    #:do [(define indices (or (syntax-property this-syntax 'reference-parts) '()))
          (define parts
            (for/list ([part (in-list (attribute part))]
                       [index (in-naturals)])
              (cond
                [(memv index indices) (short-getter part)]
                [else part])))]
    #:when (not (andmap eq? parts (attribute part)))
    #:attr replacement (datum->syntax this-syntax parts this-syntax)))

(define-refactoring-rule tensor-getter-to-short-name
  #:description
  "Use torch's short getters: `device` and `dtype` answer what `tensor-device` and `tensor-dtype` do."
  #:analyzers (list written-forms-analyzer)
  reference:getter-reference
  reference.replacement)

;; A form the analyzer could not place carries no property, so the rule stays
;; quiet rather than rewrite a step it cannot see.
(define-refactoring-rule nested-calls-to-threading
  #:description
  "This chain of three or more calls reads better threaded with `~>`: each call takes the previous result as its first argument, or as either operand of a two-operand `+` or `*`, beside only names and literals."
  #:analyzers (list written-forms-analyzer)
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
  #:rules (named-operator-to-symbol
           nested-calls-to-threading
           tensor-getter-to-short-name
           threading-from-innermost-call))

(define-refactoring-suite rktorch-recommendations
  #:suites (default-recommendations rktorch-rules))
