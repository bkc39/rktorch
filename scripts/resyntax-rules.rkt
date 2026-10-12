#lang racket/base

(require (only-in resyntax/base
                  define-refactoring-rule define-refactoring-suite)
         (only-in resyntax/default-recommendations default-recommendations)
         (only-in syntax/parse/pre define-syntax-class expr pattern ~and)
         (only-in torch ge gt le lt)
         (prefix-in torch: (only-in torch < <= > >=)))

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

(define-refactoring-suite rktorch-rules
  #:rules (named-comparison-to-operator))

(define-refactoring-suite rktorch-recommendations
  #:suites (default-recommendations rktorch-rules))
