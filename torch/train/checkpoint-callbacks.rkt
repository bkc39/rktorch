#lang racket/base

(require (only-in racket/contract/base
                  -> ->* ->i and/c any/c contract-out hash/c not/c or/c)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "../private/not-implemented.rkt" not-implemented)
         (only-in "private/types.rkt"
                  callback-property callback? define-named-contract prop:callback))

(struct model-checkpoint (settings) ;; noqa
  #:constructor-name make-model-checkpoint ;; noqa
  #:omit-define-syntaxes
  #:property prop:callback
  (callback-property 'checkpoint 'model-checkpoint
                     #:hook-ref (lambda (_cb _name) #f)
                     #:state-key (lambda (_cb) (not-implemented 'model-checkpoint 'T3))))

(provide (contract-out [model-checkpoint? (-> any/c boolean?)]))

(define-named-contract save-top-k/c (or/c -1 exact-nonnegative-integer?))

(define (checkpoint-filename? v)
  (and (string? v)
       (regexp-match? #px"^(?:[^{}]|[{][^{}:]+(?::(?:[.][0-9]+f|0[0-9]+d))?[}])+$" v)))

(define-named-contract checkpoint-filename/c checkpoint-filename?)

(define (triggered? n)
  (define given (supplied-or n))
  (and given (>= given 1)))

(define/contract-out (model-checkpoint #:dirpath dirpath ;; noqa
                                       #:filename [filename #f]
                                       #:monitor [monitor #f]
                                       #:verbose? [verbose? #f]
                                       #:save-last [save-last #f]
                                       #:save-top-k [save-top-k 1]
                                       #:save-weights-only? [weights-only? #f]
                                       #:mode [mode 'min]
                                       #:auto-insert-metric-name? [auto-insert? #t]
                                       #:every-n-train-steps [every-n-train-steps #f]
                                       #:every-n-epochs [every-n-epochs #f]
                                       #:save-on-train-epoch-end [on-epoch-end 'auto]
                                       #:enable-version-counter? [version-counter? #t])
  (->i (#:dirpath [dirpath path-string?])
       (#:filename [filename (or/c #f checkpoint-filename/c)]
        #:monitor [monitor (or/c #f symbol?)]
        #:verbose? [verbose? boolean?]
        #:save-last [save-last boolean?]
        #:save-top-k [save-top-k save-top-k/c]
        #:save-weights-only? [weights-only? boolean?]
        #:mode [mode (or/c 'min 'max)]
        #:auto-insert-metric-name? [auto-insert? boolean?]
        #:every-n-train-steps [every-n-train-steps (or/c #f exact-nonnegative-integer?)]
        #:every-n-epochs [every-n-epochs (or/c #f exact-nonnegative-integer?)]
        #:save-on-train-epoch-end [on-epoch-end (or/c 'auto boolean?)]
        #:enable-version-counter? [version-counter? boolean?])
       #:pre/name (every-n-train-steps every-n-epochs)
       "#:every-n-train-steps and #:every-n-epochs are mutually exclusive"
       (not (and (triggered? every-n-train-steps) (triggered? every-n-epochs)))
       #:pre/name (monitor save-top-k)
       "without #:monitor, #:save-top-k must be -1, 0 or 1"
       (or (supplied-or monitor) (memv (supplied-or save-top-k 1) '(-1 0 1)))
       [_ model-checkpoint?])
  (not-implemented 'model-checkpoint 'T3))

(define/contract-out (model-checkpoint-best-model-path cb) ;; noqa
  (-> model-checkpoint? (or/c #f string?))
  (not-implemented 'model-checkpoint-best-model-path 'T3))

(define/contract-out (model-checkpoint-best-model-score cb) ;; noqa
  (-> model-checkpoint? (or/c #f flonum?))
  (not-implemented 'model-checkpoint-best-model-score 'T3))

(define/contract-out (model-checkpoint-best-k-models cb) ;; noqa
  (-> model-checkpoint? (hash/c string? flonum? #:immutable #t))
  (not-implemented 'model-checkpoint-best-k-models 'T3))

(define/contract-out (model-checkpoint-last-model-path cb) ;; noqa
  (-> model-checkpoint? (or/c #f string?))
  (not-implemented 'model-checkpoint-last-model-path 'T3))

(define/contract-out (on-exception-checkpoint #:dirpath dirpath ;; noqa
                                              #:filename [filename "on_exception"])
  (->* (#:dirpath path-string?) (#:filename (and/c string? (not/c ""))) callback?)
  (not-implemented 'on-exception-checkpoint 'T3))
