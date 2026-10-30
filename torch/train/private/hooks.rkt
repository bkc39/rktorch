#lang racket/base

(require (only-in racket/list argmin filter-not last)
         (only-in racket/match match)
         (only-in racket/string string-split))

(provide (struct-out hook)
         (struct-out log-rule)
         callback-hook?
         closest-hook-name
         definable-hooks
         epoch-log
         epoch-only-log
         hook-arity-problem
         hook-for-side
         hook-named
         hooks
         loop-hook-names
         step-log
         trainer-hook?)

(struct log-rule (on-step on-epoch default-on-step? default-on-epoch?)
  #:transparent)

(define step-log (log-rule '(#f #t) '(#f #t) #t #f))
(define epoch-log (log-rule '(#f #t) '(#f #t) #f #t))
(define epoch-only-log (log-rule '(#f) '(#t) #f #t))

(struct hook (name sides formals batch? indexed? log form order) #:transparent)

(define (row name sides formals
             #:batch? [batch? #f] #:indexed? [indexed? batch?] #:log [log #f]
             #:form [form #f] #:order [order 'callbacks-first])
  (hook name sides formals batch? indexed? log form order))

(define both '(callback trainer))

(define (stage-rows stage step-log #:edge-log [edge-log epoch-only-log]
                    #:end-log [end-log #f])
  (define (named fmt) (string->symbol (format fmt stage)))
  (list (row (named "on-~a-start") both '() #:log edge-log)
        (row (named "on-~a-epoch-start") both '() #:log edge-log)
        (row (named "on-~a-batch-start") both '() #:batch? #t #:log step-log)
        (row (named "~a-step") '(trainer) '() #:batch? #t #:log step-log
             #:form (named "define-~a-step"))
        (row (named "on-~a-batch-end") both '(out) #:batch? #t #:log step-log)
        (row (named "on-~a-epoch-end") both '() #:log edge-log)
        (row (named "on-~a-end") both '() #:log end-log)))

(define hooks
  (append
   (list
    (row 'configure-callbacks '(trainer) '() #:form 'define-callbacks)
    (row 'setup both '(stage))
    (row 'configure-optimizers '(trainer) '() #:form 'define-optimizers)
    (row 'on-fit-start both '())
    (row 'on-sanity-check-start '(callback) '())
    (row 'on-sanity-check-end '(callback) '())
    (row 'on-train-start both '() #:log epoch-only-log)
    (row 'on-train-epoch-start both '() #:log epoch-only-log)
    (row 'on-before-batch-transfer '(trainer) '() #:batch? #t #:indexed? #f)
    (row 'on-after-batch-transfer '(trainer) '() #:batch? #t #:indexed? #f)
    (row 'on-train-batch-start both '() #:batch? #t #:log step-log)
    (row 'training-step '(trainer) '() #:batch? #t #:log step-log
         #:form 'define-training-step)
    (row 'on-before-zero-grad both '(optimizer) #:log step-log)
    (row 'on-before-backward both '(loss) #:log step-log)
    (row 'on-after-backward both '() #:log step-log)
    (row 'on-before-optimizer-step both '(optimizer) #:log step-log)
    (row 'on-train-batch-end both '(loss extra) #:batch? #t #:log step-log)
    (row 'on-train-epoch-end both '() #:log epoch-only-log #:order 'by-kind)
    (row 'on-train-end both '())
    (row 'on-fit-end both '())
    (row 'teardown both '(stage)))
   (stage-rows 'validation epoch-log)
   (stage-rows 'test epoch-log)
   (stage-rows 'predict #f #:edge-log #f)
   (list
    (row 'on-exception '(callback) '(exn))
    (row 'state-dict '(callback) '())
    (row 'load-state! '(callback) '(state))
    (row 'on-save-checkpoint both '(checkpoint))
    (row 'on-load-checkpoint both '(checkpoint) #:order 'trainer-first))))

(define definable-hooks (filter-not hook-form hooks))

(define loop-hooks
  (for/hasheq ([part (in-list '(validation test predict))])
    (values part (map hook-name (stage-rows part #f)))))

(define (loop-hook-names part) (hash-ref loop-hooks part))

(define ((named? name) h) (eq? name (hook-name h)))

(define (hook-named name) (findf (named? name) hooks))

(define (definable-hook-named name) (findf (named? name) definable-hooks))

(define (callback-hook? h) (and (memq 'callback (hook-sides h)) #t))
(define (trainer-hook? h) (and (memq 'trainer (hook-sides h)) #t))

(define (other-side side)
  (match side
    ['callback 'trainer]
    ['trainer 'callback]))

(define (hook-for-side name side)
  (define h (definable-hook-named name))
  (cond
    [(not h)
     (values #f (format "unknown hook ~a; closest: ~a" name (closest-hook-name name)))]
    [(memq side (hook-sides h)) (values h #f)]
    [else
     (values h (format "~a is a ~a hook; define-~a cannot define it"
                       name (other-side side) side))]))

(define (arguments formals)
  (match formals
    ['() "no arguments"]
    [(list _) (format "1 argument ~a" formals)]
    [_ (format "~a arguments ~a" (length formals) formals)]))

(define (hook-arity-problem h count rest?)
  (define expected (hook-formals h))
  (define n (length expected))
  (cond
    [(and (hook-batch? h) (< count n))
     (format "~a takes ~a then the batch values" (hook-name h) (arguments expected))]
    [(hook-batch? h) #f]
    [(or rest? (not (= count n)))
     (format "~a takes ~a" (hook-name h) (arguments expected))]
    [else #f]))

(define (edit-distance as bs substitution)
  (define first-row (for/list ([i (in-range (add1 (length bs)))]) i))
  (for/fold ([prev first-row] #:result (last prev))
            ([a (in-list as)] [i (in-naturals 1)])
    (next-row prev a i bs substitution)))

(define (next-row prev a i bs substitution)
  (match prev
    [(cons _ ups)
     (for/fold ([acc (list i)] #:result (reverse acc))
               ([b (in-list bs)] [diag (in-list prev)] [up (in-list ups)])
       (match acc
         [(cons left _)
          (cons (min (add1 up) (add1 left) (+ diag (substitution a b))) acc)]))]))

(define (word-substitution a b)
  (if (string=? a b)
      0
      (/ (* 2 (edit-distance (string->list a) (string->list b)
                             (lambda (x y) (if (char=? x y) 0 1))))
         (max (string-length a) (string-length b)))))

(define (closest-hook-name name)
  (define (words s) (string-split (symbol->string s) "-"))
  (define target (words name))
  (hook-name
   (argmin (lambda (h) (edit-distance target (words (hook-name h))
                                      word-substitution))
           definable-hooks)))
