#lang racket/base

(require (only-in racket/list argmin)
         (only-in racket/string string-split))

(provide (struct-out hook)
         callback-hook?
         closest-hook-name
         definable-hook-named
         definable-hooks
         hook-named
         hooks
         trainer-hook?)

(struct hook (name sides formals batch? log form order) #:transparent)

(define (row name sides formals
             #:batch? [batch? #f] #:log [log #f] #:form [form #f]
             #:order [order 'callbacks-first])
  (hook name sides formals batch? log form order))

(define both '(callback trainer))

(define (stage-rows stage step-log #:edge-log [edge-log 'epoch-only]
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
    (row 'on-train-start both '() #:log 'epoch-only)
    (row 'on-train-epoch-start both '() #:log 'epoch-only)
    (row 'on-before-batch-transfer '(trainer) '() #:batch? #t)
    (row 'on-after-batch-transfer '(trainer) '() #:batch? #t)
    (row 'on-train-batch-start both '() #:batch? #t #:log 'step)
    (row 'training-step '(trainer) '() #:batch? #t #:log 'step
         #:form 'define-training-step)
    (row 'on-before-zero-grad both '(optimizer) #:log 'step)
    (row 'on-before-backward both '(loss) #:log 'step)
    (row 'on-after-backward both '() #:log 'step)
    (row 'on-before-optimizer-step both '(optimizer) #:log 'step)
    (row 'on-train-batch-end both '(loss extra) #:batch? #t #:log 'step)
    (row 'on-train-epoch-end both '() #:log 'epoch-only #:order 'by-kind)
    (row 'on-train-end both '())
    (row 'on-fit-end both '())
    (row 'teardown both '(stage)))
   (stage-rows 'validation 'epoch)
   (stage-rows 'test 'epoch)
   (stage-rows 'predict #f #:edge-log #f)
   (list
    (row 'on-exception '(callback) '(exn))
    (row 'state-dict '(callback) '())
    (row 'load-state! '(callback) '(state))
    (row 'on-save-checkpoint both '(checkpoint))
    (row 'on-load-checkpoint both '(checkpoint) #:order 'trainer-first))))

(define definable-hooks (filter (lambda (h) (not (hook-form h))) hooks))

(define (hook-named name)
  (for/first ([h (in-list hooks)] #:when (eq? name (hook-name h))) h))

(define (definable-hook-named name)
  (for/first ([h (in-list definable-hooks)] #:when (eq? name (hook-name h)))
    h))

(define (callback-hook? h) (and (memq 'callback (hook-sides h)) #t))
(define (trainer-hook? h) (and (memq 'trainer (hook-sides h)) #t))

(define (edit-distance as bs substitution)
  (define row0 (for/list ([i (in-range (add1 (length bs)))]) i))
  (for/fold ([prev row0] #:result (car (reverse prev)))
            ([a (in-list as)] [i (in-naturals 1)])
    (for/fold ([acc (list i)] #:result (reverse acc))
              ([b (in-list bs)] [diag (in-list prev)] [up (in-list (cdr prev))])
      (cons (min (add1 up) (add1 (car acc)) (+ diag (substitution a b)))
            acc))))

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

(module+ test
  (require (only-in racket/list filter-map)
           rackunit)
  (check-equal? (closest-hook-name 'on-batch-end) 'on-train-batch-end)
  (check-equal? (closest-hook-name 'on-fit-strat) 'on-fit-start)
  (check-equal? (filter-map hook-form hooks)
                '(define-callbacks define-optimizers define-training-step
                  define-validation-step define-test-step
                  define-predict-step)))
