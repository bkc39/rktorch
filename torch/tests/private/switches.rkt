#lang racket/base

(require (for-syntax racket/base syntax/parse/pre)
         (only-in racket/dict in-dict)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in rackunit test-case)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../main.rkt"
                  * - @ ~> adaptive-avg-pool2d backward! conv2d
                  cuda-matmul-allow-tf32? cudnn-allow-tf32? cudnn-benchmark?
                  cudnn-deterministic? deterministic-algorithms-enabled?
                  deterministic-algorithms-warn-only? embedding exn:fail:rktorch:oom?
                  float32-matmul-precision full grid-sample index-add! index-select
                  item log-softmax max min ones rand randn
                  scaled-dot-product-attention scatter-add!
                  set-cuda-matmul-allow-tf32! set-cudnn-allow-tf32!
                  set-cudnn-benchmark! set-cudnn-deterministic!
                  set-float32-matmul-precision! sum tensor upsample-bicubic2d
                  use-deterministic-algorithms! zeros)
         (only-in "../../nn.rkt" ctc-loss nll-loss)
         (only-in "host.rkt" host)
         (only-in "within.rkt" within))

(provide bounded-switches-test
         conv-witness
         default-switches
         determinism-rows
         from-defaults
         matmul-witness
         other-switches
         precision-refusal?
         precision-sequences
         precisions
         sequence-outcome
         set-switches!
         switch-values
         tf32-settings
         witness-values
         within/switches-restored)

(define precisions '(highest high medium))

(struct switch (key read write! default))

(define switches
  (list (switch 'deterministic-algorithms deterministic-algorithms-enabled?
                (lambda (on?)
                  (use-deterministic-algorithms!
                   on? #:warn-only? (deterministic-algorithms-warn-only?)))
                #f)
        (switch 'warn-only deterministic-algorithms-warn-only?
                (lambda (warn-only?)
                  (use-deterministic-algorithms!
                   (deterministic-algorithms-enabled?) #:warn-only? warn-only?))
                #f)
        (switch 'cudnn-deterministic cudnn-deterministic? set-cudnn-deterministic! #f)
        (switch 'cudnn-benchmark cudnn-benchmark? set-cudnn-benchmark! #f)
        (switch 'float32-matmul-precision float32-matmul-precision
                set-float32-matmul-precision! 'highest)
        (switch 'cuda-matmul-allow-tf32 cuda-matmul-allow-tf32? #f #f)
        (switch 'cudnn-allow-tf32 cudnn-allow-tf32? set-cudnn-allow-tf32! #t)))

(define (switch-values)
  (for/hasheq ([s (in-list switches)])
    (values (switch-key s) ((switch-read s)))))

(define default-switches
  (for/hasheq ([s (in-list switches)])
    (values (switch-key s) (switch-default s))))

(define other-switches
  (hasheq 'deterministic-algorithms #t
          'warn-only #t
          'cudnn-deterministic #t
          'cudnn-benchmark #t
          'float32-matmul-precision 'high
          'cuda-matmul-allow-tf32 #t
          'cudnn-allow-tf32 #f))

(define (set-switches! flags)
  (for ([s (in-list switches)]
        #:when (switch-write! s))
    ((switch-write! s) (hash-ref flags (switch-key s)))))

(define precision-sequences
  '((high (precision high))
    (medium (precision medium))
    (highest (precision highest))
    (matmul-on (matmul-tf32 #t))
    (matmul-on-off (matmul-tf32 #t) (matmul-tf32 #f))
    (high-matmul-off (precision high) (matmul-tf32 #f))
    (medium-matmul-on (precision medium) (matmul-tf32 #t))
    (medium-matmul-off (precision medium) (matmul-tf32 #f))
    (high-matmul-on (precision high) (matmul-tf32 #t))
    (matmul-on-medium (matmul-tf32 #t) (precision medium))
    (cudnn-off (cudnn-tf32 #f))
    (cudnn-off-high (cudnn-tf32 #f) (precision high))
    (high-cudnn-off (precision high) (cudnn-tf32 #f))
    (high-matmul-off-highest (precision high) (matmul-tf32 #f) (precision highest))))

(define refusal-rx
  (pregexp
   (string-append
    "^float32-matmul-precision: .*tr_get_float32_matmul_precision: "
    "PyTorch is checking the matmul precision")))

(define (precision-refusal? e)
  (and (exn:fail? e)
       (not (exn:fail:contract? e))
       (not (exn:fail:rktorch:oom? e))
       (regexp-match? refusal-rx (exn-message e))))

(define (apply-step! step)
  (match step
    [`(precision ,p) (set-float32-matmul-precision! p)]
    [`(matmul-tf32 ,on?) (set-cuda-matmul-allow-tf32! on?)]
    [`(cudnn-tf32 ,on?) (set-cudnn-allow-tf32! on?)]))

(define (outcome-of read)
  (with-handlers ([precision-refusal? (lambda (_) 'raise)])
    (read)))

(define (sequence-outcome steps)
  (set-switches! default-switches)
  (for-each apply-step! steps)
  (map outcome-of
       (list float32-matmul-precision cuda-matmul-allow-tf32? cudnn-allow-tf32?)))

(define (within/switches-restored seconds thunk)
  (define saved (switch-values))
  (define started (make-custodian))
  (dynamic-wind void
                (lambda ()
                  (parameterize ([current-custodian started]
                                 [current-subprocess-custodian-mode 'kill])
                    (within seconds thunk)))
                (lambda ()
                  (custodian-shutdown-all started)
                  (set-switches! saved))))

(define-syntax-parse-rule (bounded-switches-test name:expr
                            (~optional (~seq #:seconds seconds:expr)
                                       #:defaults ([seconds #'60]))
                            body:expr ...+)
  (test-case name (within/switches-restored seconds (lambda () body ...))))

(define (bins on)
  (tensor '(0 3 1 3 7 2 3 0 5 3 6 1 4 3 2 7) #:dtype 'int64 #:device on))

(define (leaf on . dims)
  (apply randn dims #:device on #:requires-grad? #t))

(define determinism-rows
  (list
   (cons 'adaptive-avg-pool2d-backward
         (lambda (on)
           (~> (leaf on 1 2 5 5) (adaptive-avg-pool2d '(3 3)) sum backward!)))
   (cons 'ctc-loss-backward
         (lambda (on)
           (~> (leaf on 10 2 5)
               (log-softmax 2)
               (ctc-loss (tensor '((1 2 3 4) (2 3 4 1)) #:dtype 'int64
                                 #:device on)
                         #:input-lengths '(10 10)
                         #:target-lengths '(4 4))
               backward!)))
   (cons 'grid-sample-backward
         (lambda (on)
           (~> (leaf on 1 1 4 4)
               (grid-sample (~> (rand 1 3 3 2 #:device on) (* 1.8) (- 0.9)))
               sum
               backward!)))
   (cons 'upsample-bicubic2d-backward
         (lambda (on)
           (~> (leaf on 1 1 4 4) (upsample-bicubic2d '(8 8)) sum backward!)))
   (cons 'index-add
         (lambda (on)
           (index-add! (zeros 8 #:device on) 0 (bins on) (randn 16 #:device on))))
   (cons 'scatter-add
         (lambda (on)
           (scatter-add! (zeros 8 #:device on) 0 (bins on)
                         (randn 16 #:device on))))
   (cons 'index-select-backward
         (lambda (on)
           (~> (leaf on 8 4) (index-select 0 (bins on)) sum backward!)))
   (cons 'embedding-backward
         (lambda (on)
           (~> (embedding (bins on) (leaf on 8 4)) sum backward!)))
   (cons 'conv2d-backward
         (lambda (on)
           (~> (randn 1 1 8 8 #:device on) (conv2d (leaf on 2 1 3 3)) sum
               backward!)))
   (cons 'nll-loss-forward
         (lambda (on)
           (~> (randn 4 5 #:device on)
               (log-softmax 1)
               (nll-loss (tensor '(0 1 2 3) #:dtype 'int64 #:device on)))))
   (cons 'scaled-dot-product-attention-backward
         (lambda (on)
           (~> (scaled-dot-product-attention (leaf on 1 2 8 16)
                                             (leaf on 1 2 8 16)
                                             (leaf on 1 2 8 16))
               sum
               backward!)))))

(define tf32-x (+ 1.0 (expt 2.0 -12)))

(define (matmul-witness on)
  (@ (full tf32-x 1024 1024 #:device on) (ones 1024 1024 #:device on)))

(define (conv-witness on)
  (conv2d (full tf32-x 32 64 56 56 #:device on) (ones 128 64 3 3 #:device on)))

(define (distinct-values t)
  (define lo (item (min t)))
  (cond
    [(= lo (item (max t))) (list lo)]
    [else (~> t host remove-duplicates (sort <))]))

(define (precision-setting p)
  (lambda () (set-float32-matmul-precision! p)))

(define tf32-settings
  (list (cons 'high (precision-setting 'high))
        (cons 'medium (precision-setting 'medium))
        (cons 'matmul-tf32 (lambda () (set-cuda-matmul-allow-tf32! #t)))))

(define matmul-settings
  (cons (cons 'highest (precision-setting 'highest)) tf32-settings))

(define conv-settings
  (list (cons 'cudnn-on (lambda () (set-cudnn-allow-tf32! #t)))
        (cons 'cudnn-off (lambda () (set-cudnn-allow-tf32! #f)))))

(define (from-defaults setting! witness)
  (set-switches! default-switches)
  (setting!)
  (distinct-values (witness 'cuda)))

(define (witness-under settings witness)
  (for/hasheq ([(name setting!) (in-dict settings)])
    (values name (from-defaults setting! witness))))

(define (witness-values)
  (hasheq 'matmul (witness-under matmul-settings matmul-witness)
          'conv (witness-under conv-settings conv-witness)))
