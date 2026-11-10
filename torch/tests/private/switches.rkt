#lang racket/base

(require (for-syntax racket/base syntax/parse/pre)
         (only-in rackunit test-case)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../../main.rkt"
                  * - ~> adaptive-avg-pool2d backward! conv2d embedding
                  cudnn-benchmark? cudnn-deterministic?
                  deterministic-algorithms-enabled?
                  deterministic-algorithms-warn-only? grid-sample index-add!
                  index-select log-softmax rand randn scaled-dot-product-attention
                  scatter-add! set-cudnn-benchmark! set-cudnn-deterministic! sum
                  tensor upsample-bicubic2d use-deterministic-algorithms! zeros)
         (only-in "../../nn.rkt" ctc-loss nll-loss)
         (only-in "within.rkt" within))

(provide bounded-switches-test
         default-switches
         determinism-rows
         set-switches!
         switch-values
         within/switches-restored)

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
        (switch 'cudnn-benchmark cudnn-benchmark? set-cudnn-benchmark! #f)))

(define (switch-values)
  (for/hasheq ([s (in-list switches)])
    (values (switch-key s) ((switch-read s)))))

(define default-switches
  (for/hasheq ([s (in-list switches)])
    (values (switch-key s) (switch-default s))))

(define (set-switches! flags)
  (for ([s (in-list switches)])
    ((switch-write! s) (hash-ref flags (switch-key s)))))

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
