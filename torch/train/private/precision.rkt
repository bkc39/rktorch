#lang racket/base

(require (only-in racket/match match match* struct*)
         (only-in "../../foreign.rkt"
                  backward! call-with-autocast cuda-bfloat16-supported? device-index
                  device-type)
         (only-in "../../nn/clip.rkt" clip-grad-norm! clip-grad-value!)
         (only-in "../../nn/grad-scaler.rkt"
                  grad-scaler grad-scaler-scale-loss grad-scaler-step! grad-scaler-unscale!
                  grad-scaler-update!)
         (only-in (submod "../../nn/grad-scaler.rkt" private)
                  grad-scaler-armed? grad-scaler-checked? grad-scaler-forget!)
         (only-in "../../nn/optim.rkt" optimizer-parameters step!)
         (only-in (submod "../../nn/optim.rkt" private) optimizer-fused?)
         (only-in "dispatch.rkt" call-hooks)
         (only-in "state.rkt"
                  run-device run-global-step run-grad-scaler run-optimizers run-precision
                  run-skip-window? set-run-global-step! set-run-grad-scaler!
                  set-run-precision! set-run-stepped?!))

(provide backward-pass!
         call-in-precision
         check-clippable!
         (struct-out clipping)
         configure-precision!
         count-step!
         manual-optimizer-step!
         window-step!)

(define-logger rktorch-train)

(define cpu-fp16-message
  (string-append "you passed #:precision '16-mixed on the CPU, but AMP with fp16 is not"
                 " supported on the CPU; using #:precision 'bf16-mixed instead"))

(define (warn-cpu-fp16!)
  (log-rktorch-train-warning cpu-fp16-message)
  (eprintf "rktorch-train: ~a\n" cpu-fp16-message))

(define (resolved-precision requested dev)
  (match* (requested (device-type dev))
    [('auto-mixed 'cuda)
     (if (cuda-bfloat16-supported? (device-index dev)) 'bf16-mixed '16-mixed)]
    [('auto-mixed _) 'bf16-mixed]
    [('16-mixed 'cpu) (warn-cpu-fp16!) 'bf16-mixed]
    [(precision _) precision]))

(define (configure-precision! r requested)
  (define precision (resolved-precision requested (run-device r)))
  (set-run-precision! r precision)
  (set-run-grad-scaler! r (and (eq? precision '16-mixed) (grad-scaler #:device (run-device r)))))

(define (call-in-precision r thunk)
  (match (run-precision r)
    ['32-true (thunk)]
    [precision
     (call-with-autocast thunk #:device (run-device r) #:cache-enabled? #f
                         #:dtype (if (eq? precision 'bf16-mixed) 'bfloat16 'float16))]))

(define fused-clip-message
  (string-append "the current optimizer does not allow for gradient clipping because it"
                 " performs unscaling of gradients internally; HINT: are you using a"
                 " 'fused' optimizer?"))

(define (check-clippable! r clip)
  (match (and clip (run-grad-scaler r) (findf optimizer-fused? (run-optimizers r)))
    [#f (void)]
    [opt (raise-arguments-error 'fit fused-clip-message "optimizer" opt)]))

(define (armed-scaler r)
  (define s (run-grad-scaler r))
  (and s (grad-scaler-armed? s) s))

(define (backward-pass! r loss)
  (define s (run-grad-scaler r))
  (define scaled (if s (grad-scaler-scale-loss s loss) loss))
  (call-hooks r 'on-before-backward scaled)
  (backward! scaled)
  (call-hooks r 'on-after-backward)
  (void))

(struct clipping (value algorithm))

(define (clip! clip opt)
  (match clip
    [#f (void)]
    [(struct* clipping ([value c] [algorithm algorithm]))
     (define clip-grads! (if (eq? algorithm 'norm) clip-grad-norm! clip-grad-value!))
     (clip-grads! (optimizer-parameters opt) c)]))

(define (count-step! r)
  (set-run-global-step! r (add1 (run-global-step r)))
  (set-run-stepped?! r #t))

(define (settle-scale! s opts)
  (cond
    [(not s) (void)]
    [(grad-scaler-checked? s) (grad-scaler-update! s)]
    [else (for ([opt (in-list opts)]) (grad-scaler-forget! s opt))]))

(define (step-optimizers! r opts clip #:backward? [backward? #t])
  (define s (and backward? (armed-scaler r)))
  (for ([opt (in-list opts)]
        #:when (and s (not (optimizer-fused? opt))))
    (grad-scaler-unscale! s opt))
  (for ([opt (in-list opts)])
    (call-hooks r 'on-before-optimizer-step opt))
  (cond
    [(run-skip-window? r)
     (for ([opt (in-list opts)]
           #:when s)
       (grad-scaler-forget! s opt))]
    [else
     (define step? (or backward? (not (run-grad-scaler r))))
     (for ([opt (in-list opts)])
       (clip! clip opt)
       (when step?
         (if s (grad-scaler-step! s opt) (step! opt))))
     (settle-scale! s opts)]))

(define (window-step! r clip #:backward? backward?)
  (step-optimizers! r (run-optimizers r) clip #:backward? backward?))

(define (manual-optimizer-step! r opt)
  (step-optimizers! r (list opt) #f)
  (count-step! r))
