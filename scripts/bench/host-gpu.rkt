#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in "harness.rkt" rotate summarize)
         (only-in "workloads/phases.rkt"
                  phased-example phased-phases phased-settings phased-sync! run-phases!))

(provide host-gpu-modes
         timed-steps
         host-gpu-records)

(define host-gpu-modes '(item no-item phase-sync))

(define (clock-ms) (current-inexact-monotonic-milliseconds))

(define (timed-steps p mode steps)
  (define sync! (phased-sync! p))
  (define totals (make-hasheq))
  (define (timed name thunk)
    (define t0 (clock-ms))
    (thunk)
    (when (eq? mode 'phase-sync) (sync!))
    (hash-update! totals name (lambda (ms) (+ ms (- (clock-ms) t0))) 0.0))
  (sync!)
  (define t0 (clock-ms))
  (for ([_ (in-range steps)])
    (run-phases! p #:around timed #:skip (if (eq? mode 'no-item) '(loss-item) '())))
  (sync!)
  (define step-ms (/ (- (clock-ms) t0) steps))
  (values step-ms
          (for/list ([ph (in-list (phased-phases p))] #:when (hash-has-key? totals (car ph)))
            (cons (car ph) (/ (hash-ref totals (car ph)) steps)))))

(define (host-gpu-records p #:steps [steps 20] #:rounds [rounds 5] #:warmup [warmup 5]
                          #:modes [modes host-gpu-modes])
  (for ([_ (in-range warmup)])
    (run-phases! p))
  (define rows
    (for*/list ([i (in-range rounds)] [mode (in-list (rotate modes i))])
      (define-values (step-ms phases) (timed-steps p mode steps))
      (list mode step-ms phases)))
  (define (record group variant samples)
    (hasheq 'schema 1 'suite "host-gpu" 'group group 'impl "rktorch"
            'case (phased-example p) 'variant variant 'unit "ms" 'reps steps
            'settings (phased-settings p) 'samples samples 'stats (summarize samples)))
  (define (of mode) (filter (lambda (row) (eq? (car row) mode)) rows))
  (for*/list ([mode (in-list modes)]
              [mode-rows (in-value (of mode))]
              [name (in-list (cons #f (remove-duplicates
                                        (for*/list ([row (in-list mode-rows)]
                                                    [ph (in-list (caddr row))])
                                          (car ph)))))])
    (if name
        (record "phase" (format "~a/~a" mode name)
                (for/list ([row (in-list mode-rows)]) (cdr (assq name (caddr row)))))
        (record "step" (symbol->string mode) (map cadr mode-rows)))))

(module+ test
  (require rackunit
           (only-in "workloads/phases.rkt" resnet-phases))

  (test-case "each mode times its phases and its step, and only item mode reads the loss"
    (define p (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4))
    (define records (host-gpu-records p #:steps 2 #:rounds 2 #:warmup 1))
    (define (variants group)
      (for/list ([r (in-list records)] #:when (equal? (hash-ref r 'group) group))
        (hash-ref r 'variant)))
    (check-equal? (variants "step") '("item" "no-item" "phase-sync"))
    (check-not-false (member "item/loss-item" (variants "phase")))
    (check-not-false (member "phase-sync/loss-item" (variants "phase")))
    (check-false (member "no-item/loss-item" (variants "phase")))
    (for ([r (in-list records)])
      (check-equal? (length (hash-ref r 'samples)) 2)
      (check-true (andmap positive? (hash-ref r 'samples)) (hash-ref r 'variant)))
    (define-values (step-ms phases) (timed-steps p 'item 1))
    (check-true (<= (for/sum ([ph (in-list phases)]) (cdr ph)) step-ms)
                "the phases fit inside the step")))
