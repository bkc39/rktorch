#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in "harness.rkt" clock-ms rotate summarize)
         (only-in "workloads/phases.rkt"
                  call-in-phased-device phased-example phased-phases phased-settings
                  phased-sync! run-phases! step-phases!))

(provide host-gpu-modes
         timed-steps
         host-gpu-records)

(define host-gpu-modes '(item no-item phase-sync))

(define (timed-steps p mode steps)
  (define sync! (phased-sync! p))
  (define totals (make-hasheq))
  (define (timed name thunk)
    (define t0 (clock-ms))
    (thunk)
    (when (eq? mode 'phase-sync) (sync!))
    (hash-update! totals name (lambda (ms) (+ ms (- (clock-ms) t0))) 0.0))
  (define step-ms
    (call-in-phased-device
     p
     (lambda ()
       (sync!)
       (define t0 (clock-ms))
       (for ([_ (in-range steps)])
         (step-phases! p #:around timed #:skip (if (eq? mode 'no-item) '(loss-item) '())))
       (sync!)
       (/ (- (clock-ms) t0) steps))))
  (values step-ms
          (for/list ([ph (in-list (phased-phases p))] #:when (hash-has-key? totals (car ph)))
            (cons (car ph) (/ (hash-ref totals (car ph)) steps)))))

(struct row (mode step-ms phases))

(define (host-gpu-records p #:proxy [proxy #f] #:steps [steps 20] #:rounds [rounds 5]
                          #:warmup [warmup 5] #:modes [modes host-gpu-modes])
  (for* ([_ (in-range warmup)] [q (in-list (if proxy (list p proxy) (list p)))])
    (run-phases! q))
  (define rotation (if proxy (append modes '(proxy)) modes))
  (define rows
    (for*/list ([i (in-range rounds)] [mode (in-list (rotate rotation i))])
      (define-values (step-ms phases)
        (if (eq? mode 'proxy) (timed-steps proxy 'item steps) (timed-steps p mode steps)))
      (row mode step-ms phases)))
  (define (record group variant samples #:of [q p])
    (hasheq 'schema 1 'suite "host-gpu" 'group group 'impl "rktorch"
            'case (phased-example q) 'variant variant 'unit "ms" 'reps steps
            'settings (phased-settings q) 'samples samples 'stats (summarize samples)))
  (define (of mode) (filter (lambda (r) (eq? (row-mode r) mode)) rows))
  (define (phase-names mode-rows)
    (remove-duplicates (for*/list ([r (in-list mode-rows)] [ph (in-list (row-phases r))])
                         (car ph))))
  (append
   (for*/list ([mode (in-list modes)]
               [mode-rows (in-value (of mode))]
               [name (in-list (cons #f (phase-names mode-rows)))])
     (if name
         (record "phase" (format "~a/~a" mode name)
                 (for/list ([r (in-list mode-rows)]) (cdr (assq name (row-phases r)))))
         (record "step" (symbol->string mode) (map row-step-ms mode-rows))))
   (if proxy
       (list (record "step" "proxy" (map row-step-ms (of 'proxy)) #:of proxy))
       '())))

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           (only-in torch/tests/private/wedge call-with-process-deadline)
           (only-in "workloads/phases.rkt" resnet-phases))

  (define (variants records group)
    (for/list ([r (in-list records)] #:when (equal? (hash-ref r 'group) group))
      (hash-ref r 'variant)))

  (call-with-process-deadline
   600
   (lambda ()
     (test-case "each mode times its phases and its step, and only item mode reads the loss"
       (define p (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4))
       (define records (host-gpu-records p #:steps 2 #:rounds 2 #:warmup 1))
       (check-equal? (variants records "step") '("item" "no-item" "phase-sync"))
       (check-not-false (member "item/loss-item" (variants records "phase")))
       (check-not-false (member "phase-sync/loss-item" (variants records "phase")))
       (check-false (member "no-item/loss-item" (variants records "phase")))
       (for ([r (in-list records)])
         (check-equal? (length (hash-ref r 'samples)) 2)
         (check-true (andmap positive? (hash-ref r 'samples)) (hash-ref r 'variant)))
       (define-values (step-ms phases) (timed-steps p 'item 1))
       (check-true (<= (for/sum ([ph (in-list phases)]) (cdr ph)) step-ms)
                   "the phases fit inside the step"))

     (test-case "a host proxy, the same step at a negligible width, is timed beside it"
       (define p (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 8))
       (define proxy (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4))
       (define records (host-gpu-records p #:proxy proxy #:steps 2 #:rounds 2 #:warmup 1))
       (check-equal? (variants records "step") '("item" "no-item" "phase-sync" "proxy"))
       (match-define (list proxy-record)
         (filter (lambda (r) (equal? (hash-ref r 'variant) "proxy")) records))
       (check-equal? (hash-ref proxy-record 'settings) (hasheq 'batch 2 'base 4))
       (check-equal? (length (hash-ref proxy-record 'samples)) 2)
       (check-true (andmap positive? (hash-ref proxy-record 'samples)))))))
