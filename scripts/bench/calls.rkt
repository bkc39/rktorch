#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in torch reclaim-native-memory!)
         (only-in torch/foreign/raw/fault crossing-hook)
         (only-in torch/foreign/raw/race-points race-hook)
         (only-in "harness.rkt" summarize)
         (only-in "workloads/phases.rkt" phased-example phased-phases phased-settings run-phases!))

(provide call-counts
         call-records)

(define-syntax-parse-rule (with-hooks #:crossing on-crossing:expr #:tensor on-tensor:expr
                            body:expr ...+)
  (let ([saved-crossing (unbox crossing-hook)]
        [saved-race (unbox race-hook)]
        [count-crossing on-crossing]
        [count-tensor on-tensor])
    (dynamic-wind
     (lambda ()
       (set-box! crossing-hook count-crossing)
       (set-box! race-hook (lambda (name _subject)
                             (when (eq? name 'op-returned) (count-tensor)))))
     (lambda () body ...)
     (lambda ()
       (set-box! crossing-hook saved-crossing)
       (set-box! race-hook saved-race)))))

(define (call-counts p)
  (define crossings 0)
  (define tensors 0)
  (define counts '())
  (reclaim-native-memory!)
  (with-hooks #:crossing (lambda () (set! crossings (add1 crossings)))
              #:tensor (lambda () (set! tensors (add1 tensors)))
    (run-phases! p #:around (lambda (name thunk)
                              (set! crossings 0)
                              (set! tensors 0)
                              (thunk)
                              (set! counts (cons (list name crossings tensors) counts)))))
  (reverse counts))

(define (call-records p #:warmup [warmup 2] #:rounds [rounds 3])
  (for ([_ (in-range warmup)])
    (run-phases! p))
  (define per-round (for/list ([_ (in-range rounds)]) (call-counts p)))
  (define phases (map car (phased-phases p)))
  (define (phase-record name crossings tensors)
    (hasheq 'schema 1 'suite "calls" 'group "phase" 'impl "rktorch"
            'case (phased-example p) 'variant (symbol->string name) 'unit "calls"
            'settings (phased-settings p)
            'samples crossings 'stats (summarize crossings)
            'tensors tensors 'tensor_stats (summarize tensors)))
  (define (column name k)
    (for/list ([round (in-list per-round)])
      (for/first ([row (in-list round)] #:when (eq? (car row) name)) (list-ref row k))))
  (define (totals k)
    (for/list ([round (in-list per-round)]) (for/sum ([row (in-list round)]) (list-ref row k))))
  (append (for/list ([name (in-list phases)])
            (phase-record name (column name 1) (column name 2)))
          (list (phase-record 'step (totals 1) (totals 2)))))

(module+ test
  (require (only-in racket/list split-at)
           rackunit
           (only-in "workloads/phases.rkt" mnist-phases resnet-phases))

  (test-case "every phase of a step counts its crossings and its accounted tensors"
    (for ([p (in-list (list (mnist-phases 'cpu #:batch 4 #:samples 16)
                            (resnet-phases 'cpu #:batch 2 #:samples 8 #:base 4)))])
      (define records (call-records p #:warmup 1 #:rounds 2))
      (define names (map (lambda (r) (hash-ref r 'variant)) records))
      (check-equal? names (append (map (lambda (ph) (symbol->string (car ph))) (phased-phases p))
                                  '("step")))
      (for ([r (in-list records)])
        (check-equal? (length (hash-ref r 'samples)) 2)
        (check-true (andmap positive? (hash-ref r 'samples)) (hash-ref r 'variant))
        (check-true (<= (hash-ref (hash-ref r 'tensor_stats) 'median)
                        (hash-ref (hash-ref r 'stats) 'median))
                    "a tensor returned is a crossing too"))
      (define-values (phase-rows step-rows) (split-at records (sub1 (length records))))
      (check-equal? (hash-ref (car step-rows) 'samples)
                    (apply map + (map (lambda (r) (hash-ref r 'samples)) phase-rows)))))

  (test-case "the hooks are restored after counting"
    (check-false (unbox crossing-hook))
    (check-false (unbox race-hook))))
