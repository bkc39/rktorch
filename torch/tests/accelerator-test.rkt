#lang racket/base

(module+ test
  (require (only-in racket/list first last make-list)
           (only-in racket/promise force)
           rackunit
           (only-in torch
                    accelerator-if-available cpu-device cuda-available? cuda-device
                    device device-index device-type mps-available? mps-device
                    mps-if-available tensor)
           (only-in torch/foreign/accelerator mps-usable pick-accelerator)
           (only-in torch/scribblings/examples/04-mlp run-example))

  (define (warnings-during thunk)
    (define receiver (make-log-receiver (current-logger) 'warning 'rktorch-device))
    (define result (thunk))
    (values result
            (let drain ([messages '()])
              (define event (sync/timeout 0 receiver))
              (if event
                  (drain (cons (vector-ref event 1) messages))
                  (reverse messages)))))

  (define (counted-probe outcome)
    (define calls 0)
    (values (λ () (set! calls (add1 calls)) (outcome))
            (λ () calls)))

  (define (out-of-memory)
    (error 'tr_tensor_to "MPS backend out of memory (MPS allocated: 0 bytes)"))

  (test-case "an MPS that cannot allocate falls back to the CPU, probed once and logged once"
    (define-values (probe calls) (counted-probe out-of-memory))
    (define usable (mps-usable probe))
    (define-values (picks logged)
      (warnings-during
       (λ () (for/list ([_ (in-range 3)]) (pick-accelerator #f #t usable)))))
    (check-equal? picks (list (cpu-device) (cpu-device) (cpu-device)))
    (check-equal? (calls) 1)
    (check-equal? (length logged) 1)
    (check-regexp-match #rx"cannot allocate.*MPS backend out of memory" (first logged)))

  (test-case "an MPS that allocates is picked, probed once, and nothing is logged"
    (define-values (probe calls) (counted-probe void))
    (define usable (mps-usable probe))
    (define-values (picks logged)
      (warnings-during
       (λ () (list (pick-accelerator #f #t usable) (pick-accelerator #f #t usable)))))
    (check-equal? picks (list (mps-device) (mps-device)))
    (check-equal? (calls) 1)
    (check-equal? logged '()))

  (test-case "no probe runs when CUDA wins or MPS is not advertised"
    (define-values (probe calls) (counted-probe out-of-memory))
    (define usable (mps-usable probe))
    (check-equal? (pick-accelerator #t #t usable) (cuda-device))
    (check-equal? (pick-accelerator #f #f usable) (cpu-device))
    (check-equal? (calls) 0))

  (test-case "threads asking at once share one probe"
    (define gate (make-semaphore 0))
    (define-values (probe calls)
      (counted-probe (λ () (semaphore-wait gate) (out-of-memory))))
    (define usable (mps-usable probe))
    (define picks (make-vector 4 #f))
    (define workers
      (for/list ([i (in-range 4)])
        (thread
         (λ ()
           (vector-set! picks i
                        (with-handlers ([exn:fail? exn-message])
                          (pick-accelerator #f #t usable)))))))
    (sync (system-idle-evt))
    (semaphore-post gate)
    (for-each thread-wait workers)
    (check-equal? (vector->list picks) (make-list 4 (cpu-device)))
    (check-equal? (calls) 1))

  (test-case "the real probe's failure is caught and logged where there is no MPS"
    (unless (mps-available?)
      (define-values (usable logged) (warnings-during (λ () (force (mps-usable)))))
      (check-false usable)
      (check-equal? (length logged) 1)))

  (define expected-accelerator
    (cond
      [(cuda-available?) 'cuda]
      [(and (mps-available?) (force (mps-usable))) 'mps]
      [else 'cpu]))

  (test-case "accelerator-if-available prefers cuda, then an mps that allocates, then cpu"
    (define picked (accelerator-if-available))
    (check-equal? (device-type picked) expected-accelerator)
    (check-equal? (device-index picked) 0)
    (check-equal? (device (tensor '(1 2 3) #:device picked)) picked))

  (test-case "mps-if-available answers mps only where it allocates"
    (check-equal? (device-type (mps-if-available))
                  (if (eq? expected-accelerator 'mps) 'mps 'cpu))
    (check-equal? (device-index (mps-if-available)) 0))

  (test-case "04-mlp trains on the device accelerator-if-available picks"
    (define-values (losses _net used) (run-example (accelerator-if-available)))
    (check-equal? used (accelerator-if-available))
    (check < (last losses) (first losses))))
