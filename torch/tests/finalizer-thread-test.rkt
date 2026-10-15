#lang racket/base

(module+ test
  (require (only-in racket/sandbox
                    kill-evaluator sandbox-eval-limits sandbox-memory-limit
                    sandbox-path-permissions sandbox-security-guard)
           rackunit
           (only-in scribble/example make-base-eval)
           syntax/parse/define)

  (define workers 8)
  (define per-worker 2000)

  ;; A tensor registers its finalizer inside the allocator's atomic section,
  ;; which parallel threads do not race through; the drain's canary registers
  ;; outside one, so each worker's first registration is a drain, as an OOM
  ;; retry's is when the worker's first allocation fails.
  (define program
    `((require (only-in ffi/unsafe register-finalizer)
               (only-in torch ones)
               (only-in torch/foreign/raw/collector collect-and-wait!))
      (define evaluator-thread (current-thread))
      (define deadline (+ (current-inexact-milliseconds) 30000))
      (define (remaining)
        (max 0 (/ (- deadline (current-inexact-milliseconds)) 1000.0)))
      (define (bump! b)
        (let loop ()
          (define n (unbox b))
          (cond
            [(box-cas! b n (add1 n)) (add1 n)]
            [else (loop)])))
      (define runners (make-hasheq))
      (define ran (box 0))
      (define all-ran (make-semaphore 0))
      (define (record! _)
        (hash-set! runners (current-thread) #t)
        (cond
          [(= (bump! ran) ,(* workers per-worker)) (semaphore-post all-ran)]
          [else (void)]))
      (define ready (box 0))
      (define go (box #f))
      (define (work)
        (bump! ready)
        (let spin ()
          (cond
            [(unbox go) (void)]
            [else (spin)]))
        (collect-and-wait!)
        (void (ones 2 2))
        (for ([_ (in-range ,per-worker)])
          (register-finalizer (box 0) record!)))
      (define threads
        (for/list ([_ (in-range ,workers)])
          (thread work #:pool 'own)))
      (let wait ()
        (cond
          [(and (< (unbox ready) ,workers) (positive? (remaining))) (sleep 0) (wait)]
          [else (set-box! go #t)]))
      (for ([t (in-list threads)])
        (sync/timeout (remaining) t))
      (let drain ()
        (collect-garbage)
        (cond
          [(sync/timeout 0.05 all-ran) (void)]
          [(positive? (remaining)) (drain)]
          [else (void)]))
      (define (outcome)
        (list (unbox ran)
              (for/sum ([t (in-hash-keys runners)])
                (if (eq? t evaluator-thread) 0 1))))))

  (define (make-fresh-eval)
    (parameterize ([sandbox-memory-limit #f]
                   [sandbox-eval-limits '(60 #f)]
                   [sandbox-security-guard current-security-guard]
                   [sandbox-path-permissions '((exists "/"))])
      (make-base-eval)))

  (define-syntax-parse-rule (with-fresh-eval (ev:id) body:expr ...+)
    (let ([ev (make-fresh-eval)])
      (dynamic-wind void (lambda () body ...) (lambda () (kill-evaluator ev)))))

  (test-case "parallel threads making the first registrations share one finalizer thread"
    (for ([_ (in-range 3)])
      (with-fresh-eval (ev)
        (for ([form (in-list program)])
          (ev form))
        (check-equal? (ev '(outcome)) (list (* workers per-worker) 1))))))
