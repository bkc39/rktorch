#lang racket/base

(module stress racket/base
  (require (only-in racket/match match match-define)
           rackunit
           (only-in "../foreign.rkt"
                    backward! finalizer-diagnostics native-memory-limit ones
                    sum)
           (only-in "../nn.rkt" Linear)
           (only-in "private/race-harness.rkt"
                    await-roles! check-ledger churn-once! env-number join-all
                    ledger-snapshot make-roster make-watch seeded-generator
                    settle! spawn stop-roles! until-stopped watch-hook
                    with-race-hook))

  (provide mib phase)

  (define mib (* 1024 1024))

  (define seconds-per-phase (env-number "RKTORCH_STRESS_SECONDS" 0.5))

  (define start-deadline-seconds 60)

  (define (diagnostic key)
    (cdr (assq key (finalizer-diagnostics))))

  (define (ledger-entries)
    (diagnostic 'ledger-entries))

  (define (worker-step k)
    (define rng (seeded-generator 40 (list 'stress k)))
    (lambda () (churn-once! rng)))

  (define (trainer-step batch)
    (define net (Linear 32 32))
    (define x (ones batch 32))
    (lambda () (backward! (sum (net x)))))

  (define (adversary-step)
    (define n 0)
    (lambda ()
      (set! n (add1 n))
      (collect-garbage (if (zero? (modulo n 8)) 'major 'minor))
      (sleep 0.002)))

  (define (role-step label batch)
    (match label
      ['trainer (trainer-step batch)]
      ['adversary (adversary-step)]
      [k (worker-step k)]))

  (define (phase width #:batch [batch 8])
    (settle!)
    (define base (ledger-snapshot))
    (define entries (ledger-entries))
    (define before (diagnostic 'pressure-collections))
    (define w (make-watch))
    (define r (make-roster))
    (define labels (append (build-list width values) '(trainer adversary)))
    (define counts
      (with-race-hook (watch-hook w)
        (define workers
          (parameterize ([native-memory-limit (quotient mib 4)])
            (for/list ([label (in-list labels)])
              (spawn (until-stopped r label (role-step label batch))
                     #:pool 'own
                     #:label label))))
        (await-roles! r labels #:seconds start-deadline-seconds)
        (sleep seconds-per-phase)
        (stop-roles! r)
        (begin0 (join-all workers #:seconds 300)
                (settle!))))
    (match-define (list worker-steps ... trainer-steps _adversary-steps) counts) ;; noqa
    (printf "stress: ~a workers, ~a steps, ~a trainer steps, ~a collections\n"
            width
            (apply + worker-steps)
            trainer-steps
            (- (diagnostic 'pressure-collections) before))
    (check-ledger base w)
    (check-equal? (ledger-entries) entries "ledger entries left behind")
    (for ([label (in-list labels)]
          [n (in-list counts)])
      (check-true (positive? n) (format "role ~a never stepped" label)))))

(module+ test
  (require (only-in racket/future processor-count)
           (only-in racket/match match match-define)
           (only-in racket/port port->string)
           rackunit
           (submod ".." stress))

  (define widths '(2 4 8 16))

  (define (rss-bytes)
    (define status "/proc/self/status")
    (and (file-exists? status)
         (call-with-input-file status
           (lambda (in)
             (for/first ([line (in-lines in)]
                         #:when (regexp-match? #rx"^VmRSS:" line))
               (match-define (list _ kb) (regexp-match #rx"([0-9]+) kB" line))
               (* 1024 (string->number kb)))))))

  ;; The test gate pins OMP_NUM_THREADS=1; this child runs at libtorch's
  ;; default, with a trainer batch past ATen's parallel grain.
  (define default-threads-phase
    '(begin
       (require (only-in ffi/unsafe -> _fun _int get-ffi-obj)
                (only-in rackunit current-check-handler)
                (only-in torch/foreign/raw/native-library native-library)
                (submod torch/tests/race-stress-test stress))
       (define intra-op-threads
         (get-ffi-obj "_ZN2at15get_num_threadsEv" native-library (_fun -> _int)))
       (current-check-handler raise)
       (phase 4 #:batch 4096)
       (printf "intra-op threads: ~a\n" (intra-op-threads))))

  (define (run-at-default-threads program #:within seconds)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([name (in-list '(#"OMP_NUM_THREADS" #"MKL_NUM_THREADS"))])
      (environment-variables-set! env name #f))
    (define-values (sp out in err)
      (parameterize ([current-environment-variables env])
        (subprocess #f #f #f (find-system-path 'exec-file) "-e" (format "~s" program))))
    (close-output-port in)
    (define outputs (list (box "") (box "")))
    (define drains
      (for/list ([p (list out err)]
                 [b (in-list outputs)])
        (thread (lambda () (set-box! b (port->string p))))))
    (define status
      (cond
        [(sync/timeout seconds sp) (subprocess-status sp)]
        [else (subprocess-kill sp #t) 'timed-out]))
    (for ([t (in-list drains)]) (sync/timeout 10 t))
    (close-input-port out)
    (close-input-port err)
    (match-define (list stdout stderr) (map unbox outputs))
    (values status stdout stderr))

  (test-case "a phase at libtorch's default thread count runs its intra-op pool"
    (define-values (status stdout stderr)
      (run-at-default-threads default-threads-phase #:within 300))
    (with-check-info (['stdout stdout] ['stderr stderr])
      (check-equal? status 0)
      (define threads
        (match (regexp-match #rx"intra-op threads: ([0-9]+)" stdout)
          [(list _ n) (string->number n)]
          [#f 0]))
      (check-true (or (= (processor-count) 1) (> threads 1))
                  (format "~a intra-op threads on ~a cores" threads (processor-count)))))

  (test-case "parallel threads churning tensors keep the ledger invariants"
    (define rss-before (rss-bytes))
    (for ([width (in-list widths)])
      (with-check-info (['workers width])
        (phase width)))
    (define rss-after (rss-bytes))
    (cond
      [rss-before
       (printf "stress: RSS ~a MiB -> ~a MiB\n"
               (quotient rss-before mib) (quotient rss-after mib))
       (check-true (< (- rss-after rss-before) (* 512 mib))
                   "RSS grew past the churn's working set")]
      [else (void)])))
