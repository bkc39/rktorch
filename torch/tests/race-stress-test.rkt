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

  (provide mib phase proc-status-number)

  (define mib (* 1024 1024))

  (define (proc-status-number field)
    (define status "/proc/self/status")
    (define rx (pregexp (format "^~a:\\s+([0-9]+)" field)))
    (and (file-exists? status)
         (call-with-input-file status
           (lambda (in)
             (for/or ([line (in-lines in)])
               (match (regexp-match rx line)
                 [(list _ n) (string->number n)]
                 [#f #f]))))))

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
  (require (only-in racket/file file->string make-temporary-file)
           (only-in racket/match match match* match-define)
           (only-in racket/path path-only)
           (only-in racket/port port->string)
           (only-in setup/getinfo get-info)
           rackunit
           (submod ".." stress))

  (define widths '(2 4 8 16))

  (define (rss-bytes)
    (define kb (proc-status-number "VmRSS"))
    (and kb (* 1024 kb)))

  (define child-seconds 90)

  (define default-threads-width 4)

  (define (default-threads-environment)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([name (in-list '(#"OMP_NUM_THREADS" #"MKL_NUM_THREADS" #"RKTORCH_STRESS_SECONDS"))])
      (environment-variables-set! env name #f))
    env)

  ;; The test gate pins OMP_NUM_THREADS=1; this child runs at libtorch's
  ;; default, with a trainer batch past ATen's parallel grain.
  (define (default-threads-program #:phase? [phase? #t])
    `(begin
       (require (only-in ffi/unsafe -> _fun _int get-ffi-obj)
                (only-in racket/future processor-count)
                (only-in rackunit current-check-handler)
                (only-in torch/foreign/raw/native-library native-library)
                (submod torch/tests/race-stress-test stress))
       (define (native-count name fallback)
         (get-ffi-obj name native-library (_fun -> _int) (lambda () fallback)))
       (define threads-before (proc-status-number "Threads"))
       (current-check-handler raise)
       ,@(if phase? `((phase ,default-threads-width #:batch 4096)) '())
       (printf "report: ~s\n"
               (list ((native-count "omp_get_num_procs" processor-count))
                     ((native-count "_ZN2at15get_num_threadsEv" (lambda () 1)))
                     threads-before
                     (proc-status-number "Threads")))))

  ;; The phase's role threads outlive it, give or take one, and an OpenMP
  ;; team adds intra-op - 1: half a team past the roles tells them apart
  ;; once the team has at least 3 threads to add.
  (define (intra-op-pool-ran? report)
    (match report
      [(list 1 _ _ _) #t]
      [(list _ intra #f #f) (> intra 1)]
      [(list _ intra _ _) #:when (< intra 4) (> intra 1)]
      [(list _ intra before after)
       (>= (- after before) (+ default-threads-width 2 (quotient intra 2)))]
      [_ #f]))

  (define (run-at-default-threads program #:within seconds #:prefix [prefix '()])
    (define custodian (make-custodian))
    (match-define (cons command args)
      (append prefix (list (find-system-path 'exec-file) "-e" (format "~s" program))))
    (define-values (sp out in err)
      (parameterize ([current-custodian custodian]
                     [current-subprocess-custodian-mode 'kill]
                     [current-environment-variables (default-threads-environment)])
        (apply subprocess #f #f #f command args)))
    (close-output-port in)
    (define outputs (list (box "") (box "")))
    (define drains
      (parameterize ([current-custodian custodian])
        (for/list ([p (list out err)]
                   [b (in-list outputs)])
          (thread (lambda () (set-box! b (port->string p)))))))
    (define status
      (cond
        [(sync/timeout seconds sp) (subprocess-status sp)]
        [else (subprocess-kill sp #t) 'timed-out]))
    (for ([t (in-list drains)]) (sync/timeout 10 t))
    (custodian-shutdown-all custodian)
    (match-define (list stdout stderr) (map unbox outputs))
    (values status stdout stderr))

  (define (default-threads-report program #:prefix [prefix '()])
    (define-values (status stdout stderr)
      (run-at-default-threads program #:within child-seconds #:prefix prefix))
    (define report
      (match (regexp-match #rx"report: ([^\n]*)" stdout)
        [(list _ text) (read (open-input-string text))]
        [#f #f]))
    (with-check-info (['stdout stdout] ['stderr stderr])
      (check-equal? status 0)
      (check-pred list? report))
    report)

  (define (alive? pid)
    (define stat (format "/proc/~a/stat" pid))
    (and (file-exists? stat)
         (not (regexp-match? #rx"^[0-9]+ \\(.*\\) [ZX]" (call-with-input-file stat port->string)))))

  (test-case "the child drops the gates' thread pins and the soak length"
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([name (in-list '(#"OMP_NUM_THREADS" #"MKL_NUM_THREADS" #"RKTORCH_STRESS_SECONDS"))])
      (environment-variables-set! env name #"300"))
    (define child (parameterize ([current-environment-variables env])
                    (default-threads-environment)))
    (for ([name (in-list '(#"OMP_NUM_THREADS" #"MKL_NUM_THREADS" #"RKTORCH_STRESS_SECONDS"))])
      (check-false (environment-variables-ref child name) (format "~a" name))))

  (test-case "a phase at libtorch's default thread count runs its intra-op pool"
    (define report (default-threads-report (default-threads-program)))
    (check-true (intra-op-pool-ran? report) (format "procs, intra-op, threads before, after: ~s" report)))

  (test-case "the pool check fails when no phase ran"
    (define report (default-threads-report (default-threads-program #:phase? #f)))
    (match report
      [(list procs _ (? number?) _)
       #:when (> procs 1)
       (check-false (intra-op-pool-ran? report) (format "~s" report))]
      [_ (void)]))

  (test-case "the pool check passes under a one-CPU affinity mask"
    (define taskset (find-executable-path "taskset"))
    (when taskset
      (define report (default-threads-report (default-threads-program)
                                             #:prefix (list taskset "-c" "0")))
      (check-true (intra-op-pool-ran? report) (format "~s" report))))

  (define (raco-seconds)
    (define cell
      (build-path (path-only (collection-file-path "info.rkt" "torch"))
                  'up "nix" "matrix" "racket-cell.sh"))
    (match* ((assoc "tests/race-stress-test.rkt" ((get-info '("torch")) 'test-timeouts))
             (and (file-exists? cell)
                  (regexp-match #rx"raco test --timeout ([0-9]+) [.]/torch/"
                                (file->string cell))))
      [((list _ seconds) _) seconds]
      [(#f (list _ seconds)) (string->number seconds)]
      [(#f #f) #f]))

  (test-case "the children's bounds all fire before the test gate's"
    (match (raco-seconds)
      [#f (void)]
      [seconds (check < (* 3 child-seconds) seconds)]))

  (test-case "a child past its bound is killed and reported"
    (define start (current-inexact-monotonic-milliseconds))
    (define-values (status _stdout _stderr) (run-at-default-threads '(sleep 30) #:within 2))
    (check-equal? status 'timed-out)
    (check < (- (current-inexact-monotonic-milliseconds) start) 15000.0))

  (test-case "a child dies with the custodian that ran it"
    (when (directory-exists? "/proc/self")
      (define pid-file (make-temporary-file "race-stress-pid-~a"))
      (define custodian (make-custodian))
      (parameterize ([current-custodian custodian])
        (thread
         (lambda ()
           (run-at-default-threads
            `(begin
               (require (only-in racket/os getpid))
               (call-with-output-file ,(path->string pid-file) #:exists 'truncate
                 (lambda (o) (write (getpid) o)))
               (sleep 30))
            #:within 60))))
      (define pid
        (for/or ([_ (in-range 300)])
          (sleep 0.1)
          (match (call-with-input-file pid-file read)
            [(? number? n) n]
            [_ #f])))
      (custodian-shutdown-all custodian)
      (delete-file pid-file)
      (check-pred number? pid)
      (check-false (for/and ([_ (in-range 100)])
                     (sleep 0.1)
                     (alive? pid))
                   (format "child ~a outlived its custodian" pid))))

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
