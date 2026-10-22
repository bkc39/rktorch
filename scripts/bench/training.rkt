#lang racket/base

(require (only-in compiler/cm managed-compile-zo)
         (only-in json read-json write-json)
         (only-in racket/file call-with-atomic-output-file)
         (only-in racket/list drop last remove-duplicates)
         (only-in racket/match match)
         (only-in "harness.rkt" host-meta summarize)
         (only-in "shim.rkt" aten-threads checkout-module torch-dir))

(provide (struct-out recorded-run)
         collection-policy
         collect-settings
         record-steps!
         read-steps
         analyze-steps)

(struct recorded-run (events wall meta torch-dir) #:transparent)

(define (drain receiver)
  (let loop ([acc '()])
    (define event (sync/timeout 0 receiver))
    (if event
        (loop (cons (vector-ref event 2) acc))
        (reverse acc))))

(define (write-steps out data)
  (call-with-atomic-output-file out (lambda (port _tmp) (write-json data port))))

(define (share text)
  (define x (string->number text 10 'number-or-false 'decimal-as-exact))
  (and (real? x) (< 0 x) (<= x 1) x))

(define (collection-policy #:torch [dir torch-dir])
  (match (checkout-module #:torch dir "foreign" "raw" "pressure-settings.rkt")
    [#f #f]
    [m (list (dynamic-require m 'native-collect-at-troughs)
             (dynamic-require m 'native-collect-budget))]))

(define collect-variables '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" "RKTORCH_BENCH_COLLECT_BUDGET"))

(define (call-with-collection-policy env policy thunk)
  (match policy
    [#f
     (match (filter env collect-variables)
       ['() (thunk)]
       [(cons name _)
        (raise-user-error 'record-steps "~a needs a torch with foreign/raw/pressure-settings.rkt"
                          name)])]
    [(list at-troughs budget)
     (define collect (collect-settings env (at-troughs) (budget)))
     (parameterize ([at-troughs (hash-ref collect 'at-troughs)]
                    [budget (hash-ref collect 'budget)])
       (thunk))]))

(define (collect-settings env default-at-troughs default-budget)
  (define at-troughs
    (match (env "RKTORCH_BENCH_COLLECT_AT_TROUGHS")
      [#f default-at-troughs]
      ["0" #f]
      ["1" #t]
      [text (raise-user-error 'record-steps "RKTORCH_BENCH_COLLECT_AT_TROUGHS is 0 or 1, got ~s"
                              text)]))
  (define budget
    (match (env "RKTORCH_BENCH_COLLECT_BUDGET")
      [#f default-budget]
      [_ #:when (not at-troughs)
         (raise-user-error 'record-steps
                           "RKTORCH_BENCH_COLLECT_BUDGET is read only at troughs, which are off")]
      [(app share (? values x)) x]
      [text (raise-user-error 'record-steps
                              "RKTORCH_BENCH_COLLECT_BUDGET is a share in (0, 1], got ~s" text)]))
  (hasheq 'at-troughs at-troughs 'budget budget))

(define (record-steps! runner out)
  (define device-type (dynamic-require 'torch 'device-type))
  (define accelerator-if-available (dynamic-require 'torch 'accelerator-if-available))
  (define meta (host-meta #:device (device-type (accelerator-if-available))
                          #:libtorch ((dynamic-require 'torch 'torch-version))
                          #:threads (aten-threads)
                          #:repo torch-dir))
  (define torch-dir-string (path->string torch-dir))
  (write-steps out (hasheq 'meta meta 'torch_dir torch-dir-string))
  (managed-compile-zo runner)
  (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-step))
  (define policy (collection-policy))
  (define t0 (current-inexact-monotonic-milliseconds))
  (call-with-collection-policy getenv policy
                               (lambda () (dynamic-require `(submod (file ,runner) main) #f)))
  (define wall (- (current-inexact-monotonic-milliseconds) t0))
  (define steps (map vector->list (drain receiver)))
  (write-steps out (hasheq 'wall_ms wall 'steps steps 'meta meta 'torch_dir torch-dir-string)))

(define (read-steps path)
  (define data (call-with-input-file path read-json))
  (recorded-run (map list->vector (hash-ref data 'steps '()))
                (hash-ref data 'wall_ms #f)
                (hash-ref data 'meta #f)
                (hash-ref data 'torch_dir #f)))

(define (differences xs)
  (for/list ([a (in-list xs)] [b (in-list (cdr xs))]) (- b a)))

(define (every-nth xs n)
  (for/list ([x (in-list xs)] [i (in-naturals 1)] #:when (zero? (modulo i n))) x))

(define (column events k)
  (map (lambda (e) (vector-ref e k)) events))

(define (timed-steps events which)
  (define ids (remove-duplicates (column events 0)))
  (define id (and (pair? ids) (if (eq? which 'last) (last ids) (car ids))))
  (filter (lambda (e) (equal? (vector-ref e 0) id)) events))

(define (analyze-steps events
                       #:epochs [epochs #f]
                       #:warmup [warmup 0]
                       #:optimizer [which 'first])
  (define steps (timed-steps events which))
  (define total (length steps))
  (define per-epoch
    (and epochs (>= epochs 2) (zero? (remainder total epochs)) (quotient total epochs)))
  (define skip (if epochs (or per-epoch total) warmup))
  (define measured (if (< skip total) (drop steps (max 0 (sub1 skip))) '()))
  (define ms (column measured 1))
  (define gc (column measured 2))
  (define intervals (if (pair? ms) (differences ms) '()))
  (define boundaries (if per-epoch (every-nth steps per-epoch) '()))
  (define epoch-ms (if (pair? boundaries) (differences (column boundaries 1)) '()))
  (define epoch-gc (if (pair? boundaries) (differences (column boundaries 2)) '()))
  (cond
    [(null? intervals)
     (hasheq 'steps total
             'error (format "no measured steps: ~a steps, epochs ~a, warmup ~a"
                            total epochs warmup))]
    [else
     (define step-stats (summarize intervals))
     (define span (- (last ms) (car ms)))
     (hasheq 'steps total
             'steps_per_epoch per-epoch
             'measured_steps (length intervals)
             'step_ms step-stats
             'steps_per_s (/ (* 1000.0 (length intervals)) span)
             'steps_per_s_median (/ 1000.0 (hash-ref step-stats 'median))
             'epoch_s (map (lambda (x) (/ x 1000.0)) epoch-ms)
             'epoch_gc_ms epoch-gc
             'gc_ms (- (last gc) (car gc)))]))

(module+ test
  (require (only-in racket/file make-temporary-directory make-temporary-file)
           (only-in racket/match match-define)
           rackunit
           (only-in "harness.rkt" with-temporary-path))

  (define (steps-at times #:id [id 7] #:gc [gc (lambda (t) (quotient t 10))])
    (for/list ([t (in-list times)]) (vector id (exact->inexact t) (gc t))))

  (test-case "epochs after the first are timed from the last step of each"
    (define events
      (append (steps-at '(100 110 120))
              (steps-at '(200 210 220))
              (steps-at '(300 312 324))))
    (define a (analyze-steps events #:epochs 3))
    (check-equal? (hash-ref a 'steps) 9)
    (check-equal? (hash-ref a 'steps_per_epoch) 3)
    (check-equal? (hash-ref a 'epoch_s) '(0.1 0.104))
    (check-equal? (hash-ref a 'measured_steps) 6)
    (check-equal? (hash-ref (hash-ref a 'step_ms) 'median) 12.0)
    (check-= (hash-ref a 'steps_per_s) (/ 6000.0 204) 1e-9)
    (check-= (hash-ref a 'steps_per_s_median) (/ 1000.0 12) 1e-9)
    (check-equal? (hash-ref a 'epoch_gc_ms) '(10 10))
    (check-equal? (hash-ref a 'gc_ms) 20))

  (test-case "the first optimizer to step is timed, or the last to start"
    (define events
      (for*/list ([t (in-list '(10 20 30 40))]
                  [id (in-list '(1 2))])
        (vector id (exact->inexact (+ t id)) 0)))
    (define a (analyze-steps events #:epochs 2))
    (check-equal? (hash-ref a 'steps) 4)
    (check-equal? (hash-ref a 'epoch_s) '(0.02))
    (define phases (append (steps-at '(1 2 3 4) #:id 1) (steps-at '(10 20 40 80) #:id 2)))
    (check-equal? (hash-ref (analyze-steps phases #:epochs 2 #:optimizer 'last) 'epoch_s)
                  '(0.06)))

  (test-case "a run by steps skips its warmup"
    (define a (analyze-steps (steps-at '(0 50 60 70 80 95)) #:warmup 2))
    (check-equal? (hash-ref a 'measured_steps) 4)
    (check-equal? (hash-ref (hash-ref a 'step_ms) 'median) 10.0)
    (check-equal? (hash-ref a 'epoch_s) '())
    (check-false (hash-ref a 'steps_per_epoch)))

  (test-case "uneven epochs and empty runs report rather than guess"
    (check-regexp-match #rx"no measured steps"
                        (hash-ref (analyze-steps (steps-at '(1 2 3 4 5)) #:epochs 2) 'error))
    (check-regexp-match #rx"no measured steps"
                        (hash-ref (analyze-steps '() #:warmup 3) 'error))
    (check-regexp-match #rx"no measured steps"
                        (hash-ref (analyze-steps (steps-at '(1 2)) #:epochs 1) 'error)))

  (test-case "a recorded run round-trips through its file"
    (define run
      (with-temporary-path (path (make-temporary-file "steps-~a.json"))
        (call-with-output-file path #:exists 'truncate
          (lambda (port)
            (write-json (hasheq 'wall_ms 5.0 'steps '((1 2.0 3)) 'torch_dir "/t/torch/") port)))
        (read-steps path)))
    (check-equal? (recorded-run-events run) (list (vector 1 2.0 3)))
    (check-equal? (recorded-run-wall run) 5.0)
    (check-false (recorded-run-meta run))
    (check-equal? (recorded-run-torch-dir run) "/t/torch/"))

  (test-case "a run that dies after the recorder starts leaves its meta"
    (define run
      (with-temporary-path (path (make-temporary-file "steps-~a.json"))
        (write-steps path (hasheq 'meta (hasheq 'git_sha "abc") 'torch_dir "/t/torch/"))
        (read-steps path)))
    (check-equal? run (recorded-run '() #f (hasheq 'git_sha "abc") "/t/torch/")))

  (define ((env-of kvs) name) (cond [(assoc name kvs) => cdr] [else #f]))

  (test-case "the collection policy comes from the variant's settings"
    (match-define (list at-troughs budget) (collection-policy))
    (define (settings . kvs)
      (collect-settings (env-of kvs) (at-troughs) (budget)))
    (check-equal? (settings) (hasheq 'at-troughs #t 'budget 1/20))
    (check-equal? (settings '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" . "1")
                            '("RKTORCH_BENCH_COLLECT_BUDGET" . "0.2"))
                  (hasheq 'at-troughs #t 'budget 1/5))
    (check-equal? (settings '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" . "0"))
                  (hasheq 'at-troughs #f 'budget 1/20))
    (check-exn #rx"BUDGET is read only at troughs, which are off"
               (lambda () (settings '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" . "0")
                                    '("RKTORCH_BENCH_COLLECT_BUDGET" . "0.2"))))
    (check-exn #rx"AT_TROUGHS is 0 or 1, got \"no\""
               (lambda () (settings '("RKTORCH_BENCH_COLLECT_AT_TROUGHS" . "no"))))
    (for ([bad (in-list '("0" "1.5" "x"))])
      (check-exn #rx"BUDGET is a share in [(]0, 1[]]"
                 (lambda () (settings (cons "RKTORCH_BENCH_COLLECT_BUDGET" bad))))))

  (test-case "a torch without the collection policy runs with its defaults and refuses settings"
    (with-temporary-path (old (make-temporary-directory "torch-~a"))
      (check-false (collection-policy #:torch old))
      (check-equal? (call-with-collection-policy (env-of '()) #f (lambda () 'ran)) 'ran)
      (check-exn #rx"RKTORCH_BENCH_COLLECT_BUDGET needs a torch with"
                 (lambda ()
                   (call-with-collection-policy (env-of '(("RKTORCH_BENCH_COLLECT_BUDGET" . "0.2")))
                                                #f void)))))

  (test-case "a write that fails leaves the last complete record"
    (with-temporary-path (path (make-temporary-file "steps-~a.json"))
      (write-steps path (hasheq 'wall_ms 5.0 'steps '((1 2.0 3))))
      (check-exn exn:fail? (lambda () (write-steps path (hasheq 'steps (list 1 (void))))))
      (check-equal? (read-steps path) (recorded-run (list (vector 1 2.0 3)) 5.0 #f #f)))))
