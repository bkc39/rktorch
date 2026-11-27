#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in racket/file file->string)
         (only-in racket/match match)
         (only-in racket/string string-split)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>)
         (only-in (submod torch/foreign/raw/accounts private) account-counters accounts)
         (only-in torch/foreign/raw/tallies counters-entries)
         (only-in torch/tests/private/clock
                  clock-gettime clock-id make-timespec os-thread-id timespec->milliseconds)
         (only-in (submod "threads.rkt" private) [task-dir tasks]))

(provide (struct-out gc-event)
         (struct-out span)
         call-with-gc-log
         gc-overlap-ms
         os-thread-id
         published-entries
         record-span!
         spans
         task-cpu-ms
         thread-cpu-ms
         timed
         with-gc-log)

(define thread-clock (clock-id 'thread-cputime))

(define (thread-cpu-ms)
  (define ts (make-timespec 0 0))
  (and thread-clock (zero? (clock-gettime thread-clock ts)) (timespec->milliseconds ts)))

(struct span (kind tid wall₀ wall₁ cpu₀ cpu₁) #:prefab)

(define recorded (box '()))

(define (record-span! s)
  (let retry ()
    (define old (unbox recorded))
    (unless (box-cas! recorded old (cons s old))
      (retry))))

(define (spans #:clear? [clear? #f])
  (begin0 (reverse (unbox recorded))
    (when clear? (set-box! recorded '()))))

(define (timed kind thunk)
  (define tid (os-thread-id))
  (define c₀ (thread-cpu-ms))
  (define w₀ (current-inexact-milliseconds))
  (call-with-values thunk
                    (lambda results
                      (record-span! (span kind tid w₀ (current-inexact-milliseconds)
                                          c₀ (thread-cpu-ms)))
                      (apply values results))))

(struct gc-event (mode pre post cpu-ms start end live) #:prefab)

(define (gc-info->event info live)
  (match (struct->vector info)
    [(vector _ mode pre _ _ post _ cpu₀ cpu₁ start end)
     (gc-event mode pre post (- cpu₁ cpu₀) start end live)]))

(define (published-entries)
  (for/sum ([a (in-list (hash-values (unbox accounts)))])
    (~> a account-counters counters-entries unbox (max 0 _))))

(define (call-with-gc-log thunk #:live [live (lambda () #f)])
  (define receiver (make-log-receiver (current-logger) 'debug 'GC))
  (define events (box '()))
  (define stop (make-semaphore 0))
  (define stopped-at (box +inf.0))
  (define (take! v)
    (match v
      [(vector _ _ (? prefab-struct-key info) _)
       (set-box! events (cons (gc-info->event info (live)) (unbox events)))]
      [_ (void)]))
  (define (drain!)
    (match (sync/timeout 0.2 receiver)
      [#f (void)]
      [v (take! v) (drain!)]))
  (define listener
    (thread (lambda ()
              (let loop ()
                (match (sync receiver (wrap-evt stop (lambda (_) 'stop)))
                  ['stop (drain!)]
                  [v (take! v) (loop)])))))
  (define result
    (dynamic-wind void
                  thunk
                  (lambda ()
                    (set-box! stopped-at (current-inexact-milliseconds))
                    (semaphore-post stop)
                    (thread-wait listener))))
  (values result
          (filter (lambda (e) (< (gc-event-start e) (unbox stopped-at))) (reverse (unbox events)))))

(define-syntax-parse-rule (with-gc-log ((~optional (~seq #:live live:expr))) body:expr ...+)
  (call-with-gc-log (lambda () body ...) (~? (~@ #:live live))))

(define (gc-overlap-ms events w₀ w₁)
  (for/sum ([e (in-list events)])
    (max 0.0 (- (min w₁ (gc-event-end e)) (max w₀ (gc-event-start e))))))

(define user-hz-ms 10.0)

(define (stat-cpu-ms fields)
  (match (map string->number (string-split fields))
    [(list-rest _ _ _ _ _ _ _ _ _ _ _ utime stime _) (* user-hz-ms (+ utime stime))]
    [_ 0.0]))

(define (task-cpu-ms)
  (for/hash ([tid (in-list (if (directory-exists? tasks) (directory-list tasks) '()))])
    (define fields
      (with-handlers ([exn:fail:filesystem? (lambda (_e) #f)])
        (~> (build-path tasks tid "stat") file->string (regexp-match #rx"\\) (.*)$" _))))
    (values (string->number (path->string tid))
            (match fields
              [(list _ rest) (stat-cpu-ms rest)]
              [_ 0.0]))))

(module+ test
  (require (only-in racket/list last)
           (only-in racket/os getpid)
           rackunit
           (only-in torch/foreign/raw/accounts ledger-entries)
           (only-in torch/tests/private/race-harness with-race-hook))

  (test-case "a timed thunk returns its values and records one span on this OS thread"
    (spans #:clear? #t)
    (define-values (a b) (timed 'probe (lambda () (values 1 2))))
    (check-equal? (list a b) '(1 2))
    (match (spans #:clear? #t)
      [(list (span 'probe tid w₀ w₁ _ _))
       (check-equal? tid (os-thread-id))
       (check-true (<= w₀ w₁))]))

  (test-case "the thread clock advances with work on this thread, where the OS has one"
    (define c₀ (thread-cpu-ms))
    (for/fold ([x 0]) ([i (in-range 2000000)]) (+ x i))
    (when c₀ (check-true (> (thread-cpu-ms) c₀))))

  (test-case "a forced collection is logged with its mode, real interval and process CPU"
    (define-values (v events)
      (with-gc-log (#:live (lambda () 7)) (collect-garbage 'major) 'done))
    (check-equal? v 'done)
    (check-true (pair? events))
    (match (last events)
      [(gc-event mode _ _ cpu start end live)
       (check-not-false (memq mode '(major minor incremental)))
       (check-true (<= start end))
       (check-true (>= cpu 0))
       (check-equal? live 7)]))

  (test-case "a body that raises still stops and drains the listener"
    (define owner (make-custodian))
    (check-exn #rx"body failed"
               (lambda ()
                 (parameterize ([current-custodian owner])
                   (with-gc-log () (collect-garbage 'minor) (error 'probe "body failed")))))
    (check-equal? (filter (lambda (t) (and (thread? t) (not (thread-dead? t))))
                          (custodian-managed-list owner (current-custodian)))
                  '()))

  (test-case "the published live entries read without a flush and agree with the ledger after one"
    (define (flushes thunk)
      (define n (box 0))
      (with-race-hook (lambda (name _subject)
                        (when (eq? name 'tallies-flushing) (set-box! n (add1 (unbox n)))))
        (thunk))
      (unbox n))
    (check-equal? (flushes published-entries) 0)
    (check-true (positive? (flushes ledger-entries)))
    (check-equal? (published-entries) (ledger-entries)))

  (test-case "overlap counts only the part of a collection inside the span"
    (define events (list (gc-event 'minor 0 0 1 10.0 20.0 #f) (gc-event 'major 0 0 1 30.0 40.0 #f)))
    (check-equal? (gc-overlap-ms events 15.0 35.0) 10.0)
    (check-equal? (gc-overlap-ms events 0.0 5.0) 0.0))

  (test-case "a task's CPU is its user and system ticks, fields 14 and 15 of its stat line"
    (check-equal? (stat-cpu-ms "S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15") 230.0)
    (check-equal? (stat-cpu-ms "S 1") 0.0))

  (test-case "this process's tasks read with their CPU, main among them where /proc exists"
    (define cpu (task-cpu-ms))
    (when (directory-exists? tasks)
      (check-true (hash-has-key? cpu (getpid))))))
