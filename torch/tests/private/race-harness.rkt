#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe/atomic
                  call-as-atomic in-atomic-mode? make-uninterruptible-lock
                  uninterruptible-lock-acquire uninterruptible-lock-release)
         (only-in racket/dict dict-ref)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in racket/string string-join)
         (only-in rackunit define-check fail-check)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  + @ finalizer-failures native-memory-use
                  reclaim-native-memory! sum zeros)
         (only-in (submod "../../foreign.rkt" unsafe) tensor-free!)
         (only-in "../../foreign/raw/cas.rkt" cas-claim! cas-update!)
         (only-in "../../foreign/raw/fault.rkt" native-faulted)
         (only-in "../../foreign/raw/memory.rkt" native-memory-use/fold)
         (only-in "../../foreign/raw/pressure.rkt"
                  live-bytes-by-device reset-pressure-state!)
         (only-in "../../foreign/raw/race-points.rkt" race-hook)
         (only-in "../../foreign/raw/regions.rkt" with-cleanup))

(provide with-race-hook
         with-parked
         while-inside
         hooks
         labelled
         make-gate
         gate-hook
         wait-for-arrival
         release-gate!
         release-pending?
         make-watch
         watch-hook
         watch-max-holders
         watch-release-count
         watch-released
         watch-freed-count
         watch-finalized
         watch-registered
         fuzz-hook
         spawn
         join
         stop!
         worker-thread
         join-all
         make-roster
         until-stopped
         await-roles!
         stop-roles!
         settle!
         env-number
         seeded-generator
         churn-once!
         ledger-snapshot
         ledger-violations
         check-ledger
         check-known-failure)

(define-syntax-parse-rule (with-race-hook hook:expr body:expr ...+)
  (let ([installed hook]
        [previous (unbox race-hook)])
    (dynamic-wind (lambda () (set-box! race-hook installed))
                  (lambda () body ...)
                  (lambda () (set-box! race-hook previous)))))

(define (env-number name default)
  (define text (getenv name))
  (or (and text (string->number text)) default))

(define ((hooks . all) name subject)
  (for ([h (in-list all)])
    (h name subject)))

(define thread-label (make-thread-cell #f))

(define (current-label)
  (thread-cell-ref thread-label))

(define ((labelled label) _subject)
  (eq? (current-label) label))

(struct gate (point admits? resume state))

(define (make-gate point #:when [admits? (lambda (_subject) #t)])
  (gate point admits? (make-semaphore 0) (box 'waiting)))

(define (take-gate! g)
  (cas-claim! (gate-state g) 'waiting 'taken #:site 'gate))

(define ((gate-hook g) name subject)
  (define taken?
    (and (eq? name (gate-point g))
         ((gate-admits? g) subject)
         (take-gate! g)))
  (cond
    [(not taken?) (void)]
    [(in-atomic-mode?) (set-box! (gate-state g) 'atomic)]
    [else
     (set-box! (gate-state g) 'parked)
     (semaphore-wait (gate-resume g))]))

(define (wait-for-arrival g #:seconds [seconds 30])
  (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
  (let poll ()
    (define state (unbox (gate-state g)))
    (cond
      [(memq state '(parked atomic)) state]
      [(> (current-inexact-milliseconds) deadline) #f]
      [else (sleep 0.001) (poll)])))

(define (release-gate! g)
  (semaphore-post (gate-resume g)))

(define (release-pending? g)
  (and (sync/timeout 0 (semaphore-peek-evt (gate-resume g))) #t))

(struct watch (holders
               [max-holders #:mutable]
               releases
               [released #:mutable]
               [finalized #:mutable]
               [doubled #:mutable]
               unaccounts
               [unaccounted-twice #:mutable]
               freed
               [registered #:mutable]))

(define (make-watch)
  (watch (make-hasheq) 0 (make-weak-hasheq) 0 0 0 (make-weak-hasheq) 0
         (make-weak-hasheq) 0))

(define watch-lock (make-uninterruptible-lock))

(define (call-with-watch-lock thunk)
  (dynamic-wind (lambda () (uninterruptible-lock-acquire watch-lock))
                thunk
                (lambda () (uninterruptible-lock-release watch-lock))))

(define (note-unaccount! w key)
  (define unaccounts (watch-unaccounts w))
  (hash-update! unaccounts key add1 0)
  (when (= 2 (hash-ref unaccounts key))
    (set-watch-unaccounted-twice! w (add1 (watch-unaccounted-twice w)))))

(define (note-release! w handle)
  (define releases (watch-releases w))
  (hash-update! releases handle add1 0)
  (set-watch-released! w (add1 (watch-released w)))
  (when (= 2 (hash-ref releases handle))
    (set-watch-doubled! w (add1 (watch-doubled w)))))

(define (watch-release-count w handle)
  (call-with-watch-lock (lambda () (hash-ref (watch-releases w) handle 0))))

(define (watch-freed-count w handle)
  (call-with-watch-lock (lambda () (hash-ref (watch-freed w) handle 0))))

(define (note-holder! w holder)
  (define holders (watch-holders w))
  (for ([t (in-list (hash-keys holders))] #:when (thread-dead? t))
    (hash-remove! holders t))
  (hash-set! holders holder #t)
  (set-watch-max-holders! w (max (watch-max-holders w) (hash-count holders))))

(define ((watch-hook w) name subject)
  (call-with-watch-lock
   (lambda ()
     (case name
       [(collector-claimed) (note-holder! w subject)]
       [(collector-releasing) (hash-remove! (watch-holders w) subject)]
       [(free-unaccounted) (note-release! w subject)]
       [(free-released) (hash-update! (watch-freed w) subject add1 0)]
       [(entry-unaccounted) (note-unaccount! w subject)]
       [(handle-registered) (set-watch-registered! w (add1 (watch-registered w)))]
       [(ledger-entry-added) (hash-remove! (watch-unaccounts w) subject)]
       [(finalizer-releasing)
        (set-watch-finalized! w (add1 (watch-finalized w)))
        (note-release! w subject)]
       [else (void)]))))

(define (seeded-generator seed label)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (~> label equal-hash-code (+ (* 1000003 seed)) (modulo 2147483647) random-seed))
  g)

(define (perturb! r)
  (cond
    [(< r 0.5) (void)]
    [(< r 0.9) (sleep 0)]
    [else (sleep (* 0.002 r))]))

(define (fuzz-hook seed)
  (define streams (make-thread-cell #f))
  (define (generator)
    (or (thread-cell-ref streams)
        (let ([g (seeded-generator seed (current-label))])
          (thread-cell-set! streams g)
          g)))
  (lambda (_name _subject)
    (unless (in-atomic-mode?)
      (~> (generator) random perturb!))))

(struct worker (thread outcome))

(define (spawn thunk #:pool [pool #f] #:label [label #f])
  (define outcome (box #f))
  (define (body)
    (thread-cell-set! thread-label label)
    (set-box! outcome
              (with-handlers ([(lambda (_) #t) (lambda (e) (cons 'raised e))])
                (cons 'returned (thunk)))))
  (worker (if pool (thread body #:pool pool) (thread body)) outcome))

(define (join w #:seconds [seconds 60])
  (sync/timeout seconds (thread-dead-evt (worker-thread w)))
  (match (unbox (worker-outcome w))
    [(cons 'raised e) (raise e)]
    [(cons 'returned v) v]
    [#f
     (kill-thread (worker-thread w))
     (error 'join "a worker did not finish within ~a s, or was killed" seconds)]))

(define (stop! w #:seconds [seconds 30])
  (define t (worker-thread w))
  (kill-thread t)
  (unless (sync/timeout seconds (thread-dead-evt t))
    (error 'stop! "a killed worker was still running after ~a s" seconds)))

(define (join-all workers #:seconds [seconds 120])
  (map (lambda (w) (join w #:seconds seconds)) workers))

(struct roster (started stop?))

(define (make-roster)
  (roster (box '()) (box #f)))

(define (note-started! started label)
  (cas-update! started (lambda (seen) (cons label seen))))

(define ((until-stopped r label step!))
  (step!)
  (note-started! (roster-started r) label)
  (let loop ([n 1])
    (cond
      [(unbox (roster-stop? r)) n]
      [else (step!) (loop (add1 n))])))

(define (stop-roles! r)
  (set-box! (roster-stop? r) #t))

(define (await-roles! r labels #:seconds [seconds 60])
  (define deadline (+ (current-inexact-milliseconds) (* 1000 seconds)))
  (let loop ()
    (define missing (~> (roster-started r) unbox (remove* labels)))
    (cond
      [(null? missing) (void)]
      [(> (current-inexact-milliseconds) deadline)
       (stop-roles! r)
       (error 'await-roles!
              "roles never completed a first step within ~a s: ~a"
              seconds missing)]
      [else (sleep 0.001) (loop)])))

(define (parking-hook g w)
  (cond
    [w (hooks (watch-hook w) (gate-hook g))]
    [else (gate-hook g)]))

(define-syntax-parse-rule
  (with-parked [worker:id spawned:expr #:at g:expr
                (~optional (~seq #:watch w:expr) #:defaults ([w #'#f]))]
    body:expr ...+)
  (let ([the-gate g])
    (with-race-hook (parking-hook the-gate w)
      (let ([worker spawned])
        (with-cleanup body ... #:finally (stop! worker))))))

(define (while-inside enter observe #:pool [pool #f])
  (define inside (make-semaphore 0))
  (define leave (make-semaphore 0))
  (define w
    (spawn (lambda ()
             (enter (lambda ()
                      (semaphore-post inside)
                      (semaphore-wait leave))))
           #:pool pool))
  (define (observed)
    (cond
      [(sync/timeout 30 inside) (observe w)]
      [else (error 'while-inside "the worker never got inside")]))
  (with-cleanup
    (define seen (observed))
    (cond
      [(thread-dead? (worker-thread w)) seen]
      [else (semaphore-post leave) (join w) seen])
    #:finally (stop! w)))

(define (churn-once! rng)
  (define a (zeros 64 64))
  (define b (+ a 1.0))
  (define c (@ b a))
  (define total (sum c))
  (cond
    [(< (random rng) 0.5)
     (tensor-free! c)
     (tensor-free! a)]
    [else (void)])
  total)

(define (settle!)
  (for ([_ (in-range 3)])
    (reclaim-native-memory!))
  (reset-pressure-state!))

(struct snapshot (use failures))

(define (ledger-snapshot)
  (snapshot (native-memory-use) (finalizer-failures)))

(define (positive-entries totals)
  (filter (lambda (entry) (positive? (cdr entry))) totals))

(define (counter-mismatches)
  (define-values (counters fold)
    (call-as-atomic
     (lambda () (values (live-bytes-by-device) (native-memory-use/fold)))))
  (for/list ([dev (in-list (remove-duplicates (map car (append counters fold))))]
             #:unless (let ([live (dict-ref counters dev 0)])
                        (and (>= live 0) (= live (dict-ref fold dev 0)))))
    (list dev (dict-ref counters dev 0) (dict-ref fold dev 0))))

(define (ledger-violations #:baseline [base #f] #:watch [w #f])
  (define counters (positive-entries (native-memory-use)))
  (define mismatches (counter-mismatches))
  (define doubled (if w (watch-doubled w) 0))
  (define unaccounted-twice (if w (watch-unaccounted-twice w) 0))
  (filter
   values
   (list
    (and (pair? mismatches)
         (format "I1: (device counter fold) ~s" mismatches))
    (and w (> (watch-max-holders w) 1)
         (format "I2: ~a collectors held the claim at once"
                 (watch-max-holders w)))
    (and (positive? doubled)
         (format "I3: ~a handles released more than once" doubled))
    (and (positive? unaccounted-twice)
         (format "I3: ~a entries unaccounted more than once" unaccounted-twice))
    (and base (not (equal? counters (positive-entries (snapshot-use base))))
         (format "ledger ~s, not back to ~s after the drop"
                 counters (snapshot-use base)))
    (and base (> (finalizer-failures) (snapshot-failures base))
         (format "~a finalizer failures"
                 (- (finalizer-failures) (snapshot-failures base))))
    (and (native-faulted)
         (format "the fault latch is set: ~a" (native-faulted))))))

(define-check (check-ledger base w)
  (define violations (ledger-violations #:baseline base #:watch w))
  (unless (null? violations)
    (fail-check (string-join violations "; "))))

(define-check (check-known-failure issue description holds?)
  (cond
    [holds?
     (fail-check
      (format "~a looks fixed: ~a now holds, so make this an ordinary check"
              issue description))]
    [else
     (printf "known failure (~a), skipped: ~a\n" issue description)]))
