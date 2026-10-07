#lang racket/base

(require (for-syntax racket/base)
         (only-in ffi/unsafe/atomic call-as-atomic in-atomic-mode?)
         (only-in racket/dict dict-ref)
         (only-in racket/list remove-duplicates)
         (only-in racket/string string-join)
         (only-in rackunit define-check fail-check)
         ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define
         (only-in "../../foreign.rkt"
                  add finalizer-failures matmul native-memory-use
                  reclaim-native-memory! sum zeros)
         (only-in (submod "../../foreign.rkt" unsafe) tensor-free!)
         (only-in "../../foreign/raw/fault.rkt" native-faulted)
         (only-in "../../foreign/raw/memory.rkt" native-memory-use/fold)
         (only-in "../../foreign/raw/pressure.rkt"
                  live-bytes-by-device reset-pressure-state!)
         (only-in "../../foreign/raw/race-points.rkt" race-hook))

(provide with-race-hook
         hooks
         labelled
         make-gate
         gate-hook
         wait-for-arrival
         release-gate!
         make-watch
         watch-hook
         watch-max-holders
         watch-release-count
         watch-released
         watch-finalized
         fuzz-hook
         spawn
         join
         stop!
         worker-thread
         join-all
         settle!
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

(define ((hooks . all) name subject)
  (for ([h (in-list all)])
    (h name subject)))

(define thread-label (make-thread-cell #f))

(define (current-label)
  (thread-cell-ref thread-label))

(define ((labelled label) _subject)
  (eq? (current-label) label))

(struct gate (point admits? arrived resume [state #:mutable]))

(define (make-gate point #:when [admits? (lambda (_subject) #t)])
  (gate point admits? (make-semaphore 0) (make-semaphore 0) 'waiting))

(define (take-gate! g)
  (call-as-atomic
   (lambda ()
     (and (eq? (gate-state g) 'waiting)
          (set-gate-state! g 'taken)
          #t))))

(define ((gate-hook g) name subject)
  (define taken?
    (and (eq? name (gate-point g))
         ((gate-admits? g) subject)
         (take-gate! g)))
  (cond
    [(not taken?) (void)]
    [(in-atomic-mode?)
     (set-gate-state! g 'atomic)
     (semaphore-post (gate-arrived g))]
    [else
     (set-gate-state! g 'parked)
     (semaphore-post (gate-arrived g))
     (semaphore-wait (gate-resume g))]))

(define (wait-for-arrival g #:seconds [seconds 30])
  (and (sync/timeout seconds (gate-arrived g))
       (gate-state g)))

(define (release-gate! g)
  (semaphore-post (gate-resume g)))

(struct watch (holders
               [max-holders #:mutable]
               releases
               [released #:mutable]
               [finalized #:mutable]
               [doubled #:mutable]
               unaccounts
               [unaccounted-twice #:mutable]))

(define (make-watch)
  (watch (make-hasheq) 0 (make-weak-hasheq) 0 0 0 (make-weak-hasheq) 0))

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
  (call-as-atomic (lambda () (hash-ref (watch-releases w) handle 0))))

(define (note-holder! w holder)
  (define holders (watch-holders w))
  (for ([t (in-list (hash-keys holders))] #:when (thread-dead? t))
    (hash-remove! holders t))
  (hash-set! holders holder #t)
  (set-watch-max-holders! w (max (watch-max-holders w) (hash-count holders))))

(define ((watch-hook w) name subject)
  (case name
    [(collector-claimed)
     (call-as-atomic (lambda () (note-holder! w subject)))]
    [(collector-releasing)
     (call-as-atomic (lambda () (hash-remove! (watch-holders w) subject)))]
    [(free-unaccounted)
     (call-as-atomic (lambda () (note-release! w subject)))]
    [(unaccount-entry-read)
     (call-as-atomic (lambda () (note-unaccount! w subject)))]
    [(finalizer-releasing)
     (call-as-atomic
      (lambda ()
        (set-watch-finalized! w (add1 (watch-finalized w)))
        (note-release! w subject)))]
    [else (void)]))

(define (seeded-generator seed label)
  (define g (make-pseudo-random-generator))
  (parameterize ([current-pseudo-random-generator g])
    (random-seed (modulo (+ (* 1000003 seed) (equal-hash-code label))
                         2147483647)))
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
      (perturb! (random (generator))))))

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
  (define outcome (unbox (worker-outcome w)))
  (cond
    [(not outcome)
     (kill-thread (worker-thread w))
     (error 'join "a worker did not finish within ~a s, or was killed" seconds)]
    [(eq? (car outcome) 'raised) (raise (cdr outcome))]
    [else (cdr outcome)]))

(define (stop! w)
  (kill-thread (worker-thread w)))

(define (join-all workers #:seconds [seconds 120])
  (for/list ([w (in-list workers)])
    (join w #:seconds seconds)))

(define (churn-once! rng)
  (define a (zeros 64 64))
  (define b (add a 1.0))
  (define c (matmul b a))
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
  (define counters (live-bytes-by-device))
  (define fold (native-memory-use/fold))
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
