#lang racket/base

(require (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic
                  make-uninterruptible-lock uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in ffi/unsafe/os-thread
                  make-os-semaphore os-semaphore-post os-semaphore-wait)
         (only-in ffi/unsafe/vm vm-eval)
         (only-in racket/contract/base
                  -> ->* and/c any any/c flat-named-contract not/c)
         (only-in "../foreign/raw/cas.rkt" cas-add! cas-claim! cas-update!)
         (only-in "../foreign/raw/pressure-settings.rkt" reaper-count-from-env)
         (only-in "../foreign/raw/race-points.rkt" race-point)
         (only-in "../foreign/raw/regions.rkt" with-uninterruptible)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "main-thread.rkt" main-place? on-main-os-thread?))

(provide drain-deadline
         reap-when-dead!
         reaper-count)

(module+ private
  (provide heap-object/c
           reaper-statistics
           reaper-threads))

(define heap-object/c
  (flat-named-contract 'heap-object/c
    (lambda (v) (not (or (fixnum? v) (char? v) (boolean? v) (null? v)
                         (void? v) (eof-object? v))))))

(struct reaped (name release guardian register) ;; noqa
  #:constructor-name reaped/guardian
  #:omit-define-syntaxes)

(struct reaper-fence (ticket))

(struct reaper (held wake parked [thread #:mutable]))

(define batch-size 32)

(define drain-deadline (make-parameter 2000))

(define reaper-count (reaper-count-from-env))

(define kinds (box '()))

;; The guardian's take is an unsynchronized read-modify-write, so every
;; take holds this lock, and nothing else is ever taken while holding it.
(define take-lock (make-uninterruptible-lock))
(define tickets (box 0))

(define taken (box 0))
(define released (box 0))
(define parks (box 0))
(define wakes (box 0))
(define escapes (box 0))

(define/contract-out (reaped name release) ;; noqa
  (-> symbol? (-> any/c any) reaped?)
  (cond
    [(main-place?)
     (define guardian (vm-eval '(make-guardian #t)))
     (define kind (reaped/guardian name release guardian guardian))
     (cas-update! kinds (lambda (known) (append known (list kind))))
     kind]
    [else
     (define finalize (finalizing release))
     (reaped/guardian name release #f (lambda (v) (register-finalizer v finalize)))]))

(define (reap-when-dead! kind v)
  ((reaped-register kind) v))

(define (take-into! slots)
  (let loop ([known (unbox kinds)] [n 0])
    (cond
      [(or (null? known) (= n batch-size)) n]
      [((reaped-guardian (car known)))
       => (lambda (v)
            (when (reaper-fence? v)
              (set-box! (reaper-fence-ticket v) (+ (unbox tickets) n)))
            (vector-set! slots (* 2 n) (car known))
            (vector-set! slots (add1 (* 2 n)) v)
            (loop known (add1 n)))]
      [else (loop (cdr known) n)])))

(define (note-escape! _e)
  (cas-add! escapes 1))

(define (release-slot! slots i)
  (define kind (vector-ref slots (* 2 i)))
  (define v (vector-ref slots (add1 (* 2 i))))
  (vector-set! slots (* 2 i) #f)
  (vector-set! slots (add1 (* 2 i)) #f)
  (cond
    [(reaper-fence? v) 0]
    [else
     (with-handlers ([(lambda (_) #t) note-escape!])
       ((reaped-release kind) v))
     1]))

(define ((finalizing release) v)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) note-escape!])
      (release v)))
  (cas-add! taken 1)
  (cas-add! released 1))

(define (release-batch! slots n)
  (define objects
    (for/sum ([i (in-range n)])
      (release-slot! slots i)))
  (cas-add! taken objects)
  (cas-add! released objects))

;; Taking and releasing are one region, so a kill waits for the whole batch.
(define (reap-batch! slots held fences)
  (with-uninterruptible
    (uninterruptible-lock-acquire take-lock)
    (define first (unbox tickets))
    (define n (take-into! slots))
    (set-box! tickets (+ first n))
    (when (and held (positive? n))
      (cas-claim! held #f first #:site 'reaper))
    (define fenced? (and fences (andmap unbox fences)))
    (uninterruptible-lock-release take-lock)
    (when (positive? n)
      (race-point reaper-taken (current-thread))
      (release-batch! slots n)
      (when held
        (cas-claim! held first #f #:site 'reaper)))
    (values n (+ first n) fenced?)))

;; Each reaper parks on its own semaphore: Racket's OS semaphore signals
;; only when its count was 0, so two posts to one shared semaphore can wake
;; one of two waiters. A waker claims a parked reaper before posting it.
(define (wake-one!)
  (for/or ([r (in-list reapers)])
    (and (cas-claim! (reaper-parked r) #t #f #:site 'reaper)
         (begin (os-semaphore-post (reaper-wake r)) #t))))

(define (park! r slots)
  (define parked (reaper-parked r))
  (cas-claim! parked #f #t #:site 'reaper)
  (define-values (n _d _fenced?) (reap-batch! slots (reaper-held r) #f))
  (cond
    [(positive? n) (void (cas-claim! parked #t #f #:site 'reaper))]
    [else
     (race-point reaper-parking (current-thread))
     (cas-add! parks 1)
     (os-semaphore-wait (reaper-wake r))
     (race-point reaper-woken (current-thread))
     (set-box! parked #f)
     (cas-add! wakes 1)]))

(define (reap! r)
  (define slots (make-vector (* 2 batch-size) #f))
  (let loop ()
    (define-values (n _d _fenced?) (reap-batch! slots (reaper-held r) #f))
    (cond
      [(= n batch-size) (wake-one!)]
      [(zero? n) (park! r slots)]
      [else (void)])
    (loop)))

(define/contract-out (wake-reapers!) (-> void?)
  (void (wake-one!)))

;; The thread first to account after a collection releases one batch of
;; what it found before it wakes a reaper: a reaper woken first can take the
;; batch and then lose its CPU while holding it.
(define/contract-out (reap-after-collection!) (-> void?)
  (unless (null? reapers)
    (define-values (_n _d _fenced?) (reap-batch! (make-vector (* 2 batch-size) #f) #f #f))
    (wake-one!))
  (void))

(define (plant-fence! kind)
  (define ticket (box #f))
  (reap-when-dead! kind (reaper-fence ticket))
  ticket)

(define (help! slots fences deadline)
  (let loop ()
    (cond
      [(>= (current-inexact-milliseconds) deadline) #f]
      [else
       (define-values (n after fenced?) (reap-batch! slots #f fences))
       (if (or fenced? (< n batch-size)) after (loop))])))

(define (settled? r d0)
  (define ticket (unbox (reaper-held r)))
  (or (not ticket) (>= ticket d0) (thread-dead? (reaper-thread r))))

(define (await-batches d0 deadline)
  (let loop ()
    (cond
      [(andmap (lambda (r) (settled? r d0)) reapers) #t]
      [(>= (current-inexact-milliseconds) deadline) #f]
      [else
       (when (on-main-os-thread?)
         (sleep 0))
       (loop)])))

(define (drain-fenced! deadline)
  (wake-reapers!)
  (define fences (map plant-fence! (unbox kinds)))
  (define d0 (help! (make-vector (* 2 batch-size) #f) fences deadline))
  (cond
    [d0
     (race-point drain-fenced d0)
     (await-batches d0 deadline)]
    [else #f]))

;; Outside the main place, releases run on ffi's finalizer thread, which
;; runs only while this one yields: so yield until the release count has
;; stood still for a few turns.
(define quiet-turns 3)

(define (settle-finalizers! deadline)
  (let loop ([seen (unbox released)] [quiet 0])
    (sleep 0)
    (define now (unbox released))
    (cond
      [(>= (current-inexact-milliseconds) deadline) #f]
      [(not (= now seen)) (loop now 0)]
      [(< (add1 quiet) quiet-turns) (loop now (add1 quiet))]
      [else #t])))

(define/contract-out (drain-reapers! #:deadline-ms [ms (drain-deadline)]) ;; noqa
  (->* [] [#:deadline-ms (and/c real? (not/c negative?))] boolean?)
  (define deadline (+ (current-inexact-milliseconds) ms))
  (if (main-place?)
      (drain-fenced! deadline)
      (settle-finalizers! deadline)))

;; A will executor is polled by the finalizer thread, not filled by the
;; collection, so outside the main place a canary shows that thread has
;; started on this collection's batch before the release count settles.
(define/contract-out (collect-and-drain-reapers!) ;; noqa
  (-> boolean?)
  (cond
    [(main-place?)
     (collect-garbage)
     (drain-reapers!)]
    [else
     (define canary-finalized (make-semaphore 0))
     (register-finalizer (box 0) (lambda (_) (semaphore-post canary-finalized)))
     (collect-garbage)
     (sync/timeout 0.5 canary-finalized)
     (drain-reapers!)]))

(define (start-reaper! r)
  (parameterize ([current-custodian (make-custodian-at-root)])
    (set-reaper-thread! r (thread (lambda () (reap! r)) #:pool 'own))))

(define (arm-canary!)
  (register-finalizer (box 0) (lambda (_) (wake-reapers!) (arm-canary!))))

;; Reapers run only in the main place, and nothing stops them: a process
;; exits cleanly with its reapers parked, but an `'own` thread that wakes
;; from an OS wait while its place is torn down crashes the whole process.
(define reapers
  (if (main-place?)
      (for/list ([_ (in-range reaper-count)])
        (reaper (box #f) (make-os-semaphore) (box #f) #f))
      '()))

(unless (null? reapers)
  (for-each start-reaper! reapers)
  (arm-canary!))

(define (reaper-threads)
  (map reaper-thread reapers))

(define (reaper-statistics)
  (hash 'taken (unbox taken)
        'released (unbox released)
        'parks (unbox parks)
        'parked (for/sum ([r (in-list reapers)]) (if (unbox (reaper-parked r)) 1 0))
        'wakes (unbox wakes)
        'escapes (unbox escapes)))
