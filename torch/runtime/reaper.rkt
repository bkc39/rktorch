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
         (only-in "main-thread.rkt" on-main-os-thread?))

(provide drain-deadline
         reap-when-dead!
         reaper-count)

(module+ private
  (provide heap-object/c
           reaper-statistics
           reaper-threads
           stop-reapers!))

(define heap-object/c
  (flat-named-contract 'heap-object/c
    (lambda (v) (not (or (fixnum? v) (char? v) (boolean? v) (null? v)
                         (void? v) (eof-object? v))))))

(struct reaped (name release guardian)
  #:constructor-name reaped/guardian
  #:omit-define-syntaxes)

(struct reaper-fence (ticket))

(struct reaper (thread held))

(define batch-size 32)

(define drain-deadline (make-parameter 2000))

(define reaper-count (reaper-count-from-env))

(define kinds (box '()))

;; The guardian's take is an unsynchronized read-modify-write, so every
;; take holds this lock, and nothing else is ever taken while holding it.
(define take-lock (make-uninterruptible-lock))
(define tickets (box 0))

(define parked (box 0))
(define park (make-os-semaphore))
(define stopping (box #f))

(define taken (box 0))
(define released (box 0))
(define parks (box 0))
(define wakes (box 0))
(define escapes (box 0))

(define/contract-out (reaped name release)
  (-> symbol? (-> any/c any) reaped?)
  (define kind (reaped/guardian name release (vm-eval '(make-guardian #t))))
  (cas-update! kinds (lambda (known) (append known (list kind))))
  kind)

(define (reap-when-dead! kind v)
  ((reaped-guardian kind) v))

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

(define (wake-one!)
  (os-semaphore-post park))

(define (park! slots held)
  (cas-add! parked 1)
  (define-values (n _d _fenced?) (reap-batch! slots held #f))
  (when (zero? n)
    (race-point reaper-parking (current-thread))
    (cas-add! parks 1)
    (os-semaphore-wait park)
    (cas-add! wakes 1))
  (cas-add! parked -1))

(define (reap! held)
  (define slots (make-vector (* 2 batch-size) #f))
  (let loop ()
    (unless (unbox stopping)
      (define-values (n _d _fenced?) (reap-batch! slots held #f))
      (cond
        [(= n batch-size)
         (when (positive? (unbox parked))
           (wake-one!))]
        [(zero? n) (park! slots held)]
        [else (void)])
      (loop))))

(define/contract-out (wake-reapers!) (-> void?)
  (when (positive? (unbox parked))
    (wake-one!)))

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

(define/contract-out (drain-reapers! #:deadline-ms [ms (drain-deadline)])
  (->* () (#:deadline-ms (and/c real? (not/c negative?))) boolean?)
  (define deadline (+ (current-inexact-milliseconds) ms))
  (wake-reapers!)
  (define fences (map plant-fence! (unbox kinds)))
  (define d0 (help! (make-vector (* 2 batch-size) #f) fences deadline))
  (cond
    [d0
     (race-point drain-fenced d0)
     (await-batches d0 deadline)]
    [else #f]))

(define (start-reapers! k)
  (parameterize ([current-custodian (make-custodian-at-root)])
    (for/list ([_ (in-range k)])
      (define held (box #f))
      (reaper (thread (lambda () (reap! held)) #:pool 'own) held))))

(define (arm-canary!)
  (register-finalizer (box 0) (lambda (_) (wake-reapers!) (arm-canary!))))

(define (stop-reapers!)
  (set-box! stopping #t)
  (for ([_ (in-list reapers)])
    (wake-one!)))

(define reapers (start-reapers! reaper-count))

(arm-canary!)

(void (plumber-add-flush! (current-plumber) (lambda (_h) (stop-reapers!))))

(define (reaper-threads)
  (map reaper-thread reapers))

(define (reaper-statistics)
  (hash 'taken (unbox taken)
        'released (unbox released)
        'parks (unbox parks)
        'wakes (unbox wakes)
        'escapes (unbox escapes)))
