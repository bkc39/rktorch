#lang racket/base

(require (for-syntax racket/base syntax/parse)
         (only-in ffi/unsafe register-finalizer)
         (only-in ffi/unsafe/atomic
                  make-uninterruptible-lock uninterruptible-lock-acquire
                  uninterruptible-lock-release)
         (only-in ffi/unsafe/custodian make-custodian-at-root)
         (only-in ffi/unsafe/os-thread
                  make-os-semaphore os-semaphore-post os-semaphore-wait)
         (only-in ffi/unsafe/vm vm-eval)
         (only-in racket/contract/base
                  -> ->* and/c any any/c flat-named-contract not/c)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../foreign/raw/cas.rkt" cas-add! cas-claim! cas-update!)
         (only-in "../foreign/raw/pressure-settings.rkt" reaper-count-from-env)
         (only-in "../foreign/raw/race-points.rkt" race-point)
         (only-in "../foreign/raw/regions.rkt" with-cleanup with-uninterruptible)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "main-thread.rkt" main-place? on-main-os-thread?))

(provide drain-deadline
         reap-after-collection!
         reap-when-dead!
         reaper-count
         wake-reapers!)

(module+ private
  (provide call-with-canary-off
           heap-object/c
           reaper-statistics
           reaper-threads
           with-take-lock))

(define heap-object/c
  (flat-named-contract 'heap-object/c
    (lambda (v) (not (or (fixnum? v) (char? v) (boolean? v) (null? v)
                         (void? v) (eof-object? v))))))

(struct reaped (name release after-batch guardian register) ;; noqa
  #:constructor-name make-reaped
  #:omit-define-syntaxes)

(struct reaper-fence (ticket))

(struct reaper (wake parked [thread #:mutable]))

(struct slots (kinds objects))

(struct help (fences [after #:mutable] [fenced? #:mutable]))

(define batch-size 32)

(define drain-deadline (make-parameter 2000))

(define reaper-count (reaper-count-from-env))

(define kinds (box '()))

(define take-lock (make-uninterruptible-lock))
(define tickets (box 0))
(define in-flight (box '()))

(define taken (box 0))
(define released (box 0))
(define parks (box 0))
(define wakes (box 0))
(define escapes (box 0))

(define canary-wakes? (box #t))

(define-syntax-parse-rule (with-take-lock body:expr ...+)
  (begin
    (uninterruptible-lock-acquire take-lock)
    (with-cleanup body ... #:finally (uninterruptible-lock-release take-lock))))

(define/contract-out (reaped name release #:after-batch [after-batch void]) ;; noqa
  (->* [symbol? (-> any/c any)] [#:after-batch (-> any)] reaped?)
  (cond
    [(main-place?)
     (define guardian (vm-eval '(make-guardian #t)))
     (define kind (make-reaped name release after-batch guardian guardian))
     (cas-update! kinds (lambda (known) (append known (list kind))))
     kind]
    [else
     (define finalize (finalizing release after-batch))
     (make-reaped name release after-batch #f
                  (lambda (v) (register-finalizer v finalize)))]))

(define (reap-when-dead! kind v)
  ((reaped-register kind) v))

(define (fresh-slots)
  (slots (make-vector batch-size #f) (make-vector batch-size #f)))

(define (take-into! s)
  (define kinds-of (slots-kinds s))
  (define objects (slots-objects s))
  (let loop ([known (unbox kinds)] [n 0])
    (cond
      [(or (null? known) (= n batch-size)) n]
      [((reaped-guardian (car known)))
       => (lambda (v)
            (when (reaper-fence? v)
              (set-box! (reaper-fence-ticket v) (+ (unbox tickets) n)))
            (vector-set! kinds-of n (car known))
            (vector-set! objects n v)
            (loop known (add1 n)))]
      [else (loop (cdr known) n)])))

(define (note-escape! _e)
  (cas-add! escapes 1))

(define (release-slot! s i)
  (define kind (vector-ref (slots-kinds s) i))
  (define v (vector-ref (slots-objects s) i))
  (vector-set! (slots-kinds s) i #f)
  (vector-set! (slots-objects s) i #f)
  (cond
    [(reaper-fence? v) 0]
    [else
     (with-handlers ([(lambda (_) #t) note-escape!])
       ((reaped-release kind) v))
     1]))

(define ((finalizing release after-batch) v)
  (with-uninterruptible
    (with-handlers ([(lambda (_) #t) note-escape!])
      (release v))
    (with-handlers ([(lambda (_) #t) note-escape!])
      (after-batch)))
  (cas-add! taken 1)
  (cas-add! released 1))

;; Each kind's hook runs once per batch it appears in, inside the batch's
;; region, so a drain that sees the batch done sees what the hook did.
(define (release-batch! s n)
  (define afters
    (for/fold ([afters '()]) ([kind (in-vector (slots-kinds s) 0 n)])
      (define after (reaped-after-batch kind))
      (if (memq after afters) afters (cons after afters))))
  (define objects
    (for/sum ([i (in-range n)])
      (release-slot! s i)))
  (for ([after (in-list afters)])
    (with-handlers ([(lambda (_) #t) note-escape!])
      (after)))
  (cas-add! taken objects)
  (cas-add! released objects))

(define (take-batch! s h)
  (with-take-lock
    (define first (unbox tickets))
    (define n (take-into! s))
    (define after (+ first n))
    (set-box! tickets after)
    (when h
      (set-help-after! h after)
      (set-help-fenced?! h (andmap unbox (help-fences h))))
    (when (positive? n)
      (cas-update! in-flight (lambda (firsts) (cons first firsts)) #:site 'reaper))
    (values first n)))

(define (reap-batch! s [h #f])
  (with-uninterruptible
    (define-values (first n) (take-batch! s h))
    (when (positive? n)
      (with-cleanup
        (race-point reaper-taken (current-thread))
        (release-batch! s n)
        #:finally (cas-update! in-flight (lambda (firsts) (remv first firsts)) #:site 'reaper)))
    n))

(define (wake-one!)
  (for/or ([r (in-list reapers)])
    (and (cas-claim! (reaper-parked r) #t #f #:site 'reaper)
         (begin (os-semaphore-post (reaper-wake r)) #t))))

(define (park! r s)
  (define parked (reaper-parked r))
  (cas-claim! parked #f #t #:site 'reaper)
  (cond
    [(positive? (reap-batch! s)) (void (cas-claim! parked #t #f #:site 'reaper))]
    [else
     (race-point reaper-parking (current-thread))
     (cas-add! parks 1)
     (os-semaphore-wait (reaper-wake r))
     (race-point reaper-woken (current-thread))
     (set-box! parked #f)
     (cas-add! wakes 1)]))

(define (reap! r)
  (define s (fresh-slots))
  (let loop ()
    (define n (reap-batch! s))
    (cond
      [(= n batch-size) (wake-one!)]
      [(zero? n) (park! r s)]
      [else (void)])
    (loop)))

(define (wake-reapers!)
  (void (wake-one!)))

(define (reap-after-collection!)
  (unless (null? reapers)
    (reap-batch! (fresh-slots))
    (wake-one!))
  (void))

(define (plant-fence! kind)
  (define ticket (box #f))
  (reap-when-dead! kind (reaper-fence ticket))
  ticket)

(define (help! s fences deadline)
  (define h (help fences #f #f))
  (let loop ()
    (cond
      [(>= (current-inexact-milliseconds) deadline) #f]
      [else
       (define n (reap-batch! s h))
       (if (or (help-fenced? h) (< n batch-size)) (help-after h) (loop))])))

(define (await-batches d₀ deadline)
  (let loop ()
    (cond
      [(andmap (lambda (first) (>= first d₀)) (unbox in-flight)) #t]
      [(>= (current-inexact-milliseconds) deadline) #f]
      [else
       (when (on-main-os-thread?)
         (sleep 0))
       (loop)])))

(define (drain-fenced! deadline)
  (wake-reapers!)
  (define fences (map plant-fence! (unbox kinds)))
  (define d₀ (help! (fresh-slots) fences deadline))
  (cond
    [d₀
     (race-point drain-fenced d₀)
     (await-batches d₀ deadline)]
    [else #f]))

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

(define/contract-out (drain-reapers! #:deadline-ms [ms (drain-deadline)])
  (->* [] [#:deadline-ms (and/c real? (not/c negative?))] boolean?)
  (define deadline (+ (current-inexact-milliseconds) ms))
  (if (main-place?)
      (drain-fenced! deadline)
      (settle-finalizers! deadline)))

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
  (register-finalizer (box 0)
                      (lambda (_)
                        (when (unbox canary-wakes?)
                          (wake-reapers!))
                        (arm-canary!))))

(define (call-with-canary-off thunk)
  (set-box! canary-wakes? #f)
  (with-cleanup (thunk) #:finally (set-box! canary-wakes? #t)))

(define reapers
  (if (main-place?)
      (for/list ([_ (in-range reaper-count)])
        (reaper (make-os-semaphore) (box #f) #f))
      '()))

(cond
  [(null? reapers) (register-finalizer (box 0) void)]
  [else
   (for-each start-reaper! reapers)
   (arm-canary!)])

(define (reaper-threads)
  (map reaper-thread reapers))

(define (reaper-statistics)
  (hash 'taken (unbox taken)
        'released (unbox released)
        'parks (unbox parks)
        'parked (for/sum ([r (in-list reapers)]) (if (unbox (reaper-parked r)) 1 0))
        'wakes (unbox wakes)
        'escapes (unbox escapes)))
