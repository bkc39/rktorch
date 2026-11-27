#lang racket/base

(require (only-in ffi/unsafe _fun _int _long _ptr define-cstruct get-ffi-obj)
         (only-in racket/file file->string)
         (only-in racket/list last)
         (only-in racket/match match match-lambda)
         (only-in racket/os getpid)
         (only-in racket/string string-split)
         (only-in threading ~>)
         (only-in torch/tests/private/platform host-platform)
         (only-in (submod "threads.rkt" private) [task-dir tasks]))

(provide (struct-out gc-event)
         (struct-out span)
         call-with-gc-log
         gc-overlap-ms
         os-thread-id
         record-span!
         spans
         task-cpu-ms
         thread-cpu-ms
         timed)

(define-cstruct _timespec ([sec _long] [nsec _long]))

(define thread-clock (match host-platform ['linux 3] ['macosx 16] [_ #f]))

(define clock-gettime
  (get-ffi-obj "clock_gettime" #f (_fun _int (ts : (_ptr o _timespec)) -> (rc : _int) -> (values rc ts))
               (lambda () #f)))

(define gettid
  (and (eq? host-platform 'linux) (get-ffi-obj "gettid" #f (_fun -> _int) (lambda () #f))))

(define (thread-cpu-ms)
  (cond
    [(and thread-clock clock-gettime)
     (define-values (rc ts) (clock-gettime thread-clock))
     (and (zero? rc) (+ (* 1000.0 (timespec-sec ts)) (/ (timespec-nsec ts) 1e6)))]
    [else #f]))

(define (os-thread-id)
  (if gettid (gettid) (getpid)))

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

(define (call-with-gc-log thunk #:live [live (lambda () #f)])
  (define receiver (make-log-receiver (current-logger) 'debug 'GC))
  (define events (box '()))
  (define (take! v)
    (match v
      [(vector _ _ (? prefab-struct-key info) _)
       (set-box! events (cons (gc-info->event info (live)) (unbox events)))]
      [_ (void)]))
  (define listener
    (thread (lambda () (let loop () (take! (sync receiver)) (loop)))))
  (define result (thunk))
  (define stop (current-inexact-milliseconds))
  (kill-thread listener)
  (let drain ()
    (match (sync/timeout 0.2 receiver)
      [#f (void)]
      [v (take! v) (drain)]))
  (values result
          (filter (lambda (e) (< (gc-event-start e) stop)) (reverse (unbox events)))))

(define (gc-overlap-ms events w₀ w₁)
  (for/sum ([e (in-list events)])
    (max 0.0 (- (min w₁ (gc-event-end e)) (max w₀ (gc-event-start e))))))

(define (task-cpu-ms)
  (define user-hz-ms 10.0)
  (for/hash ([tid (in-list (if (directory-exists? tasks) (directory-list tasks) '()))])
    (define fields
      (with-handlers ([exn:fail:filesystem? (lambda (_e) #f)])
        (~> (build-path tasks tid "stat") file->string (regexp-match #rx"\\) (.*)$" _))))
    (values (string->number (path->string tid))
            (match fields
              [(list _ rest)
               (define ticks (map string->number (string-split rest)))
               (* user-hz-ms (+ (list-ref ticks 11) (list-ref ticks 12)))]
              [_ 0.0]))))

(module+ test
  (require rackunit)

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
      (call-with-gc-log (lambda () (collect-garbage 'major) 'done) #:live (lambda () 7)))
    (check-equal? v 'done)
    (check-true (pair? events))
    (match (last events)
      [(gc-event mode _ _ cpu start end live)
       (check-not-false (memq mode '(major minor incremental)))
       (check-true (<= start end))
       (check-true (>= cpu 0))
       (check-equal? live 7)]))

  (test-case "overlap counts only the part of a collection inside the span"
    (define events (list (gc-event 'minor 0 0 1 10.0 20.0 #f) (gc-event 'major 0 0 1 30.0 40.0 #f)))
    (check-equal? (gc-overlap-ms events 15.0 35.0) 10.0)
    (check-equal? (gc-overlap-ms events 0.0 5.0) 0.0))

  (test-case "this process's tasks read with their CPU, main among them where /proc exists"
    (define cpu (task-cpu-ms))
    (when (directory-exists? tasks)
      (check-true (hash-has-key? cpu (getpid))))))
