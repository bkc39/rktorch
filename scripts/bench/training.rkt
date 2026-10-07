#lang racket/base

(require (only-in json read-json write-json)
         (only-in racket/list drop)
         (only-in "harness.rkt" summarize))

(provide record-steps!
         read-steps
         analyze-steps)

(define (drain receiver)
  (let loop ([acc '()])
    (define event (sync/timeout 0 receiver))
    (if event
        (loop (cons (vector-ref event 2) acc))
        (reverse acc))))

(define (record-steps! runner out)
  (define receiver (make-log-receiver (current-logger) 'debug 'rktorch-step))
  (define t0 (current-inexact-monotonic-milliseconds))
  (dynamic-require `(submod (file ,runner) main) #f)
  (define wall (- (current-inexact-monotonic-milliseconds) t0))
  (define steps
    (for/list ([e (in-list (drain receiver))])
      (list (vector-ref e 0) (vector-ref e 1) (vector-ref e 2))))
  (call-with-output-file out #:exists 'truncate
    (lambda (port)
      (write-json (hasheq 'wall_ms wall 'steps steps) port))))

(define (read-steps path)
  (define data (call-with-input-file path read-json))
  (values (for/list ([s (in-list (hash-ref data 'steps))]) (list->vector s))
          (hash-ref data 'wall_ms)))

(define (differences xs)
  (for/list ([a (in-list xs)] [b (in-list (cdr xs))]) (- b a)))

(define (every-nth xs n)
  (for/list ([x (in-list xs)] [i (in-naturals 1)] #:when (zero? (modulo i n))) x))

(define (primary-steps events)
  (define id (and (pair? events) (vector-ref (car events) 0)))
  (for/list ([e (in-list events)] #:when (equal? (vector-ref e 0) id)) e))

(define (analyze-steps events #:epochs [epochs #f] #:warmup [warmup 0])
  (define steps (primary-steps events))
  (define total (length steps))
  (define per-epoch
    (and epochs (>= epochs 2) (zero? (remainder total epochs)) (quotient total epochs)))
  (define skip (if epochs (or per-epoch total) warmup))
  (define measured (if (< skip total) (drop steps (max 0 (sub1 skip))) '()))
  (define ms (map (lambda (e) (vector-ref e 1)) measured))
  (define gc (map (lambda (e) (vector-ref e 2)) measured))
  (define intervals (if (pair? ms) (differences ms) '()))
  (define boundaries (and per-epoch (every-nth steps per-epoch)))
  (define epoch-ms (if boundaries (differences (map (lambda (e) (vector-ref e 1)) boundaries)) '()))
  (define epoch-gc (if boundaries (differences (map (lambda (e) (vector-ref e 2)) boundaries)) '()))
  (cond
    [(null? intervals)
     (hasheq 'steps total
             'error (format "no measured steps: ~a steps, epochs ~a, warmup ~a"
                            total epochs warmup))]
    [else
     (define step-stats (summarize intervals))
     (define span (- (car (reverse ms)) (car ms)))
     (hasheq 'steps total
             'steps_per_epoch per-epoch
             'measured_steps (length intervals)
             'step_ms step-stats
             'steps_per_s (/ 1000.0 (hash-ref step-stats 'median))
             'steps_per_s_wall (/ (* 1000.0 (length intervals)) span)
             'epoch_s (map (lambda (x) (/ x 1000.0)) epoch-ms)
             'epoch_gc_ms epoch-gc
             'gc_ms (- (car (reverse gc)) (car gc)))]))

(module+ test
  (require (only-in racket/file make-temporary-file)
           rackunit)

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
    (check-= (hash-ref a 'steps_per_s_wall) (/ 6000.0 204) 1e-9)
    (check-equal? (hash-ref a 'epoch_gc_ms) '(10 10))
    (check-equal? (hash-ref a 'gc_ms) 20))

  (test-case "only the first optimizer to step is timed"
    (define events
      (for*/list ([t (in-list '(10 20 30 40))]
                  [id (in-list '(1 2))])
        (vector id (exact->inexact (+ t id)) 0)))
    (define a (analyze-steps events #:epochs 2))
    (check-equal? (hash-ref a 'steps) 4)
    (check-equal? (hash-ref a 'epoch_s) '(0.02)))

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
    (define path (make-temporary-file "steps-~a.json"))
    (call-with-output-file path #:exists 'truncate
      (lambda (port) (write-json (hasheq 'wall_ms 5.0 'steps '((1 2.0 3))) port)))
    (define-values (events wall) (read-steps path))
    (delete-file path)
    (check-equal? events (list (vector 1 2.0 3)))
    (check-equal? wall 5.0)))
