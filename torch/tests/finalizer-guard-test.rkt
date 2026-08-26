#lang racket/base

(module+ test
  (require rackunit
           (only-in ffi/unsafe malloc free)
           (only-in racket/port port->string)
           (only-in ffi/unsafe/alloc allocator deallocator)
           (only-in "../foreign.rkt" add ones tensor-shape)
           (only-in (submod "../foreign.rkt" unsafe) tensor-free!)
           (only-in "../foreign/raw/memory.rkt"
                    finalizer-diagnostics finalizer-failures
                    swallow-and-count-failure))

  ;; The exit dump only happens at plumber flush, so it needs a child process.
  (define (trace-stderr #:trace? trace?)
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (if trace?
          (putenv "RKTORCH_MEM_TRACE" "1")
          (environment-variables-set! (current-environment-variables)
                                      #"RKTORCH_MEM_TRACE" #f))
      (define-values (sp _o _i err)
        (subprocess #f #f #f (find-system-path 'exec-file)
                    "-e" "(require torch) (void (ones 2 2))"))
      (subprocess-wait sp)
      (begin0 (port->string err) (close-input-port err))))

  (define (collect-until ready? #:tries [tries 50])
    (let loop ([i 0])
      (collect-garbage)
      (unless (or (ready?) (>= i tries))
        (sleep 0.01)
        (loop (add1 i)))))

  (test-case "swallow-and-count-failure swallows exn:fail from the release"
    (define guarded
      (swallow-and-count-failure (lambda (t) (error 'release "boom: ~a" t))))
    (check-equal? (guarded 'handle) (void)))

  (test-case "swallow-and-count-failure swallows are counted, not invisible (#51)"
    (define before (finalizer-failures))
    ((swallow-and-count-failure (lambda (_t) (error 'boom "swallowed"))) 'handle)
    ((swallow-and-count-failure (lambda (_t) (raise 'bare))) 'handle)
    (check-equal? (finalizer-failures) (+ before 2))
    ;; successful releases don't count
    ((swallow-and-count-failure void) 'handle)
    (check-equal? (finalizer-failures) (+ before 2)))

  (test-case "RKTORCH_MEM_TRACE dumps the diagnostics at exit"
    (define out (trace-stderr #:trace? #t))
    (check-true (regexp-match? #rx"\\[rktorch mem\\]" out)
                (format "no trace line on stderr; got: ~s" out))
    (check-true (regexp-match? #rx"\\(failures \\. [0-9]+\\)" out)
                "the trace line carries no failures count"))

  (test-case "the exit dump is off unless RKTORCH_MEM_TRACE is set"
    (check-false (regexp-match? #rx"\\[rktorch mem\\]" (trace-stderr #:trace? #f))))

  (test-case "finalizer-diagnostics reports runs alongside failures"
    (define (runs-of d) (cdr (assq 'runs d)))
    (define (failures-of d) (cdr (assq 'failures d)))
    (define before (finalizer-diagnostics))
    ;; every guarded release counts as a run, whether or not it fails
    ((swallow-and-count-failure void) 'handle)
    ((swallow-and-count-failure (lambda (_t) (error 'boom "counted"))) 'handle)
    (define after (finalizer-diagnostics))
    (check-equal? (runs-of after) (+ (runs-of before) 2))
    (check-equal? (failures-of after) (+ (failures-of before) 1))
    ;; the accessor agrees with the standalone counter
    (check-equal? (failures-of after) (finalizer-failures)))

  (test-case "finalizer-diagnostics captures the exception text, not just a count"
    (define (messages-of d) (cdr (assq 'messages d)))
    (define before (length (messages-of (finalizer-diagnostics))))
    ((swallow-and-count-failure
      (lambda (_t) (error 'release "distinctive-marker-9f3a"))) 'handle)
    (define msgs (messages-of (finalizer-diagnostics)))
    (check-true (list? msgs))
    (check-true (andmap string? msgs))
    ;; bounded at 8 (capture-limit): once full, later failures add no messages
    (cond
      [(< before 8)
       (check-equal? (length msgs) (add1 before))
       (check-true (regexp-match? #rx"distinctive-marker-9f3a" (car (reverse msgs))))]
      [else (check-equal? (length msgs) 8)]))

  (test-case "message capture is bounded at 8 and keeps the MOST RECENT"
    ;; Keeping the first 8 would mean a real cascade late in a session leaves no
    ;; evidence, because early benign failures already filled the buffer.
    (for ([i (in-range 30)])
      ((swallow-and-count-failure (lambda (_t) (error 'flood "n=~a" i))) 'handle))
    ((swallow-and-count-failure (lambda (_t) (error 'flood "newest-marker-7c1e"))) 'handle)
    (define msgs (cdr (assq 'messages (finalizer-diagnostics))))
    (check-true (<= (length msgs) 8)
                (format "capture-limit exceeded: ~a messages" (length msgs)))
    (check-true (for/or ([m (in-list msgs)]) (regexp-match? #rx"newest-marker-7c1e" m))
                "the most recent failure was dropped once the buffer filled")
    (check-false (for/or ([m (in-list msgs)]) (regexp-match? #rx"n=0$" m))
                 "the oldest failure should have rotated out"))

  (test-case "record-failure! is guarded: a value whose printing fails is still counted"
    ;; record-failure! is the direct handler of swallow-and-count-failure's
    ;; with-handlers, so nothing else protects it.  An escape there is a process
    ;; death, not a raise (it runs in alloc.rkt's raw atomic region).
    (struct unprintable ()
      #:property prop:custom-write
      (lambda (v port mode) (error 'custom-write "printing this raises")))
    (define before (finalizer-failures))
    (check-equal? ((swallow-and-count-failure (lambda (_t) (raise (unprintable)))) 'handle)
                  (void))
    (check-equal? (finalizer-failures) (add1 before)))

  (test-case "swallow-and-count-failure passes successful releases through"
    (define released '())
    (define guarded
      (swallow-and-count-failure (lambda (t) (set! released (cons t released)))))
    (guarded 'a)
    (guarded 'b)
    (check-equal? released '(b a)))

  (test-case "swallow-and-count-failure's catch is total — bare raises too"
    (define guarded (swallow-and-count-failure (lambda (_) (raise 'not-an-exn))))
    (check-equal? (guarded 'handle) (void)))

  (test-case "(deallocator) cancels the pending finalizer — observably"
    ;; count, never record the pointer — a recorded pointer would stay
    ;; reachable and the no-finalizer assertion would pass vacuously
    (define finalizer-count 0)
    (define explicit-count 0)
    (define (finalizer-release p)
      (set! finalizer-count (add1 finalizer-count))
      (free p))
    (define (explicit-release p)
      (set! explicit-count (add1 explicit-count))
      (free p))
    (define alloc
      ((allocator (swallow-and-count-failure finalizer-release))
       (lambda () (malloc 16 'raw))))
    (define release ((deallocator) explicit-release))
    (void (alloc))
    (collect-until (lambda () (> finalizer-count 0)))
    (check-equal? finalizer-count 1 "finalizer ran for the dropped alloc")
    (let ([p (alloc)])
      (release p))
    (check-equal? explicit-count 1 "explicit release ran")
    (collect-until (lambda () (> finalizer-count 1)) #:tries 10)
    (check-equal? finalizer-count 1
                  "no finalizer fired after explicit release — canceled"))

  (test-case "explicit tensor-free! is unguarded; freed handles are inert"
    (define t (ones 2 2))
    (check-equal? (tensor-shape t) '(2 2))
    (tensor-free! t)
    (check-exn exn:fail? (lambda () (add t t)))
    (check-exn exn:fail:contract? (lambda () (tensor-free! t)))
    (collect-garbage)
    (collect-garbage)))
