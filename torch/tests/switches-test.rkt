#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/dict in-dict)
           (only-in racket/match match-define)
           (only-in racket/place dynamic-place place-channel-put place-kill)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in "../main.rkt"
                    conv2d cuda-available? cudnn-benchmark? cudnn-deterministic?
                    deterministic-algorithms-enabled?
                    deterministic-algorithms-warn-only? manual-seed! randn
                    set-cudnn-benchmark! set-cudnn-deterministic! tensor->bytes
                    use-deterministic-algorithms!)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/switches.rkt"
                    default-switches determinism-rows set-switches!
                    switch-values with-switches-restored)
           (only-in "private/within.rkt" bounded-test within))

  (check-equal? (switch-values) default-switches
                "S0: libtorch's defaults, and no flag leaked from an earlier file")

  (define (flipped switches)
    (for/hash ([(k v) (in-hash switches)])
      (values k (not v))))

  (define all-on (flipped default-switches))

  (define writes
    (append
     (for*/list ([on? (in-list '(#t #f))] [w (in-list '(#t #f))])
       (list (format "(use-deterministic-algorithms! ~a #:warn-only? ~a)" on? w)
             (lambda () (use-deterministic-algorithms! on? #:warn-only? w))
             (hash 'deterministic-algorithms on? 'warn-only w)))
     (list (list "(use-deterministic-algorithms! #t)"
                 (lambda () (use-deterministic-algorithms! #t))
                 (hash 'deterministic-algorithms #t 'warn-only #f)))
     (for*/list ([(key setter)
                  (in-dict (list (cons 'cudnn-deterministic set-cudnn-deterministic!)
                                 (cons 'cudnn-benchmark set-cudnn-benchmark!)))]
                 [on? (in-list '(#t #f))])
       (list (format "~a ~a" key on?)
             (lambda () (setter on?))
             (hash key on?)))))

  (define (updated start changes)
    (for/fold ([acc start]) ([(k v) (in-hash changes)])
      (hash-set acc k v)))

  (bounded-test "S1, S2: each setter reads back what it wrote and nothing else"
    (for* ([start (in-list (list default-switches all-on))]
           [row (in-list writes)])
      (match-define (list label write! changes) row)
      (with-switches-restored
        (set-switches! start)
        (write!)
        (check-equal? (switch-values) (updated start changes)
                      (format "~a from ~a" label start)))))

  (bounded-test "S1: the keyword default clears an earlier #:warn-only? #t"
    (with-switches-restored
      (use-deterministic-algorithms! #f #:warn-only? #t)
      (check-true (deterministic-algorithms-warn-only?))
      (use-deterministic-algorithms! #t)
      (check-true (deterministic-algorithms-enabled?))
      (check-false (deterministic-algorithms-warn-only?))))

  (bounded-test "S3: the cuDNN flags are stored and read back, CUDA or not"
    (define x (randn 2 3 8 8))
    (define k (randn 4 3 3 3))
    (define plain (tensor->bytes (conv2d x k)))
    (with-switches-restored
      (set-cudnn-deterministic! #t)
      (set-cudnn-benchmark! #t)
      (check-true (cudnn-deterministic?)
                  (format "cuda-available? ~a" (cuda-available?)))
      (check-true (cudnn-benchmark?))
      (check-equal? (tensor->bytes (conv2d x k)) plain
                    "a CPU convolution is unchanged by the cuDNN flags")))

  (bounded-test "S4: a non-boolean blames the caller and writes nothing"
    (define blamed? (blames-caller? "switches-test.rkt"))
    (define (arity-of? who)
      (lambda (e)
        (and (exn:fail:contract:arity? e)
             (regexp-match? (regexp (string-append "^" (regexp-quote who) ": "))
                            (exn-message e)))))
    (for ([(call refused?)
           (in-dict
            (list (cons (lambda () (use-deterministic-algorithms! 1)) blamed?)
                  (cons (lambda () (use-deterministic-algorithms! #t #:warn-only? 'yes))
                        blamed?)
                  (cons (lambda () (set-cudnn-deterministic! "true")) blamed?)
                  (cons (lambda () (set-cudnn-benchmark! (void))) blamed?)
                  (cons (lambda () (cudnn-benchmark? #t))
                        (arity-of? "cudnn-benchmark?"))))])
      (define before (switch-values))
      (check-exn refused? call)
      (check-equal? (switch-values) before)))

  (bounded-test "S5: an OS thread reads main's write, and main reads the thread's"
    (with-switches-restored
      (set-switches! all-on)
      (define seen #f)
      (thread-wait (thread (lambda () (set! seen (switch-values))) #:pool 'own))
      (check-equal? seen all-on)
      (thread-wait (thread (lambda () (set-switches! default-switches))
                           #:pool 'own))
      (check-equal? (switch-values) default-switches)))

  (define-runtime-path place-switches "private/place-switches.rkt")

  (test-case "S6: a place reads main's write, and main reads the place's"
    (within 120
            (lambda ()
              (with-switches-restored
                (set-switches! all-on)
                (define p (dynamic-place place-switches 'main))
                (define seen (sync/timeout 100 p))
                (place-channel-put p default-switches)
                (define written (sync/timeout 100 p))
                (define after (switch-values))
                (place-kill p)
                (check-equal? seen all-on "the place read main's write")
                (check-equal? written 'written)
                (check-equal? after default-switches
                              "main read the place's write")))))

  (bounded-test "S7: with the switch on, no row refuses on the CPU"
    (with-switches-restored
      (use-deterministic-algorithms! #t)
      (manual-seed! 0)
      (for ([(name run) (in-dict determinism-rows)])
        (check-not-exn (lambda () (run 'cpu)) (symbol->string name)))))

  (bounded-test "S8: with-switches-restored restores every flag when its body raises"
    (check-exn #rx"^boom$"
               (lambda ()
                 (with-switches-restored
                   (set-switches! all-on)
                   (error "boom"))))
    (check-equal? (switch-values) default-switches)))
