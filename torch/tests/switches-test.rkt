#lang racket/base

(module+ test
  (require rackunit
           (only-in ffi/unsafe _fun _int _ptr get-ffi-obj)
           (only-in racket/dict in-dict)
           (only-in racket/place dynamic-place place-channel-put place-dead-evt
                    place-kill)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in "../foreign/raw/native-library.rkt" native-library)
           (only-in "../main.rkt"
                    conv2d cuda-available? cudnn-benchmark? cudnn-deterministic?
                    deterministic-algorithms-enabled?
                    deterministic-algorithms-warn-only? manual-seed! randn
                    set-cudnn-benchmark! set-cudnn-deterministic! tensor->bytes
                    use-deterministic-algorithms!)
           (only-in "private/blame.rkt" arity-error-of? blames-caller?)
           (only-in "private/switches.rkt"
                    bounded-switches-test default-switches determinism-rows
                    set-switches! switch-values within/switches-restored)
           (only-in "private/within.rkt" within))

  (check-equal? (switch-values) default-switches
                "S0: libtorch's defaults, and no flag leaked from an earlier file")

  (define all-on
    (for/hasheq ([(k v) (in-hash default-switches)])
      (values k (not v))))

  (define (c-getter name)
    (get-ffi-obj name native-library
                 (_fun (out : (_ptr o _int))
                       -> (rc : _int)
                       -> (if (zero? rc) (list-ref '(#f #t) out) (cons 'rc rc)))))

  (define c-reads
    (hasheq 'deterministic-algorithms (c-getter "tr_get_deterministic_algorithms")
            'warn-only (c-getter "tr_get_deterministic_algorithms_warn_only")
            'cudnn-deterministic (c-getter "tr_get_cudnn_deterministic")
            'cudnn-benchmark (c-getter "tr_get_cudnn_benchmark")))

  (define (c-values)
    (for/hasheq ([(k read) (in-hash c-reads)])
      (values k (read))))

  (define c-set-algorithms
    (get-ffi-obj "tr_set_deterministic_algorithms" native-library (_fun _int _int -> _int)))
  (define c-set-cudnn-deterministic
    (get-ffi-obj "tr_set_cudnn_deterministic" native-library (_fun _int -> _int)))
  (define c-set-cudnn-benchmark
    (get-ffi-obj "tr_set_cudnn_benchmark" native-library (_fun _int -> _int)))

  (define (c-set-switches! flags)
    (define (bit k) (if (hash-ref flags k) 1 0))
    (list (c-set-algorithms (bit 'deterministic-algorithms) (bit 'warn-only))
          (c-set-cudnn-deterministic (bit 'cudnn-deterministic))
          (c-set-cudnn-benchmark (bit 'cudnn-benchmark))))

  (struct row (label write! changes))

  (define writes
    (append
     (for*/list ([on? (in-list '(#t #f))] [w (in-list '(#t #f))])
       (row (format "(use-deterministic-algorithms! ~a #:warn-only? ~a)" on? w)
            (lambda () (use-deterministic-algorithms! on? #:warn-only? w))
            (list 'deterministic-algorithms on? 'warn-only w)))
     (list (row "(use-deterministic-algorithms! #t)"
                (lambda () (use-deterministic-algorithms! #t))
                '(deterministic-algorithms #t warn-only #f)))
     (for*/list ([(key setter)
                  (in-dict (list (cons 'cudnn-deterministic set-cudnn-deterministic!)
                                 (cons 'cudnn-benchmark set-cudnn-benchmark!)))]
                 [on? (in-list '(#t #f))])
       (row (format "~a ~a" key on?)
            (lambda () (setter on?))
            (list key on?)))))

  (bounded-switches-test "S1, S2: each setter writes its own flag in libtorch, and nothing else"
    (for* ([start (in-list (list default-switches all-on))]
           [w (in-list writes)])
      (set-switches! start)
      ((row-write! w))
      (define want (apply hash-set* start (row-changes w)))
      (define where (format "~a from ~a" (row-label w) start))
      (check-equal? (c-values) want (string-append "libtorch's flags after " where))
      (check-equal? (switch-values) want (string-append "the getters after " where))))

  (define every-combination
    (for/fold ([combos (list (hasheq))]) ([k (in-hash-keys default-switches)])
      (for*/list ([c (in-list combos)] [v (in-list '(#f #t))])
        (hash-set c k v))))

  (bounded-switches-test "S1, S2: each getter reads its own flag of libtorch's"
    (for ([flags (in-list every-combination)])
      (check-equal? (c-set-switches! flags) '(0 0 0))
      (check-equal? (switch-values) flags (format "after libtorch was set to ~a" flags))))

  (bounded-switches-test "S1: the keyword default clears an earlier #:warn-only? #t"
    (use-deterministic-algorithms! #f #:warn-only? #t)
    (check-true (deterministic-algorithms-warn-only?))
    (use-deterministic-algorithms! #t)
    (check-true (deterministic-algorithms-enabled?))
    (check-false (deterministic-algorithms-warn-only?)))

  (bounded-switches-test "S3: the cuDNN flags are stored and read back, CUDA or not"
    (define x (randn 2 3 8 8))
    (define k (randn 4 3 3 3))
    (define plain (tensor->bytes (conv2d x k)))
    (set-cudnn-deterministic! #t)
    (set-cudnn-benchmark! #t)
    (check-true (cudnn-deterministic?)
                (format "cuda-available? ~a" (cuda-available?)))
    (check-true (cudnn-benchmark?))
    (check-equal? (tensor->bytes (conv2d x k)) plain
                  "a CPU convolution is unchanged by the cuDNN flags"))

  (bounded-switches-test "S4: a non-boolean blames the caller and writes nothing"
    (define blamed? (blames-caller? "switches-test.rkt"))
    (for ([(call refused?)
           (in-dict
            (list (cons (lambda () (use-deterministic-algorithms! 1)) blamed?)
                  (cons (lambda () (use-deterministic-algorithms! #t #:warn-only? 'yes))
                        blamed?)
                  (cons (lambda () (set-cudnn-deterministic! "true")) blamed?)
                  (cons (lambda () (set-cudnn-benchmark! (void))) blamed?)
                  (cons (lambda () (cudnn-benchmark? #t))
                        (arity-error-of? "cudnn-benchmark?"))))])
      (define before (switch-values))
      (check-exn refused? call)
      (check-equal? (switch-values) before)))

  (bounded-switches-test "S5: an OS thread reads main's write, and main reads the thread's"
    (set-switches! all-on)
    (define seen #f)
    (thread-wait (thread (lambda () (set! seen (switch-values))) #:pool 'own))
    (check-equal? seen all-on)
    (thread-wait (thread (lambda () (set-switches! default-switches))
                         #:pool 'own))
    (check-equal? (switch-values) default-switches))

  (define-runtime-path place-switches "private/place-switches.rkt")

  (define (seconds-from-now n)
    (alarm-evt (+ (current-inexact-milliseconds) (* 1000 n))))

  (define (reply-from p silent)
    (sync p
          (handle-evt (place-dead-evt p)
                      (lambda (_) (or (sync/timeout 0 p) 'the-place-died-before-replying)))
          (handle-evt silent (lambda (_) 'the-place-fell-silent))))

  (bounded-switches-test "S6: a place reads main's write, and main reads the place's"
    #:seconds 120
    (set-switches! all-on)
    (define p (dynamic-place place-switches 'main))
    (define silent (seconds-from-now 100))
    (define read-there (reply-from p silent))
    (define written
      (cond
        [(hash? read-there)
         (place-channel-put p default-switches)
         (reply-from p silent)]
        [else read-there]))
    (define after (switch-values))
    (place-kill p)
    (check-equal? read-there all-on "the place read main's write")
    (check-equal? written 'written "the place wrote and replied")
    (check-equal? after default-switches "main read the place's write"))

  (bounded-switches-test "S6: a place that dies before replying is reported as dead"
    #:seconds 120
    (define doomed (dynamic-place place-switches 'die))
    (define reply (reply-from doomed (seconds-from-now 100)))
    (place-kill doomed)
    (check-equal? reply 'the-place-died-before-replying))

  (bounded-switches-test "S7: with the switch on, no row refuses on the CPU"
    (use-deterministic-algorithms! #t)
    (manual-seed! 0)
    (for ([(name run) (in-dict determinism-rows)])
      (check-not-exn (lambda () (run 'cpu)) (symbol->string name))))

  (define (refusal-of thunk)
    (with-handlers ([exn:test:check? (lambda (_) 'no-result-in-time)]
                    [exn:fail? exn-message])
      (thunk)
      #f))

  (bounded-switches-test "S8: a switches test restores every flag when its body raises"
    (check-equal? (refusal-of
                   (lambda ()
                     (within/switches-restored 10
                                               (lambda ()
                                                 (set-switches! all-on)
                                                 (error "boom")))))
                  "boom")
    (check-equal? (switch-values) default-switches))

  (bounded-switches-test "S8: a switches test restores every flag when its body times out"
    (check-equal? (refusal-of
                   (lambda ()
                     (within/switches-restored 1
                                               (lambda ()
                                                 (set-switches! all-on)
                                                 (sleep 5)))))
                  'no-result-in-time)
    (check-equal? (switch-values) default-switches))

  (bounded-switches-test "S8: nothing a timed-out body started writes a flag after the restore"
    #:seconds 10
    (check-equal? (refusal-of
                   (lambda ()
                     (within/switches-restored 0.5
                                               (lambda ()
                                                 (within 5
                                                         (lambda ()
                                                           (sleep 1)
                                                           (set-switches! all-on)))))))
                  'no-result-in-time)
    (sleep 1.5)
    (check-equal? (switch-values) default-switches)))
