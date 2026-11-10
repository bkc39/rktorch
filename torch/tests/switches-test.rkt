#lang racket/base

(module+ test
  (require rackunit
           (only-in compiler/find-exe find-exe)
           (only-in ffi/unsafe _fun _int _ptr get-ffi-obj)
           (only-in racket/dict in-dict)
           (only-in racket/file file->string)
           (only-in racket/list index-of)
           (only-in racket/place dynamic-place place-channel-put place-dead-evt
                    place-kill)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-contains?)
           (only-in racket/system system*/exit-code)
           (only-in "../foreign/raw/native-library.rkt" native-library)
           (only-in "../main.rkt"
                    conv2d cuda-available? cuda-matmul-allow-tf32? cudnn-allow-tf32?
                    cudnn-benchmark? cudnn-deterministic?
                    deterministic-algorithms-enabled?
                    deterministic-algorithms-warn-only? float32-matmul-precision
                    manual-seed! randn set-cuda-matmul-allow-tf32!
                    set-cudnn-allow-tf32! set-cudnn-benchmark! set-cudnn-deterministic!
                    set-float32-matmul-precision! tensor->bytes
                    use-deterministic-algorithms!)
           (only-in "private/blame.rkt" arity-error-of? blames-caller?)
           (only-in "private/switches.rkt"
                    bounded-switches-test default-switches determinism-rows
                    matmul-witness other-switches precision-refusal? precision-sequences
                    sequence-outcome set-switches! switch-values
                    within/switches-restored)
           (only-in "private/within.rkt" within))

  (define override-variable "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")

  (check-equal? (switch-values) default-switches
                (format (string-append "S0: libtorch's defaults with ~a unset (it is ~s), "
                                       "and no flag leaked from an earlier file")
                        override-variable (getenv override-variable)))

  (define precisions '(highest high medium))

  (define (c-getter name decode)
    (get-ffi-obj name native-library
                 (_fun (out : (_ptr o _int))
                       -> (rc : _int)
                       -> (if (zero? rc) (decode out) (cons 'rc rc)))))

  (define (flag-getter name)
    (c-getter name (lambda (out) (list-ref '(#f #t) out))))

  (define c-reads
    (hasheq 'deterministic-algorithms (flag-getter "tr_get_deterministic_algorithms")
            'warn-only (flag-getter "tr_get_deterministic_algorithms_warn_only")
            'cudnn-deterministic (flag-getter "tr_get_cudnn_deterministic")
            'cudnn-benchmark (flag-getter "tr_get_cudnn_benchmark")
            'float32-matmul-precision
            (c-getter "tr_get_float32_matmul_precision"
                      (lambda (out) (list-ref precisions out)))
            'cuda-matmul-allow-tf32 (flag-getter "tr_get_cuda_matmul_allow_tf32")
            'cudnn-allow-tf32 (flag-getter "tr_get_cudnn_allow_tf32")))

  (define (c-values)
    (for/hasheq ([(k read) (in-hash c-reads)])
      (values k (read))))

  (define c-set-algorithms
    (get-ffi-obj "tr_set_deterministic_algorithms" native-library (_fun _int _int -> _int)))
  (define c-set-cudnn-deterministic
    (get-ffi-obj "tr_set_cudnn_deterministic" native-library (_fun _int -> _int)))
  (define c-set-cudnn-benchmark
    (get-ffi-obj "tr_set_cudnn_benchmark" native-library (_fun _int -> _int)))
  (define c-set-precision
    (get-ffi-obj "tr_set_float32_matmul_precision" native-library (_fun _int -> _int)))
  (define c-set-cudnn-tf32
    (get-ffi-obj "tr_set_cudnn_allow_tf32" native-library (_fun _int -> _int)))

  (define (c-set-switches! flags)
    (define (bit k) (if (hash-ref flags k) 1 0))
    (list (c-set-algorithms (bit 'deterministic-algorithms) (bit 'warn-only))
          (c-set-cudnn-deterministic (bit 'cudnn-deterministic))
          (c-set-cudnn-benchmark (bit 'cudnn-benchmark))
          (c-set-precision (index-of precisions (hash-ref flags 'float32-matmul-precision)))
          (c-set-cudnn-tf32 (bit 'cudnn-allow-tf32))))

  (define (raises? want)
    (eq? (hash-ref want 'float32-matmul-precision) 'raise))

  (define (c-view want)
    (if (raises? want)
        (hash-set want 'float32-matmul-precision '(rc . 1))
        want))

  (define (getter-values)
    (with-handlers ([precision-refusal? (lambda (_) 'raise)])
      (switch-values)))

  (define (precision-after-matmul start on?)
    (cond
      [on? 'high]
      [(eq? (hash-ref start 'float32-matmul-precision) 'highest) 'highest]
      [else 'raise]))

  (struct row (label write! changes))

  (define writes
    (append
     (for*/list ([on? (in-list '(#t #f))] [w (in-list '(#t #f))])
       (row (format "(use-deterministic-algorithms! ~a #:warn-only? ~a)" on? w)
            (lambda () (use-deterministic-algorithms! on? #:warn-only? w))
            (lambda (_) (list 'deterministic-algorithms on? 'warn-only w))))
     (list (row "(use-deterministic-algorithms! #t)"
                (lambda () (use-deterministic-algorithms! #t))
                (lambda (_) '(deterministic-algorithms #t warn-only #f))))
     (for*/list ([(key setter)
                  (in-dict (list (cons 'cudnn-deterministic set-cudnn-deterministic!)
                                 (cons 'cudnn-benchmark set-cudnn-benchmark!)))]
                 [on? (in-list '(#t #f))])
       (row (format "~a ~a" key on?)
            (lambda () (setter on?))
            (lambda (_) (list key on?))))
     (for/list ([p (in-list precisions)])
       (row (format "(set-float32-matmul-precision! '~a)" p)
            (lambda () (set-float32-matmul-precision! p))
            (lambda (_)
              (list 'float32-matmul-precision p
                    'cuda-matmul-allow-tf32 (not (eq? p 'highest))))))
     (for/list ([on? (in-list '(#t #f))])
       (row (format "(set-cuda-matmul-allow-tf32! ~a)" on?)
            (lambda () (set-cuda-matmul-allow-tf32! on?))
            (lambda (start)
              (list 'cuda-matmul-allow-tf32 on?
                    'float32-matmul-precision (precision-after-matmul start on?)))))
     (for/list ([on? (in-list '(#t #f))])
       (row (format "(set-cudnn-allow-tf32! ~a)" on?)
            (lambda () (set-cudnn-allow-tf32! on?))
            (lambda (_) (list 'cudnn-allow-tf32 on?))))))

  (bounded-switches-test "S1, S2: each setter writes its own flag in libtorch, and nothing else"
    (for* ([start (in-list (list default-switches other-switches))]
           [w (in-list writes)])
      (set-switches! start)
      ((row-write! w))
      (define want (apply hash-set* start ((row-changes w) start)))
      (define where (format "~a from ~a" (row-label w) start))
      (check-equal? (c-values) (c-view want) (string-append "libtorch's flags after " where))
      (check-equal? (getter-values) (if (raises? want) 'raise want)
                    (string-append "the getters after " where))))

  (define r3-combinations
    (for/fold ([combos (list (hasheq))])
              ([k (in-list '(deterministic-algorithms warn-only
                             cudnn-deterministic cudnn-benchmark))])
      (for*/list ([c (in-list combos)] [v (in-list '(#f #t))])
        (hash-set c k v))))

  (define every-combination
    (for*/list ([r3 (in-list r3-combinations)]
                [p (in-list precisions)]
                [cudnn-tf32? (in-list '(#f #t))])
      (hash-set* r3
                 'float32-matmul-precision p
                 'cuda-matmul-allow-tf32 (not (eq? p 'highest))
                 'cudnn-allow-tf32 cudnn-tf32?)))

  (bounded-switches-test "S1, S2: each getter reads its own flag of libtorch's"
    (check-equal? (length every-combination) 96)
    (for ([flags (in-list every-combination)])
      (check-equal? (c-set-switches! flags) '(0 0 0 0 0))
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
    (set-switches! other-switches)
    (define seen #f)
    (thread-wait (thread (lambda () (set! seen (switch-values))) #:pool 'own))
    (check-equal? seen other-switches)
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
    (set-switches! other-switches)
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
    (check-equal? read-there other-switches "the place read main's write")
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
                                                 (set-switches! other-switches)
                                                 (error "boom")))))
                  "boom")
    (check-equal? (switch-values) default-switches))

  (bounded-switches-test "S8: a switches test restores every flag when its body times out"
    (check-equal? (refusal-of
                   (lambda ()
                     (within/switches-restored 1
                                               (lambda ()
                                                 (set-switches! other-switches)
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
                                                           (set-switches! other-switches)))))))
                  'no-result-in-time)
    (sleep 1.5)
    (check-equal? (switch-values) default-switches))

  (define coupling
    '((high high #t #t)
      (medium medium #t #t)
      (highest highest #f #t)
      (matmul-on high #t #t)
      (matmul-on-off highest #f #t)
      (high-matmul-off raise #f #t)
      (medium-matmul-on raise #t #t)
      (medium-matmul-off raise #f #t)
      (high-matmul-on high #t #t)
      (matmul-on-medium medium #t #t)
      (cudnn-off highest #f #f)
      (cudnn-off-high high #t #f)
      (high-cudnn-off high #t #f)
      (high-matmul-off-highest highest #f #t)))

  (bounded-switches-test "S9: the precision and the TF32 flags couple as in PyTorch"
    (check-equal? (map car precision-sequences) (map car coupling))
    (for ([(name steps) (in-dict precision-sequences)])
      (check-equal? (sequence-outcome steps) (cdr (assq name coupling))
                    (format "~a: ~s from the defaults" name steps))))

  (define-runtime-path main-module "../main.rkt")

  (define precision-defaults-program
    `(begin
       (require (only-in (file ,(path->string main-module))
                         cuda-matmul-allow-tf32? cudnn-allow-tf32?
                         float32-matmul-precision))
       (write (list (float32-matmul-precision)
                    (cuda-matmul-allow-tf32?)
                    (cudnn-allow-tf32?)))))

  (define (defaults-in-child override)
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env (string->bytes/utf-8 override-variable) override)
    (define out (open-output-string))
    (define err (open-output-string))
    (define status
      (parameterize ([current-environment-variables env]
                     [current-output-port out]
                     [current-error-port err])
        (system*/exit-code (find-exe) "-e" (format "~s" precision-defaults-program))))
    (values status
            (read (open-input-string (get-output-string out)))
            (get-output-string err)))

  (bounded-switches-test "S10: TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 makes 'high the starting precision"
    #:seconds 240
    (for ([(override want) (in-dict (list (cons #"1" '(high #t #t))
                                          (cons #f '(highest #f #t))))])
      (define-values (status read-there err) (defaults-in-child override))
      (define where (format "~a=~a; stderr:\n~a" override-variable override err))
      (check-equal? status 0 where)
      (check-equal? read-there want where)))

  (define (cpu-witnesses)
    (manual-seed! 0)
    (define x (randn 2 3 8 8))
    (define k (randn 4 3 3 3))
    (lambda ()
      (list (tensor->bytes (matmul-witness 'cpu))
            (tensor->bytes (conv2d x k)))))

  (bounded-switches-test "S11: on the CPU the TF32 flags change no matmul or convolution"
    #:seconds 120
    (define run (cpu-witnesses))
    (define plain (run))
    (set-cuda-matmul-allow-tf32! #t)
    (check-equal? (run) plain "after (set-cuda-matmul-allow-tf32! #t)")
    (set-switches! default-switches)
    (set-cudnn-allow-tf32! #f)
    (check-equal? (run) plain "after (set-cudnn-allow-tf32! #f)"))

  (bounded-switches-test "S12: a bad precision or flag blames the caller and writes nothing"
    (define blamed? (blames-caller? "switches-test.rkt"))
    (for ([(call refused?)
           (in-dict
            (list (cons (lambda () (set-float32-matmul-precision! "high")) blamed?)
                  (cons (lambda () (set-float32-matmul-precision! 'HIGH)) blamed?)
                  (cons (lambda () (set-float32-matmul-precision! 'bf16)) blamed?)
                  (cons (lambda () (set-float32-matmul-precision! #f)) blamed?)
                  (cons (lambda () (set-cuda-matmul-allow-tf32! 1)) blamed?)
                  (cons (lambda () (set-cudnn-allow-tf32! "yes")) blamed?)
                  (cons (lambda () (float32-matmul-precision 'high))
                        (arity-error-of? "float32-matmul-precision"))
                  (cons (lambda () (cuda-matmul-allow-tf32? #t))
                        (arity-error-of? "cuda-matmul-allow-tf32?"))))])
      (define before (switch-values))
      (check-exn refused? call)
      (check-equal? (switch-values) before)))

  (define-runtime-path manual-examples "../scribblings/examples")
  (define-runtime-path example-programs "../../examples")

  (define precision-setters
    '("set-float32-matmul-precision!" "set-cuda-matmul-allow-tf32!" "set-cudnn-allow-tf32!"))

  (define (setter-mentions dir)
    (for*/list ([f (in-directory dir)]
                #:when (file-exists? f)
                #:unless (regexp-match? #rx"/compiled/" (path->string f))
                [setter (in-list precision-setters)]
                #:when (string-contains? (file->string f) setter))
      (format "~a: ~a" f setter)))

  (cond
    [(directory-exists? example-programs)
     (bounded-switches-test "S13: no example or manual program sets a precision switch"
       (check-equal? (append (setter-mentions manual-examples)
                             (setter-mentions example-programs))
                     '()))]
    [else
     (displayln "[switches-test] skipped S13: no examples/ beside the package")]))
