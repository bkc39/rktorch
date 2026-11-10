#lang racket/base

(module+ test
  (require rackunit
           (only-in compiler/find-exe find-exe)
           (only-in racket/dict in-dict)
           (only-in racket/list remove-duplicates)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*/exit-code)
           (only-in "../main.rkt"
                    * - / @ ~> abs adaptive-avg-pool2d backward! conv2d cuda-available?
                    deterministic-algorithms-enabled? detach exn:fail:rktorch:oom?
                    grad index-add! index-select item manual-seed! max mean rand randn
                    requires-grad! reshape set-cuda-matmul-allow-tf32!
                    set-cudnn-allow-tf32! set-cudnn-benchmark! set-cudnn-deterministic!
                    set-float32-matmul-precision! sub! sum to to-dtype
                    use-deterministic-algorithms! with-no-grad zeros)
           (only-in "private/host.rkt" bytes-of distinct-runs host)
           (only-in "private/switches.rkt"
                    bounded-switches-test conv-witness default-switches distinct-values
                    matmul-witness set-switches! switch-values tf32-hardware?))

  (check-equal? (switch-values) default-switches
                "no flag leaked from an earlier file")

  (define (pooled-backward! x)
    (~> x (adaptive-avg-pool2d '(3 3)) sum backward!))

  (define refusal
    (pregexp
     (string-append
      "^backward!: FFI call failed \\(rc=1\\): tr_tensor_backward: "
      "adaptive_avg_pool2d_backward_cuda does not have a deterministic "
      "implementation, but you set 'torch.use_deterministic_algorithms\\(True\\)'")))

  (define (index-add-run)
    (manual-seed! 0)
    (define src (randn (expt 2 20) #:device 'cuda))
    (define idx (~> (expt 2 20) (rand #:device 'cuda) (* 8) (to-dtype 'int64)))
    (lambda ()
      (define out (zeros 8 #:device 'cuda))
      (index-add! out 0 idx src)
      (bytes-of out)))

  (define (seeded-step)
    (manual-seed! 0)
    (define w (randn 8 16 #:device 'cuda #:requires-grad? #t))
    (define k (randn 4 1 3 3 #:device 'cuda #:requires-grad? #t))
    (define idx (~> (expt 2 18) (rand #:device 'cuda) (* 8) (to-dtype 'int64)))
    (define x (reshape (index-select w 0 idx) 1024 1 64 64))
    (~> (conv2d x k) mean backward!)
    (with-no-grad
      (sub! w (grad w) 0.1)
      (sub! k (grad k) 0.1))
    (list (bytes-of w) (bytes-of k)))

  (define (conv-weight-grad)
    (manual-seed! 0)
    (define x (randn 1024 1 64 64 #:device 'cuda))
    (define k (randn 4 1 3 3 #:device 'cuda #:requires-grad? #t))
    (~> (conv2d x k) mean backward!)
    (bytes-of (grad k)))

  (define (matmul-run)
    (manual-seed! 0)
    (define a (randn 4096 2048 #:device 'cuda #:requires-grad? #t))
    (define b (randn 2048 1024 #:device 'cuda #:requires-grad? #t))
    (define out (@ a b))
    (backward! (sum out))
    (list (bytes-of out) (bytes-of (grad a)) (bytes-of (grad b))))

  (define-runtime-path main-module "../main.rkt")

  (define (warn-only-program on)
    `(begin
       (require (only-in (file ,(path->string main-module))
                         ~> adaptive-avg-pool2d backward! randn sum
                         use-deterministic-algorithms!))
       (define captured (open-output-string))
       (define status
         (with-handlers ([(lambda (_) #t)
                          (lambda (e)
                            (eprintf "~a\n" (if (exn? e) (exn-message e) e))
                            4)])
           (use-deterministic-algorithms! #t #:warn-only? #t)
           (parameterize ([current-error-port captured])
             (~> (randn 1 2 5 5 #:device ',on #:requires-grad? #t)
                 (adaptive-avg-pool2d '(3 3))
                 sum
                 backward!))
           (if (string=? (get-output-string captured) "") 0 3)))
       (exit status)))

  (define (warn-only-child on)
    (define os-err (open-output-string))
    (define status
      (parameterize ([current-error-port os-err])
        (system*/exit-code (find-exe) "-e" (format "~s" (warn-only-program on)))))
    (values status (get-output-string os-err)))

  (define (exit-meaning status)
    (case status
      [(0) "the backward ran and wrote nothing to current-error-port"]
      [(3) "the warning reached current-error-port, not the process's standard error"]
      [(4) "the program raised; its error is on stderr below"]
      [else "racket failed outside the program's handler, e.g. in its require"]))

  (define (child-report status err)
    (format "exit ~a: ~a; stderr:\n~a" status (exit-meaning status) err))

  (bounded-switches-test "C2: a child that raises reports its error to the parent"
    #:seconds 120
    (define-values (status err) (warn-only-child 'nowhere))
    (define report (child-report status err))
    (check-equal? status 4 report)
    (check-regexp-match #rx"nowhere" report)
    (check-regexp-match #rx"the program raised" report))

  (define warning
    #rx"adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation")

  (define (cuda-tests)
    (bounded-switches-test "C1: the switch refuses a CUDA backward with no deterministic kernel"
      (define x (randn 1 2 5 5 #:device 'cuda #:requires-grad? #t))
      (use-deterministic-algorithms! #t)
      (define e
        (with-handlers ([exn:fail? values])
          (pooled-backward! x)
          #f))
      (check-pred exn:fail? e)
      (check-false (exn:fail:contract? e))
      (check-false (exn:fail:rktorch:oom? e))
      (check-regexp-match refusal (exn-message e))
      (check-true (deterministic-algorithms-enabled?))
      (use-deterministic-algorithms! #f)
      (check-not-exn (lambda () (pooled-backward! x))))

    (bounded-switches-test "C2: warn-only runs the nondeterministic kernel"
      (use-deterministic-algorithms! #t #:warn-only? #t)
      (define x-cpu (randn 1 2 5 5 #:requires-grad? #t))
      (define x-cuda (~> x-cpu detach (to 'cuda) requires-grad!))
      (define racket-err (open-output-string))
      (parameterize ([current-error-port racket-err])
        (pooled-backward! x-cuda))
      (pooled-backward! x-cpu)
      (check-equal? (get-output-string racket-err) ""
                    "the warning does not go to current-error-port")
      (for ([g (in-list (host (grad x-cuda)))]
            [want (in-list (host (grad x-cpu)))])
        (check-within g want 1e-6)))

    (bounded-switches-test "C2: warn-only's warning goes to the process's standard error"
      #:seconds 120
      (define-values (status err) (warn-only-child 'cuda))
      (check-equal? status 0 (child-report status err))
      (check-regexp-match warning err))

    (bounded-switches-test "C3: index-add! repeats bit for bit only under the switch"
      (check-true (< 1 (distinct-runs 20 (index-add-run)))
                  "the witness: runs differ with the switch off")
      (set-cudnn-deterministic! #t)
      (check-true (< 1 (distinct-runs 20 (index-add-run)))
                  "runs still differ with only the cuDNN flag on")
      (set-cudnn-deterministic! #f)
      (use-deterministic-algorithms! #t)
      (check-equal? (distinct-runs 20 (index-add-run)) 1))

    (bounded-switches-test "C4: a seeded step repeats bit for bit only under the switch"
      (check-true (< 1 (distinct-runs 10 seeded-step))
                  "the witness: runs differ with the switch off")
      (use-deterministic-algorithms! #t)
      (check-equal? (distinct-runs 10 seeded-step) 1))

    (bounded-switches-test "C5: cuDNN's flag makes a convolution's weight gradient repeat"
      (check-true (< 1 (distinct-runs 10 conv-weight-grad))
                  (string-append
                   "the witness: runs differ with the flag off, so long as cuDNN's "
                   "heuristic picks a nondeterministic weight-gradient algorithm here; "
                   "another GPU or cuDNN may pick a deterministic one, and a larger "
                   "input would not change that"))
      (set-cudnn-deterministic! #t)
      (check-equal? (distinct-runs 10 conv-weight-grad) 1)
      (set-cudnn-benchmark! #t)
      (check-equal? (distinct-runs 10 conv-weight-grad) 1
                    "also with benchmark on")))

  (define (matmul-test)
    (bounded-switches-test "C6: without CUBLAS_WORKSPACE_CONFIG a matmul repeats under the switch"
      (use-deterministic-algorithms! #t)
      (check-equal? (distinct-runs 10 matmul-run) 1)))

  (define (matmul-tf32-settings)
    (list (cons "'high" (lambda () (set-float32-matmul-precision! 'high)))
          (cons "'medium" (lambda () (set-float32-matmul-precision! 'medium)))
          (cons "(set-cuda-matmul-allow-tf32! #t)"
                (lambda () (set-cuda-matmul-allow-tf32! #t)))))

  (define (from-defaults setting! witness)
    (set-switches! default-switches)
    (setting!)
    (distinct-values (witness 'cuda)))

  (define (seeded-pair m k n)
    (manual-seed! 0)
    (values (randn m k #:device 'cuda) (randn k n #:device 'cuda)))

  (define (worst-ratio a b)
    (define a64 (to-dtype a 'float64))
    (define b64 (to-dtype b 'float64))
    (define e (abs (- (to-dtype (@ a b) 'float64) (@ a64 b64))))
    (~> (/ e (@ (abs a64) (abs b64))) max item))

  (define (under precision thunk)
    (set-float32-matmul-precision! precision)
    (thunk))

  (define (precision-tests hardware)
    (define tf32-on (if hardware '(1024.0) '(1024.25)))
    (define conv-on (if hardware '(576.0) '(576.140625)))
    (define tf32-half? (not (eq? hardware 'unknown)))

    (bounded-switches-test "C7: a float32 matmul rounds to TF32 only when allowed"
      #:seconds 120
      (for ([cudnn-tf32? (in-list '(#t #f))])
        (check-equal? (from-defaults (lambda () (set-cudnn-allow-tf32! cudnn-tf32?))
                                     matmul-witness)
                      '(1024.25)
                      (format "'highest, cuDNN TF32 ~a" cudnn-tf32?)))
      (when tf32-half?
        (for ([(label setting!) (in-dict (matmul-tf32-settings))])
          (check-equal? (from-defaults setting! matmul-witness) tf32-on
                        (format "~a, tf32-hardware? ~a" label hardware)))))

    (bounded-switches-test "C8: a cuDNN convolution follows cuDNN's flag, not the precision"
      #:seconds 120
      (check-equal? (from-defaults (lambda () (set-cudnn-allow-tf32! #f)) conv-witness)
                    '(576.140625)
                    "cuDNN TF32 off")
      (check-equal? (from-defaults (lambda ()
                                     (set-cudnn-allow-tf32! #f)
                                     (set-float32-matmul-precision! 'high))
                                   conv-witness)
                    '(576.140625)
                    "'high, cuDNN TF32 off")
      (when tf32-half?
        (check-equal? (from-defaults void conv-witness) conv-on
                      (format "the defaults, tf32-hardware? ~a" hardware))
        (check-equal? (from-defaults (lambda () (set-float32-matmul-precision! 'highest))
                                     conv-witness)
                      conv-on
                      (format "'highest, cuDNN TF32 on, tf32-hardware? ~a" hardware))))

    (bounded-switches-test "C9: the matmul's error stays within the precision's bound"
      #:seconds 120
      (define-values (a b) (seeded-pair 4096 2048 1024))
      (define k 2048)
      (define highest-ratio (under 'highest (lambda () (worst-ratio a b))))
      (define high-ratio (under 'high (lambda () (worst-ratio a b))))
      (check-true (<= highest-ratio (* k (expt 2.0 -23)))
                  (format "'highest: max E/S is ~a" highest-ratio))
      (check-true (<= high-ratio (+ (expt 2.0 -9) (* k (expt 2.0 -23))))
                  (format "'high: max E/S is ~a" high-ratio))
      (when (eq? hardware #t)
        (check-not-equal? (under 'high (lambda () (bytes-of (@ a b))))
                          (under 'highest (lambda () (bytes-of (@ a b))))
                          "TF32 hardware: 'high changes the bytes")))

    (bounded-switches-test "C10: on CUDA 'medium, 'high and the matmul flag are one setting"
      #:seconds 120
      (define-values (a b) (seeded-pair 4096 4096 4096))
      (define (product setting!)
        (set-switches! default-switches)
        (setting!)
        (bytes-of (@ a b)))
      (define reads (map product (map cdr (matmul-tf32-settings))))
      (check-equal? (remove-duplicates reads) (list (car reads))))

    (bounded-switches-test "C11: under 'high a seeded matmul and its backward repeat"
      #:seconds 120
      (set-float32-matmul-precision! 'high)
      (for ([on? (in-list '(#f #t))])
        (use-deterministic-algorithms! on?)
        (check-equal? (distinct-runs 10 matmul-run) 1
                      (format "use-deterministic-algorithms! ~a" on?)))))

  (define cuda? (cuda-available?))
  (define cublas-config (getenv "CUBLAS_WORKSPACE_CONFIG"))

  (cond
    [(not cuda?)
     (displayln "[switches-cuda-test] skipped: needs CUDA (run inside `nix develop .#cuda`)")]
    [cublas-config
     (cuda-tests)
     (precision-tests (tf32-hardware?))
     (displayln "[switches-cuda-test] skipped C6: CUBLAS_WORKSPACE_CONFIG is set")]
    [else
     (cuda-tests)
     (precision-tests (tf32-hardware?))
     (matmul-test)]))
