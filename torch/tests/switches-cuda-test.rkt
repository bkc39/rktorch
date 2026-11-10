#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list remove-duplicates)
           (only-in "../main.rkt"
                    * @ ~> adaptive-avg-pool2d backward! conv2d cuda-available?
                    deterministic-algorithms-enabled? detach exn:fail:rktorch:oom?
                    grad index-add! index-select manual-seed! mean rand randn
                    requires-grad! reshape set-cudnn-benchmark! set-cudnn-deterministic! sub!
                    sum tensor->bytes to to-dtype use-deterministic-algorithms!
                    with-no-grad zeros)
           (only-in "private/host.rkt" host)
           (only-in "private/switches.rkt"
                    default-switches switch-values with-switches-restored)
           (only-in "private/within.rkt" bounded-test))

  (define (bytes-of t)
    (tensor->bytes (to t 'cpu)))

  (define (distinct-runs n run)
    (length (remove-duplicates (for/list ([_ (in-range n)]) (run)))))

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

  (cond
    [(cuda-available?)
     (check-equal? (switch-values) default-switches
                   "no flag leaked from an earlier file")

     (bounded-test "C1: the switch refuses a CUDA backward with no deterministic kernel"
       (with-switches-restored
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
         (check-not-exn (lambda () (pooled-backward! x)))))

     (bounded-test "C2: warn-only runs the nondeterministic kernel"
       (with-switches-restored
         (use-deterministic-algorithms! #t #:warn-only? #t)
         (define x-cpu (randn 1 2 5 5 #:requires-grad? #t))
         (define x-cuda (~> x-cpu detach (to 'cuda) requires-grad!))
         (pooled-backward! x-cuda)
         (pooled-backward! x-cpu)
         (for ([g (in-list (host (grad x-cuda)))]
               [want (in-list (host (grad x-cpu)))])
           (check-within g want 1e-6))))

     (bounded-test "C3: index-add! repeats bit for bit only under the switch"
       (with-switches-restored
         (define off (index-add-run))
         (check-true (< 1 (distinct-runs 20 off))
                     "the witness: runs differ with the switch off")
         (use-deterministic-algorithms! #t)
         (check-equal? (distinct-runs 20 (index-add-run)) 1)))

     (bounded-test "C4: a seeded step repeats bit for bit only under the switch"
       (with-switches-restored
         (check-true (< 1 (distinct-runs 10 seeded-step))
                     "the witness: runs differ with the switch off")
         (use-deterministic-algorithms! #t)
         (check-equal? (distinct-runs 10 seeded-step) 1)))

     (bounded-test "C5: cuDNN's flag makes a convolution's weight gradient repeat"
       (with-switches-restored
         (check-true (< 1 (distinct-runs 10 conv-weight-grad))
                     "the witness: runs differ with the flag off")
         (set-cudnn-deterministic! #t)
         (check-equal? (distinct-runs 10 conv-weight-grad) 1)
         (set-cudnn-benchmark! #t)
         (check-equal? (distinct-runs 10 conv-weight-grad) 1
                       "also with benchmark on")))

     (cond
       [(getenv "CUBLAS_WORKSPACE_CONFIG")
        (displayln "[switches-cuda-test] skipped C6: CUBLAS_WORKSPACE_CONFIG is set")]
       [else
        (bounded-test "C6: without CUBLAS_WORKSPACE_CONFIG a matmul repeats under the switch"
          (with-switches-restored
            (use-deterministic-algorithms! #t)
            (check-equal? (distinct-runs 10 matmul-run) 1)))])]
    [else
     (displayln "[switches-cuda-test] skipped: needs CUDA (run inside `nix develop .#cuda`)")]))
