#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/dict in-dict)
           (only-in racket/match match-define)
           (only-in "../main.rkt"
                    cuda-available? manual-seed! use-deterministic-algorithms!)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?)
           (only-in "private/switches.rkt"
                    determinism-rows switch-values with-switches-restored)
           (only-in "private/within.rkt" bounded-test))

  (define ours-at-start (switch-values))

  (define kernel-rx #rx"([a-z0-9_]+) does not have a deterministic implementation")

  (define (refusal? e)
    (and (exn:fail? e) (regexp-match? kernel-rx (exn-message e))))

  (define (refused e)
    (match-define (list _ kernel) (regexp-match kernel-rx (exn-message e)))
    (string-append "raise " kernel))

  (define (outcome run on)
    (with-handlers ([refusal? refused])
      (run on)
      "ok"))

  (define (outcomes on #:warn-only? warn-only?)
    (with-switches-restored
      (use-deterministic-algorithms! #t #:warn-only? warn-only?)
      (manual-seed! 0)
      (for/hash ([(name run) (in-dict determinism-rows)])
        (values name (outcome run on)))))

  (define (twin on)
    (call-with-python-env
     #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string on)))
     (lambda () (python-check "switches.py"))))

  (define (as-hash j)
    (for/hash ([(k v) (in-hash j)])
      (values k v)))

  (define (warned->ok s)
    (if (regexp-match? #rx"^(ok|warn)" s) "ok" s))

  (cond
    [(python-torch-available?)
     (bounded-test "P1: PyTorch's fresh defaults are rktorch's"
       (check-equal? ours-at-start (as-hash (hash-ref (twin 'cpu) 'defaults))))
     (bounded-test "P3: no row refuses or warns on the CPU, in either"
       (define j (twin 'cpu))
       (for ([key (in-list '(strict warn_only))])
         (for ([(name o) (in-hash (hash-ref j key))])
           (check-equal? o "ok" (format "PyTorch ~a ~a" key name))))
       (for ([warn-only? (in-list '(#f #t))])
         (for ([(name o) (in-hash (outcomes 'cpu #:warn-only? warn-only?))])
           (check-equal? o "ok" (format "rktorch warn-only ~a ~a" warn-only? name)))))]
    [else
     (displayln "[switches-parity-test] skipped the CPU rows: needs python3 `torch`")])

  (cond
    [(and (cuda-available?) (python-torch-available?) (python-cuda-available?))
     (bounded-test "P2: on CUDA rktorch refuses exactly where PyTorch does"
       (define j (twin 'cuda))
       (define strict (as-hash (hash-ref j 'strict)))
       (define warned (as-hash (hash-ref j 'warn_only)))
       (check-equal? (outcomes 'cuda #:warn-only? #f) strict)
       (check-equal? (outcomes 'cuda #:warn-only? #t)
                     (for/hash ([(name o) (in-hash warned)])
                       (values name (warned->ok o)))))]
    [else
     (printf "[switches-parity-test] skipped the CUDA rows: needs CUDA and ~a\n"
             "python3 `torch` with CUDA (run inside `nix develop .#cuda`)")]))
