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
                    bounded-switches-test default-switches determinism-rows
                    switch-values))

  (check-equal? (switch-values) default-switches
                "no flag leaked from an earlier file")

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
    (use-deterministic-algorithms! #t #:warn-only? warn-only?)
    (manual-seed! 0)
    (for/hasheq ([(name run) (in-dict determinism-rows)])
      (values name (outcome run on))))

  (define (twin on)
    (call-with-python-env
     #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string on)))
     (lambda () (python-check "switches.py"))))

  (define (warned->ok s)
    (if (regexp-match? #rx"^(ok|warn)" s) "ok" s))

  (define (cpu-tests)
    (bounded-switches-test "P1: PyTorch's fresh defaults are rktorch's"
      (check-equal? (switch-values) (hash-ref (twin 'cpu) 'defaults)))
    (bounded-switches-test "P3: no row refuses or warns on the CPU, in either"
      (define j (twin 'cpu))
      (for* ([key (in-list '(strict warn_only))]
             [(name o) (in-hash (hash-ref j key))])
        (check-equal? o "ok" (format "PyTorch ~a ~a" key name)))
      (for* ([warn-only? (in-list '(#f #t))]
             [(name o) (in-hash (outcomes 'cpu #:warn-only? warn-only?))])
        (check-equal? o "ok" (format "rktorch warn-only ~a ~a" warn-only? name)))))

  (define (cuda-tests)
    (bounded-switches-test "P2: on CUDA rktorch refuses exactly where PyTorch does"
      (define j (twin 'cuda))
      (check-equal? (outcomes 'cuda #:warn-only? #f) (hash-ref j 'strict))
      (check-equal? (outcomes 'cuda #:warn-only? #t)
                    (for/hasheq ([(name o) (in-hash (hash-ref j 'warn_only))])
                      (values name (warned->ok o))))))

  (define python-torch? (python-torch-available?))
  (define cuda-twins? (and python-torch? (cuda-available?) (python-cuda-available?)))

  (cond
    [(not python-torch?)
     (displayln "[switches-parity-test] skipped: needs python3 `torch`")]
    [(not cuda-twins?)
     (cpu-tests)
     (printf "[switches-parity-test] skipped the CUDA rows: needs CUDA and ~a\n"
             "python3 `torch` with CUDA (run inside `nix develop .#cuda`)")]
    [else
     (cpu-tests)
     (cuda-tests)]))
