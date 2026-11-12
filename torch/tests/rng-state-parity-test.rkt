#lang racket/base

(module+ test
  (require rackunit
           (only-in file/sha1 bytes->hex-string)
           (only-in racket/sequence sequence->list)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    cuda-available? current-generator
                    generator-state in-flattened-tensor make-generator
                    manual-seed! rand randn rng-state set-generator-state!
                    set-rng-state! tensor->bytes to)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available? unpack)
           (only-in "private/within.rkt" bounded-test))

  (define (host t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

  (define (after on g)
    (parameterize ([current-generator g])
      (list (host (rand 5 #:device on))
            (host (randn 3 2 #:device on)))))

  (define (python-after j)
    (define a (hash-ref j 'after))
    (list (hash-ref a 'rand) (hash-ref a 'randn)))

  (define (twin on stream [state #f])
    (call-with-python-env
     #:env (append (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string on))
                         (cons "RKTORCH_RNG_STREAM" (symbol->string stream)))
                   (if state
                       (list (cons "RKTORCH_RNG_STATE"
                                   (bytes->hex-string (tensor->bytes state))))
                       '()))
     (lambda () (python-check "rng_state.py"))))

  (define (seeded on stream seed)
    (define g (and (eq? stream 'generator) (make-generator seed #:device on)))
    (unless g (manual-seed! seed))
    (parameterize ([current-generator g])
      (rand 5 #:device on)
      (randn 3 #:device on))
    g)

  (define (state-of on g)
    (if g (generator-state g) (rng-state #:device on)))

  (define (parity-rows on)
    (for ([stream (in-list '(generator default))])
      (define label (format "~a ~a" on stream))
      (define j (twin on stream))
      (define python-state (unpack (hash-ref j 'state) 'uint8))
      (define g (seeded on stream 7))
      (check-equal? (tensor->bytes (state-of on g)) (tensor->bytes python-state)
                    (format "P1 ~a" label))
      (define fresh (and (eq? stream 'generator) (make-generator 0 #:device on)))
      (if fresh
          (set-generator-state! fresh python-state)
          (set-rng-state! python-state #:device on))
      (check-equal? (after on fresh) (python-after j) (format "P2 ~a" label))
      (define ours (seeded on stream 11))
      (define sent (state-of on ours))
      (check-equal? (python-after (twin on stream sent)) (after on ours)
                    (format "P3 ~a" label))))

  (cond
    [(python-torch-available?)
     (bounded-test "P1-P3: CPU states pass between rktorch and PyTorch"
       (parity-rows 'cpu))]
    [else
     (displayln "[rng-state-parity-test] skipped the CPU rows: needs python3 `torch`")])

  (cond
    [(and (cuda-available?) (python-torch-available?) (python-cuda-available?))
     (bounded-test "P1-P3: CUDA states pass between rktorch and PyTorch"
       (parity-rows 'cuda))]
    [else
     (printf "[rng-state-parity-test] skipped the CUDA rows: needs CUDA and ~a\n"
             "python3 `torch` with CUDA (run inside `nix develop .#cuda`)")]))
