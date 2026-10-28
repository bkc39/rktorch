#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/match match-define)
           (only-in racket/sequence sequence->list)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    cuda-available? cuda-device current-generator
                    in-flattened-tensor make-generator manual-seed! to uniform!
                    with-default-device zeros)
           (only-in "../nn.rkt" trunc-normal-init)
           (only-in "../private/erf.rkt" erf)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define (flat t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

  (define (seeded thunk)
    (manual-seed! 3)
    (thunk))

  (define edge-a 0.5173725371405112)
  (define edge-b 2.813789560067375)

  (define cases
    (list (cons 'std (lambda () (trunc-normal-init '(64 32) #:std 0.02)))
          (cons 'bounds
                (lambda ()
                  (trunc-normal-init '(64 32) #:mean 0.01 #:std 0.02
                                     #:a -0.01 #:b 0.03)))
          (cons 'tail
                (lambda ()
                  (trunc-normal-init '(64 32) #:std 0.02 #:a 0.03 #:b 0.05)))
          (cons 'edge
                (lambda () (trunc-normal-init '(257) #:a edge-a #:b edge-b)))))

  (define (check-parity j)
    (for ([named (in-list cases)])
      (match-define (cons key run) named)
      (check-equal? (flat (seeded run)) (hash-ref j key) (format "~a" key))))

  (cond
    [(python-torch-available?)
     (bounded-test "I1: trunc-normal-init draws trunc_normal_'s values, bit for bit"
       (check-parity (python-check "trunc_normal.py")))]
    [else
     (printf "[trunc-normal-test] skipped I1: python3 `torch` ~a\n"
             "not available (run inside `nix develop`)")])

  (cond
    [(and (cuda-available?)
          (python-torch-available?)
          (python-cuda-available?))
     (bounded-test "I1: on CUDA too, bit for bit"
       (define j
         (call-with-python-env
          #:env (list (cons "RKTORCH_PARITY_DEVICE" "cuda"))
          (lambda () (python-check "trunc_normal.py"))))
       (with-default-device (cuda-device)
         (check-parity j)))]
    [else
     (printf "[trunc-normal-test] skipped I1 on CUDA: needs CUDA and ~a\n"
             "python3 `torch` with CUDA (run inside `nix develop .#cuda`)")])

  (bounded-test "erf is the correctly rounded one, as math.erf"
    (for ([known (in-list '((1e-300 1.1283791670955126e-300)
                            (1e-08 1.1283791670955126e-08)
                            (0.25 0.27632639016823696)
                            (0.5 0.5204998778130465)
                            (0.9999 0.8426592780487595)
                            (1.0 0.8427007929497149)
                            (1.4142135623730951 0.9544997361036416)
                            (1.9 0.9927904292352575)
                            (2.0 0.9953222650189527)
                            (2.5 0.999593047982555)
                            (3.5 0.9999992569016276)
                            (5.0 0.9999999999984626)
                            (5.9 0.9999999999999999)
                            (5.93 1.0)
                            (-1.7 -0.9837904585907745)
                            (0.36583762941174436 0.3951038731407312)
                            (1.989649678755553 0.9951038731407311)))])
      (match-define (list x expected) known)
      (check-equal? (erf x) expected (format "erf ~a" x)))
    (check-equal? (erf -0.0) -0.0)
    (check-true (eqv? +nan.0 (erf +nan.0)))
    (check-equal? (erf +inf.0) 1.0))

  (bounded-test "the 0.3 choice falls as PyTorch's: mass 0.29999999999999993"
    (define (cdf v)
      (~> v (/ (sqrt 2.0)) erf (+ 1.0) (/ 2.0)))
    (check-equal? (- (cdf edge-b) (cdf edge-a)) 0.29999999999999993))

  (bounded-test "both branches draw from current-generator, and replay"
    (define (under seed run)
      (seeded (lambda ()
                (parameterize ([current-generator (make-generator seed)])
                  (flat (run))))))
    (for ([run (in-list (list (lambda () (trunc-normal-init '(4) #:a 1.0 #:b 1.5))
                              (lambda () (trunc-normal-init '(4)))
                              (lambda ()
                                (define t (zeros 4))
                                (uniform! t 0.0 1.0)
                                t)))]
          [label (in-list '(uniform normal uniform!))])
      (define global (flat (seeded run)))
      (check-not-equal? (under 5 run) global (format "~a" label))
      (check-equal? (under 5 run) (under 5 run) (format "~a" label))
      (check-not-equal? (under 6 run) (under 5 run) (format "~a" label))))

  (bounded-test "every value lies in [a, b]"
    (for ([bounds (in-list '((-0.01 0.03) (-2.0 2.0) (0 1/2) (-inf.0 0.5)
                             (-1.0 +inf.0)))])
      (match-define (list a b) bounds)
      (define t (seeded (lambda ()
                          (trunc-normal-init '(16 8) #:mean 0.01 #:std 0.5
                                             #:a a #:b b))))
      (for ([v (in-list (flat t))])
        (check-true (<= a v b) (format "~a in ~a" v bounds)))))

  (bounded-test "trunc-normal-init blames a caller whose #:a is not below #:b"
    (check-exn (blames-caller? "trunc-normal-test")
               (lambda () (trunc-normal-init '(2 2) #:a 1.0 #:b 1.0))))

  (bounded-test "trunc-normal-init blames an infinite bound on the uniform branch"
    (check-exn (blames-caller? "trunc-normal-test")
               (lambda () (trunc-normal-init '(3) #:a -inf.0 #:b -5.0)))
    (check-exn (blames-caller? "trunc-normal-test")
               (lambda () (trunc-normal-init '(3) #:a 5.0 #:b +inf.0)))))
