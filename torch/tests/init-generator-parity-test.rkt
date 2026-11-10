#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/match match-define)
           (only-in racket/sequence sequence->list)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    cuda-available? current-generator in-flattened-tensor
                    make-generator normal! rand to uniform! with-default-device
                    zeros)
           (only-in "../nn.rkt"
                    Linear kaiming-uniform normal-init parameters
                    trunc-normal-init uniform-init)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define (host t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

  (define ((fill dims dtype fill!) on g)
    (define t (apply zeros dims #:device on #:dtype dtype))
    (fill! t g)
    (list t))

  (define ((init draw) on g)
    (with-default-device on (draw g)))

  (define cases
    (list
     (list 'uniform 1
           (fill '(4 3) 'float32
                 (lambda (t g) (uniform! t -0.5 0.5 #:generator g))))
     (list 'uniform-defaults 2
           (fill '(5) 'float32 (lambda (t g) (uniform! t #:generator g))))
     (list 'uniform-half 3
           (fill '(6) 'float16 (lambda (t g) (uniform! t 0 65504 #:generator g))))
     (list 'uniform-point 4
           (fill '(5) 'float32
                 (lambda (t g) (uniform! t 0.25 0.25 #:generator g))))
     (list 'normal 5
           (fill '(7) 'float32 (lambda (t g) (normal! t 0.5 2.0 #:generator g))))
     (list 'normal-double 6
           (fill '(5) 'float64 (lambda (t g) (normal! t #:generator g))))
     (list 'normal-zero-std 7
           (fill '(5) 'float32 (lambda (t g) (normal! t 1.0 0.0 #:generator g))))
     (list 'uniform-init 8
           (init (lambda (g) (list (uniform-init '(2 3) -1.0 2.0 #:generator g)))))
     (list 'kaiming 9
           (init (lambda (g) (list (kaiming-uniform '(5 4) #:generator g)))))
     (list 'kaiming-a0 10
           (init (lambda (g)
                   (list (kaiming-uniform '(3 2 3 3) #:a 0.0 #:generator g)))))
     (list 'normal-init 11
           (init (lambda (g) (list (normal-init '(3 4) #:generator g)))))
     (list 'trunc 12
           (init (lambda (g)
                   (list (trunc-normal-init '(64 32) #:std 0.02 #:generator g)))))
     (list 'trunc-tail 13
           (init (lambda (g)
                   (list (trunc-normal-init '(64 32) #:std 0.02 #:a 0.03 #:b 0.05
                                            #:generator g)))))
     (list 'linear 14
           (init (lambda (g)
                   (parameterize ([current-generator g])
                     (parameters (Linear 4 3))))))))

  (define (ours on seed draw)
    (define g (make-generator seed #:device on))
    (define drawn (draw on g))
    (list (map host drawn) (host (rand 3 #:device on #:generator g))))

  (define (parity-rows on)
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string on)))
       (lambda () (python-check "init_generator.py"))))
    (for ([c (in-list cases)])
      (match-define (list key seed draw) c)
      (define theirs (hash-ref j key))
      (check-equal? (ours on seed draw)
                    (list (hash-ref theirs 'drawn) (hash-ref theirs 'after))
                    (format "G9 ~a ~a" on key))))

  (cond
    [(python-torch-available?)
     (bounded-test "G9: an explicit generator draws as PyTorch's on the CPU"
       (parity-rows 'cpu))]
    [else
     (displayln "[init-generator-parity-test] skipped the CPU rows: needs python3 `torch`")])

  (cond
    [(and (cuda-available?) (python-torch-available?) (python-cuda-available?))
     (bounded-test "G9: an explicit generator draws as PyTorch's on CUDA"
       (parity-rows 'cuda))]
    [else
     (printf "[init-generator-parity-test] skipped the CUDA rows: needs CUDA and ~a\n"
             "python3 `torch` with CUDA (run inside `nix develop .#cuda`)")]))
