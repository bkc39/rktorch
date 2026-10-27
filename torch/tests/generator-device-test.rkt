#lang racket/base

;; R2's draws stay in the order of the twin,
;; torch/tests/python/generator_device.py.

(module+ test
  (require rackunit
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/match match-define)
           (only-in threading ~>)
           "../data/loader.rkt"
           (only-in "../main.rkt"
                    arange cpu-device cpu-generator/c cuda-available?
                    cuda-device current-generator device draw-seed
                    generator-device in-flattened-tensor make-generator
                    mps-available? mps-device multinomial rand randn randperm
                    reshape tensor to)
           (only-in "../vision/transforms.rkt"
                    random-crop random-horizontal-flip random-resized-crop)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define accelerators
    (append (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (blames-caller? e)
    (and (exn:fail:contract:blame? e)
         (regexp-match? #rx"blaming: [^\n]*generator-device-test"
                        (exn-message e))))

  (define (host t)
    (for/list ([v (in-flattened-tensor (to t 'cpu))]) v))

  (define (three-draws on g)
    (list (host (rand 5 #:device on #:generator g))
          (host (randn 3 2 #:device on #:generator g))
          (host (rand 4 #:device on #:dtype 'float64 #:generator g))))

  (bounded-test "R1: a generator is on the CPU unless placed"
    (check-equal? (generator-device (make-generator 1)) (cpu-device))
    (check-equal? (generator-device (make-generator 1 #:device 'cpu))
                  (cpu-device))
    (check-true (cpu-generator/c (make-generator 1)))
    (for ([on (in-list accelerators)])
      (define g (make-generator 7 #:device on))
      (check-equal? (generator-device g) on)
      (check-false (cpu-generator/c g))
      (check-equal? (device (rand 4 #:device on #:generator g)) on)))

  (unless (cuda-available?)
    (bounded-test "R1: a generator on a device that is not there is refused"
      (check-exn exn:fail? (lambda () (make-generator 1 #:device 'cuda)))))

  (bounded-test "R3: two generators with one seed on one device draw alike"
    (for ([on (in-list (cons (cpu-device) accelerators))])
      (check-equal? (three-draws on (make-generator 11 #:device on))
                    (three-draws on (make-generator 11 #:device on))
                    (format "~a" on))))

  (bounded-test "R3: draw-seed draws from a generator on any device"
    (for ([on (in-list (cons (cpu-device) accelerators))])
      (check-equal? (draw-seed #:generator (make-generator 3 #:device on))
                    (draw-seed #:generator (make-generator 3 #:device on))
                    (format "~a" on))))

  (bounded-test "R4: a CPU generator draws on the CPU, as current-generator too"
    (check-equal? (host (rand 3 #:generator (make-generator 2)))
                  (parameterize ([current-generator (make-generator 2)])
                    (host (rand 3 #:device 'cpu)))))

  (define (blamed-on-cuda gpu)
    (define on-gpu (make-generator 1 #:device gpu))
    (define probabilities (to (tensor '(0.1 0.2 0.3 0.4)) gpu))
    (define images (~> (arange (* 2 3 6 6)) (reshape 2 3 6 6)))
    (list (cons "rand on cuda, cpu generator"
                (lambda () (rand 3 #:device gpu #:generator (make-generator 1))))
          (cons "randn on cpu, cuda generator"
                (lambda () (randn 3 #:device 'cpu #:generator on-gpu)))
          (cons "rand on cpu through current-generator"
                (lambda ()
                  (parameterize ([current-generator on-gpu])
                    (rand 3 #:device 'cpu))))
          (cons "randperm, cuda generator"
                (lambda () (randperm 4 #:generator on-gpu)))
          (cons "dataloader, cuda generator"
                (lambda ()
                  (dataloader (tensor-dataset (arange 4)) #:shuffle? #t
                              #:generator on-gpu)))
          (cons "random-crop, cuda generator"
                (lambda () (random-crop images #:generator on-gpu)))
          (cons "random-horizontal-flip, cuda generator"
                (lambda () (random-horizontal-flip images #:generator on-gpu)))
          (cons "random-resized-crop, cuda generator"
                (lambda () (random-resized-crop images 4 #:generator on-gpu)))
          (cons "multinomial on cuda, cpu generator"
                (lambda ()
                  (multinomial probabilities 2
                               #:generator (make-generator 1))))))

  (when (cuda-available?)
    (bounded-test "R4: a generator on another device blames the caller"
      (for ([row (in-list (blamed-on-cuda (cuda-device)))])
        (match-define (cons label thunk) row)
        (check-exn blames-caller? thunk label)))

    (when (and (python-torch-available?) (python-cuda-available?))
      (bounded-test "R2: a CUDA generator draws as torch.Generator('cuda')"
        (define j
          (call-with-python-env
           #:env (list (cons "RKTORCH_PARITY_DEVICE" "cuda"))
           (lambda () (python-check "generator_device.py"))))
        (check-equal? (three-draws (cuda-device)
                                   (make-generator 7 #:device (cuda-device)))
                      (list (hash-ref j 'rand)
                            (hash-ref j 'randn)
                            (hash-ref j 'rand64)))))))
