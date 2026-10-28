#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/match match-define)
           (only-in racket/sequence sequence->list)
           (only-in threading ~>)
           "../data/loader.rkt"
           (only-in "../main.rkt"
                    arange cpu-device cpu-generator/c cuda-available?
                    cuda-device current-generator device draw-seed
                    generator-device in-flattened-tensor make-generator
                    mps-available? mps-device multinomial rand rand-like randn
                    randn-like randperm reshape tensor to uniform! zeros)
           (only-in "../nn.rkt" kaiming-uniform trunc-normal-init uniform-init)
           (only-in "../vision/transforms.rkt"
                    random-crop random-horizontal-flip random-resized-crop)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/op-calls.rkt" op-calls-of)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define accelerators
    (append (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (host t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

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

  (bounded-test "rand without a generator crosses into libtorch as zeros does"
    (define (rand-4) (rand 4))
    (define (zeros-4) (zeros 4))
    (rand-4)
    (zeros-4)
    (check-equal? (op-calls-of rand-4) (op-calls-of zeros-4)))

  (define (blamed-on accelerator)
    (define elsewhere (make-generator 1 #:device accelerator))
    (define probabilities (to (tensor '(0.1 0.2 0.3 0.4)) accelerator))
    (define images (~> (* 2 3 6 6) arange (reshape 2 3 6 6)))
    (list (cons "rand on the accelerator, cpu generator"
                (lambda ()
                  (rand 3 #:device accelerator #:generator (make-generator 1))))
          (cons "randn on cpu, accelerator generator"
                (lambda () (randn 3 #:device 'cpu #:generator elsewhere)))
          (cons "rand on cpu through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (rand 3 #:device 'cpu))))
          (cons "rand-like on the accelerator, cpu generator"
                (lambda ()
                  (rand-like (to (zeros 3) accelerator)
                             #:generator (make-generator 1))))
          (cons "randn-like on cpu, accelerator generator"
                (lambda () (randn-like (zeros 3) #:generator elsewhere)))
          (cons "rand-like moved to the accelerator, cpu generator"
                (lambda ()
                  (rand-like (zeros 3) #:device accelerator
                             #:generator (make-generator 1))))
          (cons "rand-like through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (rand-like (zeros 3)))))
          (cons "uniform! through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (uniform! (zeros 3) 0.0 1.0))))
          (cons "uniform-init through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (uniform-init '(3) 0.0 1.0))))
          (cons "kaiming-uniform through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (kaiming-uniform '(3 2)))))
          (cons "trunc-normal-init through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (trunc-normal-init '(3)))))
          (cons "trunc-normal-init's uniform branch through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (trunc-normal-init '(3) #:a 1.0 #:b 1.5))))
          (cons "randperm, accelerator generator"
                (lambda () (randperm 4 #:generator elsewhere)))
          (cons "randperm through current-generator"
                (lambda ()
                  (parameterize ([current-generator elsewhere])
                    (randperm 4))))
          (cons "dataloader, accelerator generator"
                (lambda ()
                  (~> 4
                      arange
                      tensor-dataset
                      (dataloader #:shuffle? #t #:generator elsewhere))))
          (cons "random-crop, accelerator generator"
                (lambda () (random-crop images #:generator elsewhere)))
          (cons "random-horizontal-flip, accelerator generator"
                (lambda ()
                  (random-horizontal-flip images #:generator elsewhere)))
          (cons "random-resized-crop, accelerator generator"
                (lambda ()
                  (random-resized-crop images 4 #:generator elsewhere)))
          (cons "multinomial on the accelerator, cpu generator"
                (lambda ()
                  (multinomial probabilities 2
                               #:generator (make-generator 1))))))

  (bounded-test "R4: a CPU generator drawing for CUDA blames the caller, CUDA or not"
    (for ([thunk (in-list
                  (list (lambda ()
                          (rand 3 #:device 'cuda #:generator (make-generator 1)))
                        (lambda ()
                          (rand-like (zeros 3) #:device 'cuda
                                     #:generator (make-generator 1)))))])
      (check-exn (blames-caller? "generator-device-test") thunk)))

  (cond
    [(pair? accelerators)
     (bounded-test "R4: a generator on another device blames the caller"
       (for* ([accelerator (in-list accelerators)]
              [row (in-list (blamed-on accelerator))])
         (match-define (cons label thunk) row)
         (check-exn (blames-caller? "generator-device-test") thunk
                    (format "~a: ~a" accelerator label))))
     (bounded-test "R4: rand-like moved to the CPU takes a CPU generator"
       (for ([accelerator (in-list accelerators)])
         (check-equal? (~> (zeros 3)
                           (to accelerator)
                           (rand-like #:device 'cpu
                                      #:generator (make-generator 1))
                           device)
                       (cpu-device))))]
    [else
     (printf "[generator-device-test] skipped R4 rows: needs an ~a\n"
             "accelerator (CUDA or MPS) for a generator on another device")])

  (cond
    [(and (cuda-available?)
          (python-torch-available?)
          (python-cuda-available?))
     (bounded-test "R2: a CUDA generator draws as torch.Generator('cuda')"
       (define j
         (call-with-python-env
          #:env (list (cons "RKTORCH_PARITY_DEVICE" "cuda"))
          (lambda () (python-check "generator_device.py"))))
       (check-equal? (three-draws (cuda-device)
                                  (make-generator 7 #:device (cuda-device)))
                     (list (hash-ref j 'rand)
                           (hash-ref j 'randn)
                           (hash-ref j 'rand64))))]
    [else
     (printf "[generator-device-test] skipped R2: needs CUDA and python3 ~a\n"
             "`torch` with CUDA (run inside `nix develop .#cuda`)")]))
