#lang racket/base

(module+ test
  (require (for-syntax racket/base (only-in syntax/parse/pre expr id))
           rackunit
           (only-in racket/math infinite? nan?)
           (only-in racket/sequence sequence->list)
           (only-in syntax/parse/define define-syntax-parse-rule)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    cpu-device cuda-available? cuda-device current-generator
                    generator-state in-flattened-tensor make-generator
                    manual-seed! mps-available? mps-device normal! rng-state
                    shape tensor->bytes to uniform! with-default-device zeros)
           (only-in "../nn.rkt"
                    Conv2d Linear kaiming-uniform normal-init parameters
                    trunc-normal-init uniform-init)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/within.rkt" bounded-test))

  (define devices
    (append (list (cpu-device))
            (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (host t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

  (define (filled on fill!)
    (define t (zeros 3 4 #:device on))
    (fill! t)
    (host t))

  (define-syntax-parse-rule (fill-row on:expr label:expr (fill!:id arg ...))
    (cons label
          (case-lambda
            [() (filled on (lambda (t) (fill! t arg ...)))]
            [(g) (filled on (lambda (t) (fill! t arg ... #:generator g)))])))

  (define-syntax-parse-rule (init-row on:expr label:expr (init:id arg ...))
    (cons label
          (case-lambda
            [() (with-default-device on (host (init arg ...)))]
            [(g) (with-default-device on (host (init arg ... #:generator g)))])))

  (define (draws on)
    (list (fill-row on "uniform! -0.5 0.5" (uniform! -0.5 0.5))
          (fill-row on "uniform!" (uniform!))
          (fill-row on "normal! 0.5 2.0" (normal! 0.5 2.0))
          (init-row on "uniform-init" (uniform-init '(2 3) -1 2))
          (init-row on "kaiming-uniform" (kaiming-uniform '(5 4)))
          (init-row on "normal-init" (normal-init '(3 4)))
          (init-row on "trunc-normal-init" (trunc-normal-init '(64 32) #:std 0.02))
          (init-row on "trunc-normal-init's uniform branch"
                    (trunc-normal-init '(64 32) #:std 0.02 #:a 0.03 #:b 0.05))))

  (define (built on make-layer)
    (with-default-device on
      (parameterize ([current-generator (make-generator 3 #:device on)])
        (map host (parameters (make-layer))))))

  (define (state-bytes on)
    (tensor->bytes (rng-state #:device on)))

  (bounded-test "G1: a seeded generator replays every draw on every device"
    (for* ([on (in-list devices)] [row (in-list (draws on))])
      (define d (cdr row))
      (check-equal? (d (make-generator 7 #:device on))
                    (d (make-generator 7 #:device on))
                    (format "~a: ~a" on (car row))))
    (for* ([on (in-list devices)]
           [make-layer (in-list (list (lambda () (Linear 4 3))
                                      (lambda () (Conv2d 1 8 3))))])
      (check-equal? (built on make-layer) (built on make-layer) (format "~a" on))))

  (bounded-test "G2: an explicit generator wins and leaves current-generator's alone"
    (for* ([on (in-list devices)] [row (in-list (draws on))])
      (define d (cdr row))
      (define h (make-generator 99 #:device on))
      (define before (tensor->bytes (generator-state h)))
      (define label (format "~a: ~a" on (car row)))
      (check-equal? (parameterize ([current-generator h])
                      (d (make-generator 7 #:device on)))
                    (d (make-generator 7 #:device on))
                    label)
      (check-equal? (tensor->bytes (generator-state h)) before label)))

  (bounded-test "G3: drawing from a generator leaves every default stream alone"
    (manual-seed! 5)
    (define before (map state-bytes devices))
    (for* ([on (in-list devices)] [row (in-list (draws on))])
      ((cdr row) (make-generator 7 #:device on)))
    (check-equal? (map state-bytes devices) before))

  (bounded-test "G4: without a generator a draw takes the default stream"
    (for* ([on (in-list devices)] [row (in-list (draws on))])
      (define d (cdr row))
      (define label (format "~a: ~a" on (car row)))
      (define seeded (d (make-generator 7 #:device on)))
      (manual-seed! 7)
      (check-equal? (d #f) seeded label)
      (manual-seed! 7)
      (check-equal? (d) seeded (format "~a, omitted" label))))

  (define caller "init-generator-test")

  (define (filled-with dtype fill!)
    (define t (zeros 3 #:dtype dtype))
    (fill! t)
    (host t))

  (bounded-test "G6: arguments libtorch refuses blame the caller"
    (define refused
      (list (lambda () (uniform! (zeros 3 #:dtype 'int64)))
            (lambda () (uniform! (zeros 3 #:dtype 'uint8)))
            (lambda () (uniform! (zeros 3 #:dtype 'bool)))
            (lambda () (normal! (zeros 3 #:dtype 'int64)))
            (lambda () (uniform! (zeros 3) 1 0))
            (lambda () (uniform! (zeros 3) +nan.0 0))
            (lambda () (uniform! (zeros 3) 0 +inf.0))
            (lambda () (uniform! (zeros 3 #:dtype 'float16) 0 65505))
            (lambda () (uniform! (zeros 3) -3e38 3e38))
            (lambda () (normal! (zeros 3) 0 -1))
            (lambda () (normal! (zeros 3) 0 +nan.0))
            (lambda () (uniform-init '(3) 1 0))
            (lambda ()
              (trunc-normal-init '(3) #:a 3.3e38 #:b 3.5e38 #:mean 0.0))))
    (for ([thunk (in-list refused)] [k (in-naturals)])
      (check-exn (blames-caller? caller) thunk (format "refused row ~a" k))))

  (bounded-test "G6: arguments libtorch accepts are accepted"
    (check-true (for/and ([v (in-list (filled-with 'float16
                                                   (lambda (t) (uniform! t 0 65504))))])
                  (<= 0 v 65504)))
    (check-true (for/and ([v (in-list (filled-with 'bfloat16
                                                   (lambda (t) (uniform! t 0 3e38))))])
                  (<= 0 v 3e38)))
    (check-equal? (filled-with 'float32 (lambda (t) (uniform! t 0.25 0.25)))
                  '(0.25 0.25 0.25))
    (check-equal? (filled-with 'float32 (lambda (t) (normal! t 1.5 0)))
                  '(1.5 1.5 1.5))
    (check-true (andmap infinite?
                        (filled-with 'float32 (lambda (t) (normal! t 0 +inf.0)))))
    (check-true (andmap nan?
                        (filled-with 'float32 (lambda (t) (normal! t +nan.0 1)))))
    (check-true (for/and ([v (in-list (host (trunc-normal-init '(3) #:a 3.3e38
                                                               #:b 3.4e38
                                                               #:mean 0.0)))])
                  (<= 3.29e38 v 3.41e38))))

  (bounded-test "G10: kaiming-uniform needs two dimensions and draws nothing for zero fan"
    (check-exn (blames-caller? caller) (lambda () (kaiming-uniform '(3))))
    (define g (make-generator 4))
    (define before (tensor->bytes (generator-state g)))
    (define w (kaiming-uniform '(3 0) #:generator g))
    (check-equal? (shape w) '(3 0))
    (check-equal? (tensor->bytes (generator-state g)) before)
    (check-equal? (host (kaiming-uniform '(0 4) #:generator g)) '())
    (check-equal? (tensor->bytes (generator-state g)) before)))
