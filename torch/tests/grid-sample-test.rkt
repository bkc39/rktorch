#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/contract exn:fail:contract:blame?)
           (only-in racket/match match-define)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * - affine-grid cuda-available? cuda-device flip grid-sample
                    make-generator rand randn reshape tensor tensor->list
                    tensor-shape to to-dtype upsample-bicubic2d)
           (only-in "private/gradcheck.rkt" gradcheck)
           (only-in "private/within.rkt" bounded-test))

  (define (identity-theta n dtype)
    (~> (for/list ([_ (in-range n)]) '((1.0 0.0 0.0) (0.0 1.0 0.0)))
        tensor
        (to-dtype dtype)))

  (define (mirror-theta n dtype)
    (~> (for/list ([_ (in-range n)]) '((-1.0 0.0 0.0) (0.0 1.0 0.0)))
        tensor
        (to-dtype dtype)))

  (define (linspace steps)
    (for/list ([i (in-range steps)])
      (- (/ (* 2.0 i) (sub1 steps)) 1.0)))

  (bounded-test "G1: the identity's aligned grid is the linspace mesh"
    (for ([dtype (in-list '(float32 float64))])
      (define grid (affine-grid (identity-theta 2 dtype) '(2 1 3 5)
                                #:align-corners? #t))
      (define mesh
        (for*/list ([y (in-list (linspace 3))]
                    [x (in-list (linspace 5))]
                    [v (in-list (list x y))])
          v))
      (check-equal? (tensor-shape grid) '(2 3 5 2))
      (check-equal? (tensor->list grid) (append mesh mesh) (format "~a" dtype))))

  (bounded-test "G2: nearest sampling of the identity and the mirror is exact"
    (for ([dtype (in-list '(float32 float64))])
      (define x (to-dtype (randn 2 3 4 5) dtype))
      (define (sampled theta)
        (grid-sample x (affine-grid theta '(2 3 4 5)) #:mode 'nearest))
      (check-equal? (tensor->list (sampled (identity-theta 2 dtype)))
                    (tensor->list x)
                    (format "identity ~a" dtype))
      (check-equal? (tensor->list (sampled (mirror-theta 2 dtype)))
                    (tensor->list (flip x 3))
                    (format "mirror ~a" dtype))))

  (define (inner-grid . shape)
    (~> (rand shape #:dtype 'float64 #:generator (make-generator 5))
        (* 1.8)
        (- 0.9)))

  (bounded-test "G3: grid-sample's gradients match finite differences"
    (for* ([mode (in-list '(bilinear bicubic))]
           [padding (in-list '(zeros border reflection))])
      (check-true
       (gradcheck (lambda (x grid)
                    (grid-sample x grid #:mode mode #:padding-mode padding))
                  (list (randn 1 2 4 5 #:dtype 'float64) (inner-grid 1 3 3 2)))
       (format "~a ~a" mode padding))))

  (bounded-test "G3: affine-grid's and upsample-bicubic2d's gradients match finite differences"
    (check-true (gradcheck (lambda (theta) (affine-grid theta '(2 1 3 4)))
                           (list (randn 2 2 3 #:dtype 'float64)))
                "affine-grid")
    (check-true (gradcheck (lambda (x) (upsample-bicubic2d x '(5 7)))
                           (list (randn 1 2 3 4 #:dtype 'float64)))
                "upsample-bicubic2d"))

  (bounded-test "G4: bicubic upsampling to the input's own size is the identity"
    (for ([dtype (in-list '(float32 float64))])
      (define x (to-dtype (randn 2 3 6 7) dtype))
      (check-equal? (tensor->list (upsample-bicubic2d x '(6 7)))
                    (tensor->list x)
                    (format "~a" dtype))))

  (define (blames-caller? e)
    (and (exn:fail:contract:blame? e)
         (regexp-match? #rx"blaming: [^\n]*grid-sample-test" (exn-message e))))

  (define x (randn 2 3 4 5))
  (define grid (randn 2 6 7 2))
  (define theta (randn 2 2 3))

  (define blamed
    (list (list "input of rank 3" (lambda () (grid-sample (randn 3 4 5) grid)))
          (list "grid ending in 3" (lambda () (grid-sample x (randn 2 6 7 3))))
          (list "grid batch differs" (lambda () (grid-sample x (randn 3 6 7 2))))
          (list "float64 grid, float32 input"
                (lambda () (grid-sample x (to-dtype grid 'float64))))
          (list "mode 'linear" (lambda () (grid-sample x grid #:mode 'linear)))
          (list "uint8 input"
                (lambda () (grid-sample (to-dtype x 'uint8) grid)))
          (list "theta of 3 x 3" (lambda () (affine-grid (randn 2 3 3) '(2 3 4 5))))
          (list "size of length 3" (lambda () (affine-grid theta '(2 4 5))))
          (list "size batch differs" (lambda () (affine-grid theta '(3 3 4 5))))))

  (bounded-test "G5: malformed arguments blame the caller"
    (for ([row (in-list blamed)])
      (match-define (list label thunk) row)
      (check-exn blames-caller? thunk label)))

  (when (cuda-available?)
    (bounded-test "G5: input and grid on different devices blame the caller"
      (check-exn blames-caller?
                 (lambda () (grid-sample x (to grid (cuda-device))))))))
