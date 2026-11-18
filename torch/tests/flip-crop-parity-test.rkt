#lang racket/base

(module+ test
  (require (only-in racket/match match match-define)
           (only-in rackunit check-equal?)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    make-generator select stack tensor->list tensor-shape
                    unsqueeze)
           (only-in "../vision/transforms.rkt" random-crop random-horizontal-flip)
           (only-in "private/python-env.rkt"
                    python-check python-module-available? unpack))

  (define (step op generator)
    (match op
      [(list "flip" p) (lambda (x) (random-horizontal-flip x #:p p #:generator generator))]
      [(list "crop" padding)
       (lambda (x) (random-crop x #:padding padding #:generator generator))]))

  (define (per-sample ops images seed)
    (define generator (make-generator seed))
    (define steps (for/list ([op (in-list ops)]) (step op generator)))
    (define (one image)
      (for/fold ([x (unsqueeze image 0)] #:result (select x 0 0))
                ([f (in-list steps)])
        (f x)))
    (~> (for/list ([i (in-range (car (tensor-shape images)))])
          (one (select images 0 i)))
        (stack 0)))

  (define (batched ops images seed)
    (match-define (list op) ops)
    ((step op (make-generator seed)) images))

  (define (check-case j)
    (define dtype (string->symbol (hash-ref j 'dtype)))
    (define images (unpack (hash-ref j 'input) dtype))
    (define want (tensor->list (unpack (hash-ref j 'output) dtype)))
    (define ops (hash-ref j 'ops))
    (define seed (hash-ref j 'seed))
    (check-equal? (tensor->list (per-sample ops images seed)) want
                  (format "~a, image by image" (hash-ref j 'name)))
    (when (= 1 (length ops))
      (check-equal? (tensor->list (batched ops images seed)) want
                    (format "~a, as one batch" (hash-ref j 'name)))))

  (cond
    [(not (python-module-available? "torchvision"))
     (printf "[flip-crop-parity-test] skipped: python3 `torchvision` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (for ([j (in-list (python-check "flip_crop_parity.py"))])
       (check-case j))]))
