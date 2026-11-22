#lang racket/base

(module+ test
  (require (only-in racket/contract/combinator exn:fail:contract:blame?)
           (prefix-in base: (only-in racket/base + -))
           rackunit
           "../main.rkt"
           (only-in "../generated.rkt" div-scalar! div-tensor!)
           "../vision/transforms.rkt"
           (only-in "private/op-calls.rkt" op-calls-of))

  (define image-dtypes '(uint8 float16 bfloat16 float32 float64))
  (define float-dtypes '(float16 bfloat16 float32 float64))

  (define (bits t)
    (tensor->list (to-dtype t 'float64)))

  (define (calls thunk)
    (thunk)
    (op-calls-of thunk))

  (define (torchvision-convert x target)
    (define from (dtype x))
    (cond
      [(eq? from target) x]
      [(eq? from 'uint8) (/ (to-dtype x target) 255)]
      [(eq? target 'uint8)
       (to-dtype (* x (base:- (base:+ 255.0 1.0) 1e-3)) 'uint8)]
      [else (to-dtype x target)]))

  (define every-byte (~> (arange 256) (to-dtype 'uint8) (reshape 1 16 16)))
  (define unit-floats
    (reshape (cat (list (tensor '(0.0 1.0 0.5 0.25 0.999 0.001))
                        (rand 42 #:generator (make-generator 3))))
             3 4 4))

  (define (source-of from)
    (if (eq? from 'uint8) every-byte (to-dtype unit-floats from)))

  (test-case "convert-image-dtype: every dtype pair, bit for bit torchvision's"
    (for* ([from (in-list image-dtypes)] [target (in-list image-dtypes)])
      (define x (source-of from))
      (define got (convert-image-dtype x target))
      (define pair (format "~a to ~a" from target))
      (check-equal? (dtype got) target pair)
      (check-equal? (bits got) (bits (torchvision-convert x target)) pair)
      (when (eq? from target)
        (check-eq? got x pair))))

  (test-case "convert-image-dtype: from bytes, one cast and one in-place divide"
    (for ([target (in-list float-dtypes)])
      (define scratch (to-dtype unit-floats target))
      (check-equal? (calls (lambda () (convert-image-dtype every-byte target)))
                    (base:+ (calls (lambda () (dtype every-byte)))
                            (calls (lambda () (dtype every-byte)))
                            (calls (lambda () (to-dtype every-byte target)))
                            (calls (lambda () (div-scalar! scratch 1.0))))
                    (symbol->string target))))

  (define (channel-tensor vs dt dev)
    (tensor (for/list ([v (in-list vs)]) (list (list (exact->inexact v))))
            #:dtype dt
            #:device dev))

  (define (torchvision-normalize x mean std)
    (define (as-tensor vs) (channel-tensor vs (dtype x) (device x)))
    (/ (- x (as-tensor mean)) (as-tensor std)))

  (define stats '((0.485 0.456 0.406) (0.229 0.224 0.225)))

  (test-case "normalize: torchvision's subtract then divide, on a copy"
    (for* ([dt (in-list float-dtypes)]
           [shape (in-list '((3 5 4) (2 3 5 4)))])
      (define x (to-dtype (apply rand shape) dt))
      (define before (bits x))
      (define got (apply normalize x stats))
      (define label (format "~a ~a" dt shape))
      (check-equal? (bits got) (bits (apply torchvision-normalize x stats)) label)
      (check-equal? (bits x) before label)
      (check-equal? (dtype got) dt label)))

  (test-case "normalize: #:inplace? overwrites the image and answers it"
    (for ([dt (in-list float-dtypes)])
      (define x (to-dtype (rand 2 3 5 4) dt))
      (define want (bits (apply torchvision-normalize x stats)))
      (define got (normalize x (car stats) (cadr stats) #:inplace? #t))
      (check-eq? got x (symbol->string dt))
      (check-equal? (bits x) want (symbol->string dt))))

  (test-case "normalize: one value serves every channel; any nonzero std divides"
    (define x (rand 3 2 2))
    (check-equal? (bits (normalize x '(0.5) '(0.25)))
                  (bits (torchvision-normalize x '(0.5 0.5 0.5)
                                               '(0.25 0.25 0.25))))
    (check-equal? (bits (normalize x '(0 0 0) '(-2 1 0.5)))
                  (bits (torchvision-normalize x '(0 0 0) '(-2 1 0.5)))))

  (define ((refusal? rx) e)
    (and (exn:fail:contract? e)
         (not (exn:fail:contract:blame? e))
         (regexp-match? rx (exn-message e))))

  (test-case "normalize: a list of another length or a zero std, refused by name"
    (define x (rand 2 3 4 4))
    (check-exn (refusal? #rx"^normalize: expected one mean per channel")
               (lambda () (normalize x '(0.1 0.2) '(1 1 1))))
    (check-exn (refusal? #rx"^normalize: expected one std per channel")
               (lambda () (normalize x '(0.1 0.2 0.3) '())))
    (check-exn (refusal?
                #rx"^normalize: std evaluated to zero, leading to division by zero")
               (lambda () (normalize x '(0.1 0.2 0.3) '(1 0 1))))
    (check-exn (refusal? #rx"std evaluated to zero")
               (lambda () (normalize x '(0.1) '(-0.0))))
    (check-exn (refusal? #rx"std evaluated to zero")
               (lambda () (normalize x '(0.1) '(0)))))

  (test-case "normalize: two small tensors, one subtraction, one in-place divide"
    (define x (rand 3 8 8))
    (define dev (device x))
    (define m (channel-tensor (car stats) 'float32 dev))
    (define (reads)
      (base:+ (calls (lambda () (dtype x)))
              (calls (lambda () (dtype x)))
              (calls (lambda () (device x)))
              (calls (lambda () (channel-tensor (car stats) 'float32 dev)))
              (calls (lambda () (channel-tensor (cadr stats) 'float32 dev)))))
    (define scratch (rand 3 8 8))
    (define divide (calls (lambda () (div-tensor! scratch m))))
    (check-equal? (calls (lambda () (apply normalize x stats)))
                  (base:+ (reads) (calls (lambda () (- x m))) divide))
    (check-equal? (calls (lambda () (normalize x (car stats) (cadr stats)
                                               #:inplace? #t)))
                  (base:+ (reads) (calls (lambda () (sub! x m))) divide))))
