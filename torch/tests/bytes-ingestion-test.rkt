#lang racket/base

(module+ test
  (require (only-in ffi/vector f32vector)
           (only-in racket/list range)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" bytes->tensor cpu-device cuda-available?
                    cuda-device default-device div full full-like item reshape
                    tensor tensor-device with-default-device
                    tensor->list tensor->repr tensor->vector tensor-dtype
                    tensor-shape to to-dtype zeros))

  (test-case "a byte string builds a uint8 tensor and comes back as one"
    (define t (tensor #"\0\1\2\377"))
    (check-equal? (tensor-dtype t) 'uint8)
    (check-equal? (tensor-shape t) '(4))
    (check-equal? (tensor->list t) '(0 1 2 255))
    (check-equal? (tensor->vector t) #"\0\1\2\377")
    (check-equal? (tensor->repr t)
                  "tensor([  0,   1,   2, 255], dtype=torch.uint8)")
    (check-equal? (tensor->repr (reshape t 2 2))
                  "tensor([[  0,   1],\n        [  2, 255]], dtype=torch.uint8)")
    (check-equal? (tensor->repr (tensor #"")) "tensor([], dtype=torch.uint8)")
    (check-equal? (item (tensor #"\7")) 7)
    (check-equal? (tensor->list (tensor #"\5" #:device 'cpu)) '(5)))

  (test-case "#:dtype converts bytes on the way in, and lists to uint8"
    (define f (tensor #"\1\2" #:dtype 'float32))
    (check-equal? (tensor-dtype f) 'float32)
    (check-equal? (tensor->list f) '(1.0 2.0))
    (check-equal? (tensor-dtype (tensor #"\1\2" #:dtype 'int64)) 'int64)
    (define u (tensor '((1 2) (3 255)) #:dtype 'uint8))
    (check-equal? (tensor-dtype u) 'uint8)
    (check-equal? (tensor-shape u) '(2 2))
    (check-equal? (tensor->vector u) #"\1\2\3\377")
    (check-equal? (tensor->list (tensor '(1.0 2.5 255.0) #:dtype 'uint8))
                  '(1 2 255) "inexact values truncate, as for int64")
    (check-equal? (tensor->list (tensor (f32vector 3.0 4.0) #:dtype 'uint8))
                  '(3 4))
    (check-exn #rx"^tensor: cannot convert value to uint8"
               (lambda () (tensor '(256) #:dtype 'uint8)))
    (check-exn #rx"^tensor: cannot convert value to uint8"
               (lambda () (tensor '(-1.0) #:dtype 'uint8)))
    (check-exn #rx"^tensor: cannot convert non-finite value to uint8"
               (lambda () (tensor (list +inf.0) #:dtype 'uint8)))
    (check-exn exn:fail? (lambda () (tensor #"\1" #:requires-grad? #t))))

  (test-case "uint8 is a dtype everywhere a dtype goes"
    (check-equal? (tensor-dtype (to (tensor '(1 2)) 'uint8)) 'uint8)
    (check-equal? (tensor-dtype (zeros 2 #:dtype 'uint8)) 'uint8)
    (check-equal? (tensor->repr (zeros 2 #:dtype 'uint8))
                  "tensor([0, 0], dtype=torch.uint8)")
    (check-equal? (tensor->list (full 7 2 #:dtype 'uint8)) '(7 7))
    (check-equal? (tensor->list (full-like (zeros 2) 255.0 #:dtype 'uint8))
                  '(255 255))
    (check-exn #rx"uint8 fill value must be an integer from 0 to 255"
               (lambda () (full 300 2 #:dtype 'uint8)))
    (check-exn #rx"uint8 fill value must be an integer from 0 to 255"
               (lambda () (full 1.5 2 #:dtype 'uint8)))
    (check-exn #rx"uint8 fill value must be an integer from 0 to 255"
               (lambda () (full-like (zeros 2) -1 #:dtype 'uint8)))
    ;; torch truncates a fractional fill for both dtypes; int64 refuses it
    ;; here for the reason uint8 does
    (check-equal? (tensor->list (full 2.0 2 #:dtype 'int64)) '(2 2))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full 0.5 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full 1/2 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (add1 (expt 2 60)) 2 #:dtype 'int64)))
    (check-exn #rx"int64 fill value must be an integer"
               (lambda () (full (expt 2 63) 2 #:dtype 'int64)))
    (check-equal? (tensor->list (full 1 2 #:dtype 'bool)) '(1.0 1.0))
    (check-exn #rx"bool fill value must be 0 or 1"
               (lambda () (full 0.5 2 #:dtype 'bool)))
    (check-equal? (tensor->repr (to (tensor '(1.5 -2.0)) 'float64))
                  "tensor([ 1.5000, -2.0000], dtype=torch.float64)"))

  (test-case "bytes->tensor takes a device and leaves the default alone"
    (define bs (bytes 0 0 128 63 0 0 0 64))
    (define here (bytes->tensor bs 'float32 '(2) #:device 'cpu))
    (check-equal? (tensor-device here) (cpu-device))
    (check-equal? (tensor->list here) '(1.0 2.0))
    (check-equal? (tensor-device (bytes->tensor bs 'float32 '(2)))
                  (default-device) "without #:device it is the default's")
    (when (cuda-available?)
      (with-default-device 'cuda
        (check-equal? (tensor-device (bytes->tensor bs 'float32 '(2)
                                                    #:device 'cpu))
                      (cpu-device) "#:device wins over the default")
        (check-equal? (default-device) (cuda-device)
                      "and the default is not borrowed to get there"))))

  (test-case "bytes widen where they are wanted, half stages on the host"
    (define bs (bytes 1 2 3 4))
    (for ([dt (in-list '(float32 int64))]
          [want (in-list '((1.0 2.0 3.0 4.0) (1 2 3 4)))])
      (define t (tensor bs #:dtype dt))
      (check-equal? (tensor-dtype t) dt)
      (check-equal? (tensor->list t) want))
    (when (cuda-available?)
      (for ([dt (in-list '(float32 float16))])
        (define g (tensor bs #:dtype dt #:device 'cuda))
        (check-equal? (tensor-device g) (cuda-device))
        (check-equal? (tensor-dtype g) dt))))

  (test-case "bool bytes are read as nonzero, whatever the byte"
    (define b (bytes->tensor (bytes 0 1 2 255) 'bool '(4)))
    (check-equal? (tensor-dtype b) 'bool)
    (check-equal? (tensor->list b) '(0.0 1.0 1.0 1.0)))

  (test-case "uint8 -> float32 -> / 255 equals the boxed double path bit for bit"
    (define bs (list->bytes (range 256)))
    (define fast (tensor->list (div (to-dtype (tensor bs) 'float32) 255.0)))
    (define boxed
      (tensor->list
       (tensor (for/list ([b (in-bytes bs)]) (/ (exact->inexact b) 255.0)))))
    (check-equal? fast boxed)))
