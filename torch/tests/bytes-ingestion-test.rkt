#lang racket/base

(module+ test
  (require (only-in racket/list range)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" div full full-like item reshape tensor
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
    (check-exn #rx"^tensor: cannot convert value to uint8"
               (lambda () (tensor '(256) #:dtype 'uint8)))
    (check-exn #rx"^tensor: cannot convert value to uint8"
               (lambda () (tensor '(1.5) #:dtype 'uint8)))
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
    (check-equal? (tensor->repr (to (tensor '(1.5 -2.0)) 'float64))
                  "tensor([ 1.5000, -2.0000], dtype=torch.float64)"))

  (test-case "uint8 -> float32 -> / 255 equals the boxed double path bit for bit"
    (define bs (list->bytes (range 256)))
    (define fast (tensor->list (div (to-dtype (tensor bs) 'float32) 255.0)))
    (define boxed
      (tensor->list
       (tensor (for/list ([b (in-bytes bs)]) (/ (exact->inexact b) 255.0)))))
    (check-equal? fast boxed)))
