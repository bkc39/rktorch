#lang racket/base

(module+ test
  (require (only-in racket/match match)
           rackunit
           (only-in "../main.rkt"
                    ~> cuda-available? cuda-device mps-available? mps-device sum tensor to)
           (only-in "../train/private/book.rkt"
                    host-reads make-book note-step! read-groups read-steps!))

  (define kinds
    (hasheq 'a '(mps float32) 'b '(cpu float64) 'c '(cpu float32) 'd '(mps float32)))

  (define (fake-groups . names)
    (read-groups names
                 #:device (lambda (n) (match (hash-ref kinds n) [(list dev _) dev]))
                 #:dtype (lambda (n) (match (hash-ref kinds n) [(list _ dt) dt]))))

  (test-case "each device is read once, in the widest dtype its own metrics hold"
    (check-equal? (fake-groups 'a 'b 'c 'd) '((mps float32 (a d)) (cpu float64 (b c))))
    (check-equal? (fake-groups 'b 'a) '((cpu float64 (b)) (mps float32 (a))))
    (check-equal? (fake-groups 'c 'a) '((cpu float32 (c)) (mps float32 (a)))))

  (define (scalar x dt) (~> (tensor (list x)) sum (to dt)))

  (define (published metrics)
    (define b (make-book))
    (for ([m (in-list metrics)])
      (match m [(list key v) (note-step! b 'train key v #t #f)]))
    (define before (host-reads))
    (define out (read-steps! b 'train))
    (list out (- (host-reads) before)))

  (test-case "float32 and float64 metrics on one device share one read"
    (check-equal? (published (list (list 'narrow (scalar 2.5 'float32))
                                   (list 'wide (scalar 0.25 'float64))))
                  (list (hasheq 'narrow 2.5 'wide 0.25) 1)))

  (test-case "an accelerator float32 metric beside a CPU float64 one reads each on its device"
    (for ([accel (in-list (list (and (cuda-available?) (cuda-device))
                                (and (mps-available?) (mps-device))))]
          #:when accel)
      (define fast (to (scalar 2.5 'float32) accel))
      (define wide (scalar 0.25 'float64))
      (for ([order (in-list (list (list (list 'fast fast) (list 'wide wide))
                                  (list (list 'wide wide) (list 'fast fast))))])
        (check-equal? (published order) (list (hasheq 'fast 2.5 'wide 0.25) 2))))))
