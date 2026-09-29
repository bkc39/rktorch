#lang racket/base

(module+ test
  (require (only-in rackunit check-equal? check-exn check-true test-case)
           (only-in "../main.rkt" numel randn shape)
           (only-in "../nn.rkt" named-parameters)
           (only-in "../vision/vgg.rkt" vgg16-features vgg16-features?)
           (only-in "../vision/weights.rkt" pretrained-weights-cached?))

  (define net (vgg16-features))

  (test-case "torchvision's VGG-16 features, in torchvision's slots"
    (check-true (vgg16-features? net))
    (check-equal? (for/sum ([p (in-list (named-parameters net))])
                    (numel (cdr p)))
                  14714688)
    (check-equal? (map car (named-parameters net))
                  (for*/list ([i (in-list '(0 2 5 7 10 12 14 17 19 21 24 26 28))]
                              [field (in-list '("weight" "bias"))])
                    (format "features.~a.~a" i field))))

  (test-case "five stages halve the image five times"
    (check-equal? (shape (net (randn 2 3 64 96))) '(2 512 2 3))
    (check-exn #rx"rgb-image-batch"
               (lambda () (net (randn 2 1 64 64)))))

  (when (pretrained-weights-cached? 'vgg16-features-imagenet1k-v1)
    (test-case "the pretrained features load under their own names"
      (check-true (vgg16-features? (vgg16-features #:pretrained? #t))))))
