#lang racket/base

(module+ test
  (require (only-in rackunit
                    check-equal? check-exn check-not-false check-true
                    test-case)
           (only-in "../main.rkt" numel tensor->list tensor-shape zeros)
           (only-in "../nn.rkt"
                    Linear in-eval-mode named-parameters parameters
                    save-state! state-dict)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../vision/resnet.rkt"
                    Bottleneck ImageNetResNet bottleneck? imagenet-resnet?
                    resnet18 resnet34
                    resnet50 torchvision-key)
           (only-in "../vision/weights.rkt" pretrained-weights-cached?))

  (define (parameter-count net)
    (for/sum ([p (in-list (parameters net))]) (numel p)))

  (define (keys net) (map car (state-dict net)))

  (test-case "the three networks have torchvision's parameters and entries"
    (for ([build (in-list (list resnet18 resnet34 resnet50))]
          [params (in-list '(11689512 21797672 25557032))]
          [entries (in-list '(122 218 320))])
      (define net (build))
      (check-true (imagenet-resnet? net))
      (check-equal? (parameter-count net) params)
      (check-equal? (length (keys net)) entries)))

  (test-case "field names are torchvision's, with Racket's hyphens"
    (define k18 (keys (resnet18)))
    (for ([key (in-list '("conv1.weight" "bn1.running-mean"
                          "bn1.num-batches-tracked" "layer1.0.conv2.weight"
                          "layer2.0.shortcut.0.weight"
                          "layer4.1.bn2.running-var" "fc.bias"))])
      (check-not-false (member key k18) key))
    (check-equal? (member "layer1.0.shortcut.0.weight" k18) #f
                  "no projection where the shape does not change")
    (check-not-false (member "layer1.0.shortcut.0.weight" (keys (resnet50)))
                     "a bottleneck widens four times, so layer1 projects")
    (check-not-false (member "layer3.5.conv3.weight" (keys (resnet50)))))

  (test-case "torchvision-key maps the file's names onto the network's"
    (check-equal? (torchvision-key "layer1.0.downsample.0.weight")
                  "layer1.0.shortcut.0.weight")
    (check-equal? (torchvision-key "bn1.running_mean") "bn1.running-mean")
    (check-equal? (torchvision-key "layer4.1.bn2.num_batches_tracked")
                  "layer4.1.bn2.num-batches-tracked")
    (check-equal? (torchvision-key "fc.weight") "fc.weight"))

  (test-case "a batch of images becomes a batch of logits"
    (for ([build (in-list (list resnet18 resnet50))])
      (define net (build))
      (check-equal? (tensor-shape (in-eval-mode net (net (zeros 2 3 64 64))))
                    '(2 1000)))
    (check-equal? (tensor-shape ((resnet18 #:classes 2) (zeros 1 3 64 64)))
                  '(1 2))
    (check-exn exn:fail:contract? (lambda () ((resnet18) (zeros 1 1 64 64)))))

  (test-case "ImageNetResNet builds other depths, basic blocks by default"
    (define net (ImageNetResNet '(1 1 1 1)))
    (check-equal? (length (filter (lambda (key) (regexp-match? #rx"conv2" key))
                                  (keys net)))
                  4)
    (check-equal? (tensor-shape (in-eval-mode net (net (zeros 1 3 32 32))))
                  '(1 1000)))

  (test-case "a bottleneck is 1x1, 3x3 at the stride, 1x1 four times wider"
    (define block (Bottleneck 64 32 #:stride 2))
    (check-true (bottleneck? block))
    (check-equal? (map (lambda (np) (tensor-shape (cdr np)))
                       (filter (lambda (np) (regexp-match? #rx"conv" (car np)))
                               (named-parameters block)))
                  '((32 64 1 1) (32 32 3 3) (128 32 1 1)))
    (check-equal? (tensor-shape (block (zeros 1 64 8 8))) '(1 128 4 4)))

  (test-case "a checkpoint that lacks the backbone is refused"
    (with-temporary-directory (cache)
      (save-state! (Linear 512 1000)
                   (build-path cache "resnet18-imagenet1k-v1.safetensors"))
      (define env (environment-variables-copy (current-environment-variables)))
      (environment-variables-set! env #"RKTORCH_WEIGHTS_DIR"
                                  (string->bytes/utf-8 (path->string cache)))
      (parameterize ([current-environment-variables env])
        (check-exn #rx"does not fit the backbone.*missing: .*conv1.weight"
                   (lambda () (resnet18 #:pretrained? #t #:classes 2)))
        (check-exn #rx"checkpoint does not match the model"
                   (lambda () (resnet18 #:pretrained? #t))))))

  (test-case "a checkpoint with keys the backbone lacks is refused too"
    (with-temporary-directory (cache)
      (save-state! (ImageNetResNet '(2 2 2 3))
                   (build-path cache "resnet18-imagenet1k-v1.safetensors"))
      (define env (environment-variables-copy (current-environment-variables)))
      (environment-variables-set! env #"RKTORCH_WEIGHTS_DIR"
                                  (string->bytes/utf-8 (path->string cache)))
      (parameterize ([current-environment-variables env])
        (check-exn #rx"does not fit the backbone.*unexpected: .*layer4[.]2[.]"
                   (lambda () (resnet18 #:pretrained? #t #:classes 2))))))

  (when (pretrained-weights-cached? 'resnet18-imagenet1k-v1)
    (test-case "another head keeps the pretrained backbone and starts fresh"
      (define full (resnet18 #:pretrained? #t))
      (define two (resnet18 #:pretrained? #t #:classes 2))
      (define (weight net key)
        (cdr (assoc key (named-parameters net))))
      (check-equal? (tensor->list (weight two "layer4.1.conv2.weight"))
                    (tensor->list (weight full "layer4.1.conv2.weight")))
      (check-equal? (tensor-shape (weight two "fc.weight")) '(2 512)))))
