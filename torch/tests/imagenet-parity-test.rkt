#lang racket/base

(module+ test
  (require (only-in racket/path path-only)
           ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in rackunit check-equal? check-true)
           (only-in "../main.rkt"
                    abs bytes->tensor item max select stack sub tensor-shape
                    tensor->list topk with-no-grad)
           (only-in "../nn.rkt" in-eval-mode)
           (only-in "../vision/image.rkt" read-image)
           (only-in "../vision/resnet.rkt" resnet18 resnet34 resnet50)
           (only-in "../vision/transforms.rkt" imagenet-preprocess)
           (only-in "../vision/weights.rkt"
                    pretrained-weights pretrained-weights-cached?)
           "private/python-env.rkt")

  (define-runtime-path photos-dir "../vision/fixtures/hymenoptera")

  (define (hex->bytes s)
    (apply bytes
           (for/list ([i (in-range 0 (string-length s) 2)])
             (string->number (substring s i (+ i 2)) 16))))

  (define (unpack j dtype)
    (bytes->tensor (hex->bytes (hash-ref j 'hex)) dtype (hash-ref j 'shape)))

  (define models
    (list (list 'resnet18 resnet18 'resnet18-imagenet1k-v1)
          (list 'resnet34 resnet34 'resnet34-imagenet1k-v1)
          (list 'resnet50 resnet50 'resnet50-imagenet1k-v1)))

  (define (top5 logits)
    (for/list ([i (in-range (car (tensor-shape logits)))])
      (define-values (_ indices) (topk (select logits 0 i) 5))
      (tensor->list indices)))

  (cond
    [(not (python-module-available? "torchvision"))
     (displayln "[imagenet-parity-test] skipped: python3 `torchvision` not available")]
    [(not (for/and ([m (in-list models)]) (pretrained-weights-cached? (caddr m))))
     (printf "[imagenet-parity-test] skipped: pretrained weights not cached ~a\n"
             "(pretrained-weights fetches them)")]
    [else
     (define weights-dir (path-only (pretrained-weights 'resnet18-imagenet1k-v1)))
     (define j
       (call-with-python-env
        #:env (list (cons "RKTORCH_PARITY_WEIGHTS" (path->string weights-dir)))
        (lambda () (python-check "imagenet_resnet.py"))))
     (define their-batch
       (stack (for/list ([p (in-list (hash-ref j 'pixels))])
                (imagenet-preprocess (unpack p 'uint8)))
              0))
     (define our-batch
       (stack (for/list ([photo (in-list (hash-ref j 'photos))])
                (imagenet-preprocess
                 (read-image (build-path photos-dir photo) #:mode 'rgb)))
              0))
     (for ([m (in-list models)])
       (define name (car m))
       (define net ((cadr m) #:pretrained? #t))
       (define theirs (unpack (hash-ref (hash-ref j 'logits) name) 'float32))
       (define-values (ours on-our-decode)
         (in-eval-mode net
           (with-no-grad (values (net their-batch) (net our-batch)))))
       (check-equal? (tensor-shape ours) (tensor-shape theirs))
       (define worst (item (max (abs (sub ours theirs)))))
       (check-true (<= worst 1e-3)
                   (format "~a: logits on torchvision's pixels, max |difference| ~a"
                           name worst))
       (check-equal? (top5 on-our-decode) (hash-ref (hash-ref j 'top5) name)
                     (format "~a: the top five from our own decode" name)))]))
