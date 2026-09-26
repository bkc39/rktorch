#lang racket/base

(module+ test
  (require (only-in racket/path path-only)
           ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in rackunit check-= check-equal? check-true)
           (only-in "../main.rkt"
                    abs backward! bytes->tensor copy! item max requires-grad!
                    select stack sub tensor tensor-shape tensor->list topk
                    with-no-grad)
           (only-in "../nn.rkt"
                    cross-entropy in-eval-mode named-parameters parameters sgd
                    state-dict step! zero-grads!)
           (only-in "../vision/image.rkt" read-image)
           (only-in "../vision/resnet.rkt"
                    resnet18 resnet34 resnet50 torchvision-key)
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

  (define (check-finetune-twin batch weights-dir)
    (define ft
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_WEIGHTS" (path->string weights-dir)))
       (lambda () (python-check "finetune_twin.py"))))
    (define net (resnet18 #:pretrained? #t #:classes 2))
    (define (entry key) (cdr (assoc key (state-dict net))))
    (define head (hash-ref ft 'head))
    (with-no-grad
      (copy! (entry "fc.weight") (unpack (hash-ref head 'weight) 'float32))
      (copy! (entry "fc.bias") (unpack (hash-ref head 'bias) 'float32)))
    (define labels (tensor '(0 0 1 1)))
    (define (run params lr steps)
      (define opt (sgd params #:lr lr #:momentum 0.9))
      (for/list ([_ (in-range steps)])
        (zero-grads! opt)
        (define loss (cross-entropy (net batch) labels))
        (backward! loss)
        (step! opt)
        (item loss)))
    (for ([np (in-list (named-parameters net))])
      (requires-grad! (cdr np) (regexp-match? #rx"^fc[.]" (car np))))
    (define frozen (run (list (entry "fc.weight") (entry "fc.bias")) 0.001 3))
    (for ([p (in-list (parameters net))]) (requires-grad! p #t))
    (define tuned (run (parameters net) 0.0001 3))
    (for ([ours (in-list (append frozen tuned))]
          [theirs (in-list (hash-ref ft 'losses))]
          [step (in-naturals 1)])
      (check-= ours theirs 1e-4 (format "fine-tune twin: loss at step ~a" step)))
    (for ([(key theirs) (in-hash (hash-ref ft 'after))])
      (define name (torchvision-key (symbol->string key)))
      (define worst
        (item (max (abs (sub (entry name) (unpack theirs 'float32))))))
      (check-true (<= worst 1e-4)
                  (format "fine-tune twin: ~a after six steps, max |difference| ~a"
                          name worst))))

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
                     (format "~a: the top five from our own decode" name)))
     (check-finetune-twin their-batch weights-dir)]))
