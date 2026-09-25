#lang racket/base

(require (only-in racket/contract/base -> ->* flat-named-contract list/c or/c)
         (only-in racket/string string-replace)
         (only-in threading ~>)
         (only-in "../foreign.rkt"
                  adaptive-avg-pool2d add flatten relu tensor-shape tensor?)
         (only-in "../foreign/contracts.rkt" image-batch/c)
         (only-in "../nn/batch-norm.rkt" BatchNorm2d)
         (only-in "../nn/conv.rkt" Conv2d MaxPool2d)
         (only-in "../nn/layer.rkt" define-layer)
         (only-in "../nn/linear.rkt" Linear)
         (only-in "../nn/sequential.rkt" Sequential)
         (only-in "../nn/state-dict.rkt" load-state!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "weights.rkt" pretrained-weights))

(define (projection in out stride)
  (and (or (not (= stride 1)) (not (= in out)))
       (Sequential (Conv2d in out 1 #:stride stride #:bias? #f)
                   (BatchNorm2d out))))

(define-layer BasicBlock (conv1 bn1 conv2 bn2 shortcut) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:stride exact-positive-integer?]
                  basic-block?)
  #:init (in out #:stride [stride 1])
  (set! conv1 (Conv2d in out 3 #:stride stride #:padding 1 #:bias? #f))
  (set! bn1 (BatchNorm2d out))
  (set! conv2 (Conv2d out out 3 #:padding 1 #:bias? #f))
  (set! bn2 (BatchNorm2d out))
  (set! shortcut (projection in out stride))
  #:forward ([x : image-batch/c])
  (relu (add (bn2 (conv2 (relu (bn1 (conv1 x)))))
             (if shortcut (shortcut x) x))))

(define-layer Bottleneck (conv1 bn1 conv2 bn2 conv3 bn3 shortcut) ;; noqa
  #:contract (->* [exact-positive-integer? exact-positive-integer?]
                  [#:stride exact-positive-integer?]
                  bottleneck?)
  #:init (in width #:stride [stride 1])
  (set! conv1 (Conv2d in width 1 #:bias? #f))
  (set! bn1 (BatchNorm2d width))
  (set! conv2 (Conv2d width width 3 #:stride stride #:padding 1 #:bias? #f))
  (set! bn2 (BatchNorm2d width))
  (set! conv3 (Conv2d width (* 4 width) 1 #:bias? #f))
  (set! bn3 (BatchNorm2d (* 4 width)))
  (set! shortcut (projection in (* 4 width) stride))
  #:forward ([x : image-batch/c])
  (relu (add (~> x conv1 bn1 relu conv2 bn2 relu conv3 bn3)
             (if shortcut (shortcut x) x))))

(define rgb-image-batch/c
  (flat-named-contract 'rgb-image-batch
                       (lambda (x)
                         (and (tensor? x)
                              (let ([shape (tensor-shape x)])
                                (and (= 4 (length shape))
                                     (= 3 (cadr shape))))))))

(define four-stage-depths/c
  (list/c exact-positive-integer? exact-positive-integer?
          exact-positive-integer? exact-positive-integer?))

(define (expansion block) (if (eq? block 'basic) 1 4))

(define (stage block in width blocks stride)
  (define make (if (eq? block 'basic) BasicBlock Bottleneck))
  (define out (* (expansion block) width))
  (apply Sequential
         (make in width #:stride stride)
         (for/list ([_ (in-range (sub1 blocks))])
           (make out width))))

(define-layer ResNet (stem bn layer1 layer2 layer3 layer4 fc) ;; noqa
  #:predicate resnet?
  #:contract (->* []
                  [#:classes exact-positive-integer?
                   #:base exact-positive-integer?
                   #:blocks four-stage-depths/c]
                  resnet?)
  #:init (#:classes [classes 10] #:base [base 64] #:blocks [blocks '(2 2 2 2)])
  (set! stem (Conv2d 3 base 3 #:padding 1 #:bias? #f))
  (set! bn (BatchNorm2d base))
  (set! layer1 (stage 'basic base base (list-ref blocks 0) 1))
  (set! layer2 (stage 'basic base (* 2 base) (list-ref blocks 1) 2))
  (set! layer3 (stage 'basic (* 2 base) (* 4 base) (list-ref blocks 2) 2))
  (set! layer4 (stage 'basic (* 4 base) (* 8 base) (list-ref blocks 3) 2))
  (set! fc (Linear (* 8 base) classes))
  #:forward ([x : rgb-image-batch/c])
  (~> x stem bn relu layer1 layer2 layer3 layer4
      (adaptive-avg-pool2d 1) (flatten 1) fc))

(define block/c (or/c 'basic 'bottleneck))

(define-layer ImageNetResNet ;; noqa
  (conv1 bn1 maxpool layer1 layer2 layer3 layer4 fc)
  #:predicate imagenet-resnet?
  #:contract (->* [four-stage-depths/c]
                  [#:block block/c #:classes exact-positive-integer?]
                  imagenet-resnet?)
  #:init (blocks #:block [block 'basic] #:classes [classes 1000])
  (define e (expansion block))
  (set! conv1 (Conv2d 3 64 7 #:stride 2 #:padding 3 #:bias? #f))
  (set! bn1 (BatchNorm2d 64))
  (set! maxpool (MaxPool2d 3 #:stride 2 #:padding 1))
  (set! layer1 (stage block 64 64 (list-ref blocks 0) 1))
  (set! layer2 (stage block (* 64 e) 128 (list-ref blocks 1) 2))
  (set! layer3 (stage block (* 128 e) 256 (list-ref blocks 2) 2))
  (set! layer4 (stage block (* 256 e) 512 (list-ref blocks 3) 2))
  (set! fc (Linear (* 512 e) classes))
  #:forward ([x : rgb-image-batch/c])
  (~> x conv1 bn1 relu maxpool layer1 layer2 layer3 layer4
      (adaptive-avg-pool2d 1) (flatten 1) fc))

(define/contract-out (torchvision-key key) ;; noqa
  (-> string? string?)
  (string-replace (string-replace key ".downsample." ".shortcut.") "_" "-"))

(define (head-key? key) (regexp-match? #rx"^fc[.]" key))

(define (pretrained blocks block checkpoint pretrained? classes)
  (define net (ImageNetResNet blocks #:block block #:classes classes))
  (when pretrained?
    (define path (pretrained-weights checkpoint))
    (cond
      [(= classes 1000) (load-state! net path #:rename torchvision-key)]
      [else
       (define-values (missing unexpected)
         (load-state! net path
                      #:strict? #f
                      #:rename (lambda (key)
                                 (and (not (head-key? key))
                                      (torchvision-key key)))))
       (define backbone-missing
         (filter (lambda (key) (not (head-key? key))) missing))
       (unless (and (null? backbone-missing) (null? unexpected))
         (raise-arguments-error 'pretrained
                                "the checkpoint does not fit the backbone"
                                "missing" backbone-missing
                                "unexpected" unexpected))]))
  net)

(define builder/c
  (->* [] [#:pretrained? boolean? #:classes exact-positive-integer?]
       imagenet-resnet?))

(define/contract-out (resnet18 #:pretrained? [pretrained? #f] ;; noqa
                               #:classes [classes 1000])
  builder/c
  (pretrained '(2 2 2 2) 'basic 'resnet18-imagenet1k-v1 pretrained? classes))

(define/contract-out (resnet34 #:pretrained? [pretrained? #f] ;; noqa
                               #:classes [classes 1000])
  builder/c
  (pretrained '(3 4 6 3) 'basic 'resnet34-imagenet1k-v1 pretrained? classes))

(define/contract-out (resnet50 #:pretrained? [pretrained? #f] ;; noqa
                               #:classes [classes 1000])
  builder/c
  (pretrained '(3 4 6 3) 'bottleneck 'resnet50-imagenet1k-v1 pretrained?
              classes))
