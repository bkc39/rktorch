#lang racket/base

(module+ test
  (require (only-in racket/file make-temporary-file)
           (only-in rackunit check-= check-equal?)
           (only-in "../main.rkt"
                    cuda-available? in-flattened-tensor manual-seed! shape to
                    with-no-grad)
           (only-in "../nn.rkt" eval! save-state!)
           (only-in "../vision/vit.rkt" VisionTransformer)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-torch-available? unpack)
           (only-in "private/within.rkt" bounded-test))

  (define (saved vit)
    (define path (make-temporary-file "rkt-vit-~a.safetensors"))
    (save-state! vit path)
    path)

  (define (check-close label ours theirs tolerance)
    (check-equal? (shape ours) (shape theirs) label)
    (for ([a (in-flattened-tensor (to ours 'cpu))]
          [b (in-flattened-tensor theirs)]
          [i (in-naturals)])
      (check-= a b tolerance (format "~a ~a" label i))))

  (define (check-references device tolerance)
    (manual-seed! 0)
    (define twin (VisionTransformer #:image-size 32 #:patch 4 #:width 192
                                    #:depth 9 #:heads 3))
    (define reference (VisionTransformer #:image-size 32 #:patch 4
                                         #:width 192 #:depth 2 #:heads 3
                                         #:classes 10))
    (define twin-path (saved twin))
    (define reference-path (saved reference))
    (define j
      (call-with-python-env
       #:env (list (cons "RKTORCH_PARITY_DEVICE" (format "~a" device))
                   (cons "RKTORCH_VIT_TWIN" (path->string twin-path))
                   (cons "RKTORCH_VIT_TORCHVISION"
                         (path->string reference-path)))
       (lambda () (python-check "vit_forward.py"))))
    (delete-file twin-path)
    (delete-file reference-path)
    (define (input key) (to (unpack (hash-ref j key) 'float32) device))
    (define (expected key) (unpack (hash-ref j key) 'float32))
    (to twin device)
    (to reference device)
    (eval! twin)
    (eval! reference)
    (with-no-grad
      (check-close (format "V11 twin at 32 px [~a]" device)
                   (twin (input 'x32) #:output 'cls) (expected 'twin32)
                   tolerance)
      (check-close (format "V11 twin at 16 px [~a]" device)
                   (twin (input 'x16) #:output 'cls) (expected 'twin16)
                   tolerance)
      (check-close (format "V12 torchvision [~a]" device)
                   (reference (input 'x32)) (expected 'torchvision)
                   tolerance)))

  (cond
    [(not (python-torch-available?))
     (displayln "[vit-parity-test] skipped: no python3 `torch`")]
    [else
     (bounded-test "V11, V12: the twin's ViT and torchvision's match on the CPU"
       (check-references 'cpu 1e-5))
     (when (and (cuda-available?) (python-cuda-available?))
       (bounded-test "V11, V12: the twin's ViT and torchvision's match on CUDA"
         (check-references 'cuda 1e-4)))]))
