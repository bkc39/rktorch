#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list make-list)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in "../main.rkt"
                    bytes->tensor cuda-available? index-select make-generator
                    tensor tensor->list to)
           (only-in "../vision/augment.rkt"
                    adjust-brightness adjust-contrast adjust-hue
                    adjust-saturation crop-view multi-crop resized-crop
                    rgb->grayscale solarize)
           (only-in "private/python-env.rkt"
                    call-with-python-env python-check python-cuda-available?
                    python-module-available? python-torch-available? unpack))

  (define-runtime-path cifar-fixture "../vision/fixtures/cifar10-256.bin")

  (define (flat v)
    (if (list? v) (apply append (map flat v)) (list v)))

  (define (check-close label got expected tolerance)
    (define a (flat (tensor->list (to got 'cpu))))
    (define b (flat (tensor->list (to expected 'cpu))))
    (check-equal? (length a) (length b) label)
    (define worst
      (for/fold ([worst 0.0]) ([u (in-list a)] [v (in-list b)])
        (max worst (abs (- u v)))))
    (check-true (<= worst tolerance) (format "~a: max abs difference ~a" label worst)))

  (define (twin name device)
    (call-with-python-env
     #:env (list (cons "RKTORCH_PARITY_DEVICE" (symbol->string device)))
     (lambda () (python-check name))))

  (define functional
    (list (cons 'brightness adjust-brightness)
          (cons 'contrast adjust-contrast)
          (cons 'saturation adjust-saturation)
          (cons 'hue adjust-hue)
          (cons 'solarize solarize)))

  (define (interior t rows cols on)
    (index-select (index-select t 2 (to (tensor rows #:dtype 'int64) on))
                  3 (to (tensor cols #:dtype 'int64) on)))

  (define (check-v2 device tolerance)
    (define j (twin "augment_v2.py" device))
    (define x (to (unpack (hash-ref j 'x) 'float32) device))
    (define (expected e) (unpack e 'float32))
    (define factors (hash-ref j 'factors))
    (for ([named (in-list functional)])
      (define name (car named))
      (for ([f (in-list (hash-ref factors name))]
            [e (in-list (hash-ref j name))])
        (check-close (format "A10 ~a ~a [~a]" name f device)
                     ((cdr named) x f) (expected e) tolerance)))
    (for ([k (in-list '(1 3))]
          [e (in-list (hash-ref j 'gray))])
      (check-close (format "A10 grayscale ~a [~a]" k device)
                   (rgb->grayscale x #:channels k) (expected e) tolerance))
    (for ([c (in-list (hash-ref j 'crops))])
      (define box (map exact->inexact (hash-ref c 'box)))
      (define out (resized-crop x (to (tensor (make-list 2 box)) device)
                                (hash-ref c 'size)))
      (define rows (hash-ref c 'rows))
      (define cols (hash-ref c 'cols))
      (check-close (format "A10 resized-crop ~a [~a]" box device)
                   (interior out rows cols device)
                   (interior (expected (hash-ref c 'out)) rows cols 'cpu)
                   tolerance)))

  (define (fixture-images n device)
    (define data
      (call-with-input-file cifar-fixture (lambda (in) (read-bytes (* n 3073) in))))
    (define pixels
      (apply bytes-append
             (for/list ([i (in-range n)])
               (subbytes data (+ 1 (* i 3073)) (* (add1 i) 3073)))))
    (to (bytes->tensor pixels 'uint8 (list n 3 32 32)) device))

  (define mean '(0.4914 0.4822 0.4465))
  (define std '(0.2470 0.2435 0.2616))

  (define (dino-views)
    (values (list (crop-view #:size 32 #:scale '(0.4 1.0) #:mean mean #:std std)
                  (crop-view #:size 32 #:scale '(0.4 1.0) #:solarize-p 0.2
                             #:mean mean #:std std))
            (make-list 6 (crop-view #:size 16 #:scale '(0.05 0.4)
                                    #:mean mean #:std std))))

  (define (check-multi-crop device tolerance)
    (define j (twin "multi_crop.py" device))
    (define images (fixture-images 16 device))
    (define-values (globals locals) (dino-views))
    (for ([seed (in-list '(0 1))])
      (define e (hash-ref j (string->symbol (number->string seed))))
      (define-values (big small)
        (multi-crop images globals locals
                    #:generator (make-generator seed #:device device)))
      (check-close (format "A11 globals seed ~a [~a]" seed device)
                   big (unpack (hash-ref e 'global) 'float32) tolerance)
      (check-close (format "A11 locals seed ~a [~a]" seed device)
                   small (unpack (hash-ref e 'local) 'float32) tolerance)))

  (cond
    [(not (and (python-torch-available?) (python-module-available? "torchvision")))
     (printf "[augment-parity-test] skipped: python3 `torch` and `torchvision` ~a\n"
             "not available (run inside `nix develop`)")]
    [else
     (check-v2 'cpu 1e-5)
     (check-multi-crop 'cpu 1e-5)
     (when (and (cuda-available?) (python-cuda-available?))
       (check-v2 'cuda 1e-4)
       (check-multi-crop 'cuda 1e-4))]))
