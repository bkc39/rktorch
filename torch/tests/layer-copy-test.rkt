#lang racket/base

(module+ test
  (require rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in (submod "../nn/recurrent.rkt" private) flattened-placement)
           (only-in "../vision/diffusion.rkt" UNet)
           (only-in "../vision/resnet.rkt" ImageNetResNet ResNet)
           (only-in "../vision/vgg.rkt" vgg16-features))

  (define (close? a b [eps 1e-6])
    (and (equal? (shape a) (shape b))
         (for/and ([x (in-flattened-tensor a)] [y (in-flattened-tensor b)])
           (<= (abs (- x y)) eps))))

  (define ((message-matching pattern) e)
    (and (exn:fail? e)
         (regexp-match? pattern
                        (regexp-replace* #rx"[ \n]+" (exn-message e) " "))))

  (define (state m)
    (for/list ([e (in-list (append (named-parameters m) (named-buffers m)))])
      (cons (car e) (tensor->list (cdr e)))))

  (define (tensors m)
    (append (parameters m) (buffers m)))

  (define (same? a b)
    (and (equal? (shape a) (shape b))
         (eq? (dtype a) (dtype b))
         (equal? (device a) (device b))
         (zero? (item (max (abs (- (to-dtype a 'float64)
                                   (to-dtype b 'float64))))))))

  (define (same-state? m n)
    (define (entries l) (append (named-parameters l) (named-buffers l)))
    (and (equal? (map car (entries m)) (map car (entries n)))
         (for/and ([a (in-list (entries m))] [b (in-list (entries n))])
           (same? (cdr a) (cdr b)))))

  (define (first-output m inputs)
    (call-with-values (lambda () (apply m inputs))
                      (lambda (out . _rest) out)))

  (define (check-copy label make inputs)
    (manual-seed! 0)
    (define m (make))
    (define stream (tensor->list (randn 4)))
    (manual-seed! 0)
    (define original (make))
    (define c (layer-copy original))
    (check-equal? (tensor->list (randn 4)) stream
                  (format "~a: the copy draws nothing" label))
    (check-true (same-state? c original) (format "~a: values" label))
    (check-equal? (map car (named-children c))
                  (map car (named-children original))
                  (format "~a: children" label))
    (check-false (eq? c original))
    (for ([t (in-list (tensors c))])
      (check-false (memq t (tensors original))
                   (format "~a: fresh tensors" label)))
    (check-equal? (map requires-grad? (parameters c))
                  (map requires-grad? (parameters original)))
    (when inputs
      (eval! original)
      (eval! c)
      (check-equal? (layer-training? c) #f)
      (define before (first-output c inputs))
      (check-true (close? (first-output original inputs) before)
                  (format "~a: same output" label))
      (with-no-grad
        (for ([t (in-list (tensors original))]) (zero! t)))
      (check-true (close? (first-output c inputs) before)
                  (format "~a: the copy keeps its values" label)))
    (void m))

  (manual-seed! 1)
  (define x8 (randn 5 2 8))
  (define images (randn 2 3 32 32))

  (test-case "every built-in layer family copies with no draws and no sharing"
    (check-copy "Linear" (lambda () (Linear 3 4)) (list (randn 2 3)))
    (check-copy "Conv1d" (lambda () (Conv1d 2 3 3)) (list (randn 1 2 8)))
    (check-copy "Conv2d" (lambda () (Conv2d 2 3 3)) (list (randn 1 2 6 6)))
    (check-copy "ConvTranspose2d" (lambda () (ConvTranspose2d 2 3 3))
                (list (randn 1 2 4 4)))
    (check-copy "MaxPool2d" (lambda () (MaxPool2d 2)) (list (randn 1 2 4 4)))
    (check-copy "Flatten" (lambda () (Flatten)) (list (randn 2 3 4)))
    (check-copy "BatchNorm1d"
                (lambda ()
                  (define bn (BatchNorm1d 3))
                  (bn (randn 4 3))
                  bn)
                (list (randn 4 3)))
    (check-copy "BatchNorm2d" (lambda () (BatchNorm2d 2))
                (list (randn 2 2 3 3)))
    (check-copy "LayerNorm" (lambda () (LayerNorm 4 #:bias? #f))
                (list (randn 2 4)))
    (check-copy "GroupNorm" (lambda () (GroupNorm 2 4))
                (list (randn 1 4 3 3)))
    (check-copy "Embedding" (lambda () (Embedding 10 4))
                (list (tensor '((1 2 3)))))
    (check-copy "LSTM" (lambda () (LSTM 3 4 #:num-layers 2 #:bidirectional? #t))
                (list (randn 5 2 3)))
    (check-copy "GRU" (lambda () (GRU 3 4)) (list (randn 5 2 3)))
    (check-copy "Dropout" (lambda () (Dropout)) (list (randn 2 3)))
    (check-copy "MultiheadAttention"
                (lambda () (MultiheadAttention 8 #:heads 2))
                (list x8 x8 x8))
    (check-copy "TransformerEncoderLayer"
                (lambda () (TransformerEncoderLayer 8 #:heads 2 #:ffn-width 16))
                (list x8))
    (check-copy "TransformerDecoderLayer"
                (lambda () (TransformerDecoderLayer 8 #:heads 2 #:ffn-width 16))
                (list (randn 4 2 8) x8))
    (check-copy "TransformerEncoder"
                (lambda () (TransformerEncoder 8 #:heads 2 #:ffn-width 16
                                               #:layers 2 #:norm #t))
                (list x8))
    (check-copy "TransformerDecoder"
                (lambda () (TransformerDecoder 8 #:heads 2 #:ffn-width 16
                                               #:layers 2))
                (list (randn 4 2 8) x8))
    (check-copy "Sequential" (lambda () (Sequential (Linear 3 4) relu))
                (list (randn 2 3)))
    (check-copy "LayerList" (lambda () (LayerList (list (Linear 3 4)))) #f)
    (check-copy "LayerHash"
                (lambda () (LayerHash (list (cons "a" (Linear 3 4))))) #f)
    (check-copy "ResNet"
                (lambda () (ResNet #:base 8 #:blocks '(1 1 1 1)))
                (list images))
    (check-copy "ImageNetResNet"
                (lambda () (ImageNetResNet '(1 1 1 1) #:block 'bottleneck
                                           #:classes 10))
                #f)
    (check-copy "UNet"
                (lambda () (UNet #:base 32 #:mults '(1 2) #:blocks 1))
                (list images (tensor '(3 7)) #f))
    (check-copy "VGG16Features" vgg16-features #f))

  (test-case "the copy and the original train apart"
    (manual-seed! 0)
    (define net (Sequential (Linear 3 4) relu (Linear 4 1)))
    (define twin (layer-copy net))
    (define before (state net))
    (define opt (sgd (parameters twin) #:lr 0.5))
    (backward! (sum (twin (randn 8 3))))
    (step! opt)
    (check-equal? (state net) before "training the copy leaves the original")
    (check-false (equal? (state twin) before))
    (check-false (has-grad? (car (parameters net))))
    (define again (state twin))
    (with-no-grad
      (for ([p (in-list (parameters net))]) (zero! p)))
    (check-equal? (state twin) again "and the other way round"))

  (test-case "gradients stay behind; requires-grad, dtype and modes carry over"
    (define l (to (Linear 2 3) 'float64))
    (requires-grad! (car (parameters l)) #f)
    (backward! (sum (l (to (randn 4 2) 'float64))))
    (define c (layer-copy l))
    (check-equal? (map requires-grad? (parameters c)) '(#f #t))
    (check-equal? (map dtype (parameters c)) '(float64 float64))
    (check-false (has-grad? (cadr (parameters c))))
    (define net (Sequential (Linear 2 2) (Dropout)))
    (eval! (child-ref net "1"))
    (define copied (layer-copy net))
    (check-true (layer-training? (child-ref copied "0")))
    (check-false (layer-training? (child-ref copied "1"))))

  (test-case "shared children and tensors stay shared within the copy"
    (define l (Linear 2 2))
    (define twice (Sequential l l))
    (define c (layer-copy twice))
    (check-eq? (child-ref c "0") (child-ref c "1"))
    (check-false (eq? (child-ref c "0") l))
    (define-layer tied (a b)
      #:init (p)
      (set! a p)
      (set! b p)
      #:forward (x)
      (* x a b))
    (define t (tied (Parameter (ones 2))))
    (define tc (layer-copy t))
    (check-equal? (length (parameters tc)) 1)
    (check-equal? (map car (named-parameters tc)) '("a")))

  (test-case "tensors in plain fields, lists and boxes are copied too"
    (define-layer holder (table pair stash fixed)
      #:init (t)
      (set! table t)
      (set! pair (list t (vector t) (vector->immutable-vector (vector t))))
      (set! stash (box t))
      (set! fixed (ones 2))
      #:forward (x)
      (+ x table (car pair) (vector-ref (cadr pair) 0)
         (vector-ref (caddr pair) 0) (unbox stash) fixed))
    (define t (requires-grad! (ones 2)))
    (define h (holder t))
    (define c (layer-copy h))
    (with-no-grad (zero! t))
    (check-equal? (tensor->list (c (zeros 2))) '(6.0 6.0))
    (check-equal? (tensor->list (h (zeros 2))) '(1.0 1.0)))

  (test-case "procedure layers copy when stateless and refuse otherwise"
    (define stateless (procedure->Layer relu))
    (define c (layer-copy stateless))
    (check-false (eq? c stateless))
    (check-equal? (tensor->list (c (tensor '(-1.0 2.0)))) '(0.0 2.0))
    (check-exn (message-matching #rx"procedure->Layer layer with parameters")
               (lambda ()
                 (layer-copy
                  (procedure->Layer
                   (lambda (x) x)
                   #:parameters (list (cons "w" (Parameter (ones 2)))))))))

  (test-case "a hand-written layer copies through its own layer-rebuild"
    (struct plain ()
      #:methods gen:layer
      [(define (layer-forward self . inputs) (car inputs))])
    (check-exn (message-matching
                #rx"^layer-copy: a hand-written gen:layer has no layer-rebuild")
               (lambda () (layer-copy (plain))))
    (struct scaled (w)
      #:methods gen:layer
      [(define (layer-forward self . inputs)
         (* (car inputs) (scaled-w self)))
       (define (layer-parameters self) (list (scaled-w self)))
       (define (layer-rebuild self child tensor)
         (scaled (tensor (scaled-w self))))])
    (define s (scaled (Parameter (ones 2))))
    (define c (layer-copy s))
    (check-false (eq? (scaled-w c) (scaled-w s)))
    (check-equal? (tensor->list (scaled-w c)) '(1.0 1.0))
    (check-exn exn:fail:contract? (lambda () (layer-copy 3))))

  (test-case "layer-rebuild maps children by name and tensors as given"
    (define net (Sequential (Linear 2 2) (Linear 2 2)))
    (define seen '())
    (define rebuilt
      (layer-rebuild net
                     (lambda (name child)
                       (set! seen (cons name seen))
                       child)
                     values))
    (check-equal? (reverse seen) '("0" "1"))
    (check-eq? (child-ref rebuilt "0") (child-ref net "0"))
    (check-false (eq? rebuilt net)))

  (when (cuda-available?)
    (test-case "on CUDA the copy lives there too, the LSTM flattened afresh"
      (manual-seed! 0)
      (define lstm (to (LSTM 3 4 #:num-layers 2) 'cuda))
      (define x (to-device (randn 5 2 3) 'cuda))
      (define reference (first-output lstm (list x)))
      (define c (layer-copy lstm))
      (check-false (flattened-placement (car (parameters c))))
      (check-equal? (map device-type (map device (parameters c)))
                    (map (lambda (_p) 'cuda) (parameters c)))
      (define copied (first-output c (list x)))
      (check-true (close? (to-device copied 'cpu) (to-device reference 'cpu)
                          1e-5))
      (with-no-grad
        (for ([p (in-list (parameters lstm))]) (zero! p)))
      (check-true (close? (to-device (first-output c (list x)) 'cpu)
                          (to-device reference 'cpu) 1e-5)))))
