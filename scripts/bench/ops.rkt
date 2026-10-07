#lang racket/base

(require (only-in ffi/unsafe _fun _int64)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in torch
                  [add facade-add] [conv2d facade-conv2d] [matmul facade-matmul]
                  [sum facade-sum] backward! item randn reclaim-native-memory! to)
         (only-in torch/foreign/error check-ok)
         (only-in torch/foreign/raw/autograd tr-tensor-backward/raw)
         (only-in torch/foreign/raw/elementwise tr-add/raw)
         (only-in torch/foreign/raw/linalg tr-matmul/raw)
         (only-in torch/foreign/raw/memory tensor-allocator)
         (only-in torch/foreign/raw/pressure collect-at-trough!)
         (only-in torch/foreign/raw/pressure-settings native-collect-at-troughs)
         (only-in torch/foreign/raw/reduce tr-sum/raw)
         (only-in torch/foreign/raw/syntax _Tensor _Tensor/null define-torch)
         (only-in torch/foreign/structs wrap-tensor)
         (only-in torch/foreign/tensor-ops
                  [add unchecked-add] [matmul unchecked-matmul] [sum unchecked-sum])
         (prefix-in g: (only-in torch/generated conv2d linear))
         (only-in torch/nn Linear parameters)
         (only-in "harness.rkt" bench-case))

(provide op-cases
         op-names)

(define-torch conv2d/raw
  (_fun _Tensor _Tensor _Tensor/null
        (_s64vector i) _int64 (_s64vector i) _int64 (_s64vector i) _int64 _int64
        -> _Tensor/null)
  #:c-id tr_gen_conv2d
  #:wrap tensor-allocator)

(define-torch linear/raw
  (_fun _Tensor _Tensor _Tensor/null -> _Tensor/null)
  #:c-id tr_gen_linear
  #:wrap tensor-allocator)

(define ones-2d (list->s64vector '(1 1)))

(define (backward/unchecked t)
  (check-ok (tr-tensor-backward/raw t) 'backward!)
  (when (native-collect-at-troughs)
    (collect-at-trough!)))

(define (synchronizer device)
  (if (eq? device 'cpu)
      void
      (lambda (t) (item (facade-sum t)))))

(struct op (name reps variants))

(define (binary-op name reps x y f-facade f-unchecked f-raw)
  (op name reps
      (list (list 'facade (lambda () (f-facade x y)) values)
            (list 'unchecked (lambda () (f-unchecked x y)) values)
            (list 'raw (lambda () (f-raw x y)) wrap-tensor))))

(define (conv-op device reps)
  (define x (randn 1 3 32 32 #:device device))
  (define w (randn 16 3 3 3 #:device device))
  (define b (randn 16 #:device device))
  (op 'conv2d-1x3x32x32-k16 reps
      (list (list 'facade (lambda () (facade-conv2d x w #:bias b #:padding 1)) values)
            (list 'unchecked (lambda () (g:conv2d x w b '(1 1) '(1 1) '(1 1) 1)) values)
            (list 'raw
                  (lambda () (conv2d/raw x w b ones-2d 2 ones-2d 2 ones-2d 2 1))
                  wrap-tensor))))

(define (linear-op device reps)
  (define layer (to (Linear 64 64) device))
  (define-values (w b) (apply values (parameters layer)))
  (define x (randn 32 64 #:device device))
  (op 'linear-64-fwd-bwd reps
      (list (list 'facade (lambda () (backward! (facade-sum (layer x))) w) values)
            (list 'unchecked
                  (lambda () (backward/unchecked (unchecked-sum (g:linear x w b))) w)
                  values)
            (list 'raw
                  (lambda () (tr-tensor-backward/raw (tr-sum/raw (linear/raw x w b))) w)
                  values))))

(define (ops-for device scale)
  (define (n reps) (max 1 (inexact->exact (round (* reps scale)))))
  (define (pair . dims)
    (values (apply randn #:device device dims)
            (apply randn #:device device dims)))
  (define-values (a8 b8) (pair 8 8))
  (define-values (a64 b64) (pair 64 64))
  (define-values (a512 b512) (pair 512 512))
  (list (binary-op 'add-8x8 (n 20000) a8 b8 facade-add unchecked-add tr-add/raw)
        (binary-op 'matmul-64 (n 5000) a64 b64 facade-matmul unchecked-matmul tr-matmul/raw)
        (binary-op 'matmul-512 (n 100) a512 b512 facade-matmul unchecked-matmul tr-matmul/raw)
        (conv-op device (n 2000))
        (linear-op device (n 2000))))

(define op-names
  '(add-8x8 matmul-64 matmul-512 conv2d-1x3x32x32-k16 linear-64-fwd-bwd))

(define (op-cases #:device [device 'cpu] #:scale [scale 1])
  (define sync! (synchronizer device))
  (for/list ([o (in-list (ops-for device scale))])
    (cons (op-name o)
          (for/list ([v (in-list (op-variants o))])
            (define call (cadr v))
            (define as-tensor (caddr v))
            (bench-case (car v)
                        (lambda (reps)
                          (define last (for/last ([_ (in-range reps)]) (call)))
                          (sync! (as-tensor last)))
                        #:reps (op-reps o)
                        #:before reclaim-native-memory!)))))
