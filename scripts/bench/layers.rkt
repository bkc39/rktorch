#lang racket/base

(require (only-in ffi/unsafe _fun _void get-ffi-obj)
         (only-in ffi/unsafe/alloc allocator)
         (only-in torch [add facade-add] randn reclaim-native-memory!)
         (only-in torch/foreign/error check-handle)
         (only-in torch/foreign/raw/fault latched)
         (only-in torch/foreign/raw/memory
                  collect-and-drain!
                  tensor-allocator
                  tensor-allocator/no-retry
                  tr-tensor-free/finalizer)
         (only-in torch/foreign/raw/syntax _Tensor _Tensor/null)
         (only-in torch/foreign/structs wrap-tensor)
         (only-in torch/foreign/tensor-ops [add unchecked-add])
         (only-in "harness.rkt" bench-case)
         (only-in "shim.rkt" shim-library))

(provide layer-cases
         layer-order)

(define tr-add (get-ffi-obj "tr_add" shim-library (_fun _Tensor _Tensor -> _Tensor/null)))
(define tr-free (get-ffi-obj "tr_tensor_free" shim-library (_fun _Tensor -> _void)))

(define tr-add/latched (latched tr-add))
(define tr-free/latched (latched tr-free))
(define tr-add/allocator ((allocator tr-tensor-free/finalizer) tr-add/latched))
(define tr-add/accounted (tensor-allocator/no-retry tr-add/latched))
(define tr-add/retry (tensor-allocator tr-add/latched))

(define (add/shape a b)
  (wrap-tensor (check-handle 'add (tr-add/retry a b))))

(define layer-order
  '(ffi-call fault-latch allocator accounting oom-retry shape-readback
    dispatch facade-contract))

(define (layer-cases #:reps [reps 20000] #:drain [drain 2000])
  (define a (randn 8 8))
  (define b (randn 8 8))
  (define ((calls f) n)
    (for ([_ (in-range n)])
      (f)))
  (define (case name f)
    (bench-case name (calls f) #:reps reps #:before reclaim-native-memory!))
  (list
   (case 'ffi-call (lambda () (tr-free (tr-add a b))))
   (case 'fault-latch (lambda () (tr-free/latched (tr-add/latched a b))))
   (case 'allocator (lambda () (tr-add/allocator a b)))
   (case 'accounting (lambda () (tr-add/accounted a b)))
   (case 'oom-retry (lambda () (tr-add/retry a b)))
   (case 'shape-readback (lambda () (add/shape a b)))
   (case 'dispatch (lambda () (unchecked-add a b)))
   (case 'facade-contract (lambda () (facade-add a b)))
   (bench-case 'finalization
               (lambda (n)
                 (collect-and-drain!)
                 (define t0 (current-inexact-monotonic-milliseconds))
                 (collect-and-drain!)
                 (define empty (- (current-inexact-monotonic-milliseconds) t0))
                 (for ([_ (in-range n)]) (tr-add/retry a b))
                 (define t1 (current-inexact-monotonic-milliseconds))
                 (collect-and-drain!)
                 (- (current-inexact-monotonic-milliseconds) t1 empty))
               #:reps drain
               #:self-timed? #t)))
