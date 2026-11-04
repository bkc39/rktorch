#lang racket/base

(require (only-in ffi/unsafe _double _fun _int _int64 _list _ptr _void)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in racket/list make-list)
         (only-in torch + [add contracted-add] narrow randn select shape stack zeros)
         (only-in torch/foreign/creation-ops [zeros unchecked-zeros])
         (only-in torch/foreign/device-type device)
         (only-in torch/foreign/error check-handle)
         (only-in torch/foreign/raw/fault with-crossing-hook)
         (only-in torch/foreign/raw/memory
                  _tr-device-type oom-retry record-allocation! registering tensor-allocator
                  tr-tensor-device/raw)
         (only-in torch/foreign/raw/syntax _Tensor _Tensor/null define-torch)
         (only-in torch/foreign/raw/tensor _tr-dtype)
         (only-in torch/foreign/structs wrap-tensor)
         (only-in torch/foreign/tensor-ops [add unchecked-add])
         (prefix-in g: (only-in torch/generated narrow))
         (only-in torch/tests/private/race-harness with-race-hook)
         (only-in "op-cost-layers.rkt" op-cost-names)
         (only-in "workloads/pipelines.rkt" cifar-augment cifar-sources))

(provide ladder-ops
         (struct-out op-case)
         op-case-layer
         crossings-per-call
         tensors-per-call
         crossing-thunk)

(struct op-case (name layers) #:transparent)

(define (op-case-layer c name)
  (hash-ref (op-case-layers c) name #f))

(define-torch tr-tensor-free/bench
  (_fun _Tensor -> _void)
  #:c-id tr_tensor_free)

(define-torch tr-tensor-nbytes/bench
  (_fun _Tensor (out : (_ptr o _int64)) -> (rc : _int) -> (values rc out))
  #:c-id tr_tensor_nbytes)

(define-torch tr-last-error-kind/bench
  (_fun -> _int)
  #:c-id tr_last_error_kind)

(define-torch add-scalar/c
  (_fun _Tensor _double -> _Tensor/null)
  #:c-id tr_add_scalar)

(define-torch add/c
  (_fun _Tensor _Tensor -> _Tensor/null)
  #:c-id tr_add)

(define-torch zeros-on/c
  (_fun _s64vector _int64 _tr-device-type _int64 _tr-dtype -> _Tensor/null)
  #:c-id tr_zeros_on)

(define-torch narrow/c
  (_fun _Tensor _int64 _int64 _int64 -> _Tensor/null)
  #:c-id tr_gen_narrow)

(define-torch stack/c
  (_fun (ts : (_list i _Tensor)) _int64 _int64 -> _Tensor/null)
  #:c-id tr_stack)

(define (crossing-thunk)
  tr-last-error-kind/bench)

(define (placement t)
  (with-handlers ([exn:fail? (lambda (_e) (values #f #f))])
    (define-values (nb-rc nbytes) (tr-tensor-nbytes/bench t))
    (define-values (dev-rc type index) (tr-tensor-device/raw t))
    (cond
      [(and (zero? nb-rc) (zero? dev-rc))
       (values nbytes (device type (if (eq? type 'cpu) 0 index)))]
      [else (values #f #f)])))

(define ((accounting wrapped) . args)
  (define t (apply wrapped args))
  (when t
    (define-values (nbytes dev) (placement t))
    (when dev
      (record-allocation! t nbytes dev)))
  t)

(define (raw-ladder who c-call args)
  (define registered (registering c-call))
  (define retry ((oom-retry) (registering c-call)))
  (define accounted (accounting ((oom-retry) (registering c-call))))
  (define checked (tensor-allocator c-call))
  (hasheq 'raw (lambda () (tr-tensor-free/bench (apply c-call args)))
          'registered (lambda () (apply registered args))
          'retry (lambda () (apply retry args))
          'accounted (lambda () (apply accounted args))
          'checked (lambda () (apply checked args))
          'wrapped (lambda () (wrap-tensor (check-handle who (apply checked args))))))

(define (with-layers base . kvs)
  (let loop ([h base] [kvs kvs])
    (cond
      [(null? kvs) h]
      [else (loop (hash-set h (car kvs) (cadr kvs)) (cddr kvs))])))

(define (scalar-add-case)
  (define x (zeros 1))
  (op-case 'scalar-add
           (with-layers (raw-ladder 'add add-scalar/c (list x 1.0))
             'unchecked (lambda () (unchecked-add x 1.0))
             'facade (lambda () (+ x 1.0))
             'contracted (lambda () (contracted-add x 1.0)))))

(define (add-8x8-case)
  (define a (randn 8 8))
  (define b (randn 8 8))
  (op-case 'add-8x8
           (with-layers (raw-ladder 'add add/c (list a b))
             'unchecked (lambda () (unchecked-add a b))
             'facade (lambda () (+ a b))
             'contracted (lambda () (contracted-add a b)))))

(define (zeros-case)
  (define dims (list->s64vector '(8 8)))
  (op-case 'zeros-8x8
           (with-layers (raw-ladder 'zeros zeros-on/c (list dims 2 'cpu 0 'float32))
             'unchecked (lambda () (unchecked-zeros 8 8))
             'facade (lambda () (zeros 8 8)))))

(define (narrow-case)
  (define x (randn 3 40 40))
  (op-case 'narrow
           (with-layers (raw-ladder 'narrow narrow/c (list x 1 4 32))
             'unchecked (lambda () (g:narrow x 1 4 32))
             'facade (lambda () (narrow x 1 4 32)))))

(define (stack-case)
  (define ts (make-list 32 (randn 3 32 32)))
  (op-case 'stack-32
           (with-layers (raw-ladder 'stack stack/c (list ts 32 0))
             'facade (lambda () (stack ts)))))

(define (cifar-chain-case)
  (define-values (images _labels) (cifar-sources))
  (define n (car (shape images)))
  (define k (box 0))
  (op-case 'cifar-chain
           (hasheq 'facade
                   (lambda ()
                     (set-box! k (modulo (add1 (unbox k)) n))
                     (cifar-augment (select images 0 (unbox k)))))))

(define (ladder-ops [names op-cost-names])
  (for/list ([name (in-list names)])
    ((case name
       [(scalar-add) scalar-add-case]
       [(add-8x8) add-8x8-case]
       [(zeros-8x8) zeros-case]
       [(narrow) narrow-case]
       [(stack-32) stack-case]
       [(cifar-chain) cifar-chain-case]
       [else (raise-user-error 'op-cost "unknown op ~a; known: ~a" name op-cost-names)]))))

(define (crossings-per-call thunk [calls 64])
  (define n (box 0))
  (with-crossing-hook (lambda () (set-box! n (add1 (unbox n))))
    (for ([_ (in-range calls)]) (thunk)))
  (/ (unbox n) calls))

(define (tensors-per-call thunk [calls 16])
  (define n (box 0))
  (with-race-hook (lambda (name _subject)
                    (when (eq? name 'handle-registered) (set-box! n (add1 (unbox n)))))
    (for ([_ (in-range calls)]) (thunk)))
  (/ (unbox n) calls))

(module+ test
  (require rackunit
           (only-in torch/foreign/raw/elementwise tr-add/raw))

  (define cases (ladder-ops '(scalar-add add-8x8 stack-32)))

  (define (case-named name)
    (for/first ([c (in-list cases)] #:when (eq? (op-case-name c) name)) c))

  (define (charges thunk [calls 8])
    (define n (box 0))
    (with-race-hook (lambda (name _subject)
                      (when (eq? name 'ledger-entry-added) (set-box! n (add1 (unbox n)))))
      (for ([_ (in-range calls)]) (thunk)))
    (/ (unbox n) calls))

  (test-case "every ladder op has every layer, but stack has no unchecked entry point"
    (check-equal? (sort (hash-keys (op-case-layers (case-named 'add-8x8))) symbol<?)
                  '(accounted checked contracted facade raw registered retry unchecked wrapped))
    (check-false (op-case-layer (case-named 'stack-32) 'unchecked))
    (check-equal? (hash-keys (op-case-layers (car (ladder-ops '(cifar-chain))))) '(facade)))

  (test-case "the raw layer frees what it makes, in one call and one free"
    (define raw (op-case-layer (case-named 'add-8x8) 'raw))
    (check-equal? (tensors-per-call raw) 0)
    (check-equal? (crossings-per-call raw) 2)
    (check-equal? (charges raw) 0))

  (test-case "the layers charge the ledger where the library's raw op does, and only there"
    (define c (case-named 'add-8x8))
    (define a (randn 8 8))
    (check-equal? (charges (lambda () (tr-add/raw a a))) 1)
    (for ([l (in-list '(registered retry))])
      (check-equal? (charges (op-case-layer c l)) 0 (symbol->string l))
      (check-equal? (tensors-per-call (op-case-layer c l)) 1 (symbol->string l)))
    (for ([l (in-list '(accounted checked wrapped unchecked facade contracted))])
      (check-equal? (charges (op-case-layer c l)) 1 (symbol->string l)))))
