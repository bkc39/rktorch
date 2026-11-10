#lang racket/base

(require (only-in ffi/unsafe _double _fun _int64 _list)
         (only-in ffi/vector _s64vector list->s64vector)
         (only-in threading ~>)
         (only-in torch
                  + [add contracted-add] current-generator make-generator narrow randn select shape
                  stack zeros)
         (only-in torch/foreign/creation-ops [zeros unchecked-zeros])
         (only-in torch/foreign/device-type device)
         (only-in torch/foreign/raw/cas cas-add!)
         (only-in torch/foreign/error check-handle)
         (only-in torch/foreign/raw/collector wake-after-collection!)
         (only-in torch/foreign/raw/fault with-crossing-hook)
         (only-in torch/foreign/raw/memory
                  _tr-device-type account! oom-retry registering tensor-allocator
                  tensor-allocator/no-retry tr-tensor-free/unwrapped)
         (only-in torch/foreign/raw/oom tr-last-error-kind/raw)
         (only-in torch/foreign/raw/pressure collect-under-pressure!)
         (only-in torch/foreign/raw/syntax _Tensor _Tensor/null define-torch)
         (only-in torch/foreign/raw/tensor _tr-dtype)
         (only-in torch/foreign/structs wrap-tensor)
         (only-in torch/foreign/tensor-ops [add unchecked-add])
         (prefix-in g: (only-in torch/generated narrow))
         (only-in torch/tests/private/race-harness with-race-hook)
         (only-in "ops.rkt" settle-native-memory!)
         (only-in "workloads/pipelines.rkt" cifar-augment cifar-sources))

(provide build-case
         op-cost-names
         ladder-ops
         (struct-out op-case)
         op-case-layer
         crossings-per-call
         tensors-per-call
         crossing-thunk)

(struct op-case (name layers) #:transparent)

(define (op-case-layer c name)
  (hash-ref (op-case-layers c) name #f))

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
  tr-last-error-kind/raw)

(define cpu (device 'cpu 0))

;; The library's accounting wakes the reapers inside its pressure check, so
;; every lower layer that registers wakes them the same way: the releases
;; then keep pace with the piece that made the handles, not with the check.
(define ((waking f) . args)
  (begin0 (apply f args) (wake-after-collection!)))

(define ((accounting f) . args)
  (define t (apply f args))
  (when t (account! t))
  t)

(define ((pressure-checked f) . args)
  (define t (apply f args))
  (when t (collect-under-pressure! cpu))
  t)

;; A large op leaves out the layers that hold native memory with no phantom
;; bytes behind it: the Racket collector never sees it, so they price a
;; backlog of major collections rather than the piece.
(define (raw-ladder who c-call args #:large? [large? #f])
  (define (call f) (lambda () (apply f args)))
  (define (wrapping f) (lambda () (wrap-tensor (check-handle who (apply f args)))))
  (define registered (registering c-call))
  (define retried ((oom-retry) registered))
  (define accounted (~> retried accounting waking))
  (define checked (tensor-allocator c-call))
  (define layers
    (hasheq 'raw (lambda () (tr-tensor-free/unwrapped (apply c-call args)))
            'accounted (call accounted)
            'checked (call checked)
            'wrapped (wrapping checked)
            'without-retry (wrapping (tensor-allocator/no-retry c-call))
            'without-pressure (wrapping accounted)))
  (if large?
      layers
      (hash-set* layers
                 'registered (call (waking registered))
                 'retry (call (waking retried))
                 'without-accounting (wrapping (pressure-checked retried)))))

(define (scalar-add-layers)
  (define x (zeros 1))
  (hash-set* (raw-ladder 'add add-scalar/c (list x 1.0))
             'unchecked (lambda () (unchecked-add x 1.0))
             'facade (lambda () (+ x 1.0))
             'contracted (lambda () (contracted-add x 1.0))))

(define (add-8x8-layers)
  (define a (randn 8 8))
  (define b (randn 8 8))
  (hash-set* (raw-ladder 'add add/c (list a b))
             'unchecked (lambda () (unchecked-add a b))
             'facade (lambda () (+ a b))
             'contracted (lambda () (contracted-add a b))))

(define (zeros-layers)
  (define dims (list->s64vector '(8 8)))
  (hash-set* (raw-ladder 'zeros zeros-on/c (list dims 2 'cpu 0 'float32))
             'unchecked (lambda () (unchecked-zeros 8 8))
             'facade (lambda () (zeros 8 8))))

(define (narrow-layers)
  (define x (randn 3 40 40))
  (hash-set* (raw-ladder 'narrow narrow/c (list x 1 4 32))
             'unchecked (lambda () (g:narrow x 1 4 32))
             'facade (lambda () (narrow x 1 4 32))))

(define (stack-layers)
  (define ts (for/list ([_ (in-range 32)]) (randn 3 32 32)))
  (hash-set (raw-ladder 'stack stack/c (list ts 32 0) #:large? #t)
            'facade (lambda () (stack ts))))

(define seeds (box 0))

;; one generator per case, as the loader's workers have one each
(define (cifar-chain-layers)
  (define-values (images _labels) (cifar-sources))
  (define n (~> images shape car))
  (define seed (unbox seeds))
  (set-box! seeds (add1 seed))
  (define generator (make-generator seed))
  (define k (box 0))
  (hasheq 'facade
          (lambda ()
            (define i (unbox k))
            (set-box! k (modulo (add1 i) n))
            (parameterize ([current-generator generator])
              (cifar-augment (select images 0 i))))))

(define op-builders
  (hasheq 'scalar-add scalar-add-layers
          'add-8x8 add-8x8-layers
          'zeros-8x8 zeros-layers
          'narrow narrow-layers
          'stack-32 stack-layers
          'cifar-chain cifar-chain-layers))

(define op-cost-names (sort (hash-keys op-builders) symbol<?))

(define (build-case name)
  (define layers
    (hash-ref op-builders name
              (lambda () (raise-user-error 'op-cost "unknown op ~a; known: ~a" name op-cost-names))))
  (op-case name (layers)))

(define (ladder-ops [names op-cost-names])
  (map build-case names))

;; the deferred frees count too, so the window drains them; they cross on
;; reaper threads and main at once, so the count is a CAS
(define (per-call thunk calls call-counting)
  (define n (box 0))
  (settle-native-memory!)
  (call-counting (lambda () (cas-add! n 1))
                 (lambda ()
                   (for ([_ (in-range calls)]) (thunk))
                   (settle-native-memory!)))
  (/ (unbox n) calls))

(define ((race-point-counting point) bump! run)
  (with-race-hook (lambda (name _subject) (when (eq? name point) (bump!)))
    (run)))

(define (crossings-per-call thunk [calls 64])
  (per-call thunk calls (lambda (bump! run) (with-crossing-hook bump! (run)))))

(define (tensors-per-call thunk [calls 16])
  (per-call thunk calls (race-point-counting 'handle-registered)))

(module+ test
  (require (only-in racket/contract/base has-contract?)
           rackunit
           (only-in torch/foreign/raw/elementwise tr-add/raw))

  (define cases (ladder-ops '(scalar-add add-8x8 stack-32)))

  (define (case-named name)
    (findf (lambda (c) (eq? (op-case-name c) name)) cases))

  (define (charges thunk [calls 8])
    (per-call thunk calls (race-point-counting 'ledger-entry-added)))

  (test-case "every ladder op has every layer, but stack has no unchecked entry point"
    (check-equal? (sort (hash-keys (op-case-layers (case-named 'add-8x8))) symbol<?)
                  '(accounted checked contracted facade raw registered retry unchecked
                              without-accounting without-pressure without-retry wrapped))
    (check-false (op-case-layer (case-named 'stack-32) 'unchecked))
    (check-equal? (hash-keys (op-case-layers (build-case 'cifar-chain))) '(facade)))

  (test-case "stack, 393 KB per op, has no layer holding native memory without phantom bytes"
    (for ([l (in-list '(registered retry without-accounting))])
      (check-false (op-case-layer (case-named 'stack-32) l) (symbol->string l)))
    (check-true (procedure? (op-case-layer (case-named 'stack-32) 'accounted))))

  (test-case "an unknown op is refused with the known ones"
    (check-exn #rx"unknown op nope; known: \\(add-8x8" (lambda () (build-case 'nope))))

  (test-case "each built case is its own: workers share no state"
    (define a (op-case-layer (build-case 'cifar-chain) 'facade))
    (define b (op-case-layer (build-case 'cifar-chain) 'facade))
    (check-false (eq? a b)))

  (test-case "the contracted layer is the contracted add; dispatch and the operator are not"
    (check-true (has-contract? contracted-add))
    (check-false (has-contract? unchecked-add))
    (check-false (has-contract? +)))

  (test-case "the raw layer frees what it makes, in one call and one free"
    (define raw (op-case-layer (case-named 'add-8x8) 'raw))
    (check-equal? (tensors-per-call raw) 0)
    (check-equal? (crossings-per-call raw) 2)
    (check-equal? (charges raw) 0))

  (test-case "a registered layer's crossings count the finalizer's deferred free"
    (check-equal? (crossings-per-call (op-case-layer (case-named 'add-8x8) 'registered)) 2))

  (test-case "the layers charge the ledger where the library's raw op does, and only there"
    (define c (case-named 'add-8x8))
    (define a (randn 8 8))
    (check-equal? (charges (lambda () (tr-add/raw a a))) 1)
    (for ([l (in-list '(registered retry without-accounting))])
      (check-equal? (charges (op-case-layer c l)) 0 (symbol->string l)))
    (for ([l (in-list '(registered retry))])
      (check-equal? (tensors-per-call (op-case-layer c l)) 1 (symbol->string l)))
    (for ([l (in-list '(accounted checked wrapped unchecked facade contracted
                                  without-retry without-pressure))])
      (check-equal? (charges (op-case-layer c l)) 1 (symbol->string l)))))
