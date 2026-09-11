#lang racket/base

(require ;; whole-module: the pattern's syntax classes live at phase 1 and
         ;; only-in would strip them
         syntax/parse/define
         (only-in ffi/vector
                  f32vector->list
                  f32vector?
                  f64vector->list
                  f64vector?
                  list->s64vector
                  make-f32vector
                  make-f64vector
                  make-s64vector
                  s64vector->list
                  s64vector-ref
                  s64vector?)
         (only-in racket/contract/base
                  -> ->* ->i any any/c cons/c contract-out list/c listof none/c
                  or/c)
         (only-in "../private/contract.rkt"
                  define/checked-out define/contract-out)
         (only-in "device-type.rkt"
                  [device make-device]
                  cpu-device cuda-device device-index device-type device/c
                  device? mps-device)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "raw/device.rkt"
                  tr-cuda-device-count/raw
                  tr-cuda-empty-cache/raw
                  tr-cuda-is-available/raw
                  tr-cuda-memory-stats/raw
                  tr-get-default-device/raw
                  tr-mps-empty-cache/raw
                  tr-mps-is-available/raw
                  tr-set-default-device/raw
                  tr-tensor-device/raw
                  tr-tensor-to!/raw
                  tr-tensor-to/raw)
         (only-in "raw/global.rkt" tr-manual-seed/raw tr-version/raw)
         (only-in "raw/memory.rkt"
                  collect-and-drain!
                  [finalizer-diagnostics raw:finalizer-diagnostics]
                  [finalizer-failures raw:finalizer-failures]
                  [native-memory-use raw:native-memory-use]
                  oom-retry/status
                  reaccount!)
         (only-in "raw/random.rkt" tr-rand/raw tr-randn/raw tr-tensor-uniform!/raw)
         (only-in "raw/tensor.rkt"
                  dtype-code->symbol
                  tr-tensor-copy-data-f64/raw
                  tr-tensor-copy-data-i64/raw
                  tr-tensor-copy-data/raw
                  tr-tensor-dtype/raw
                  tr-tensor-item/raw
                  tr-tensor-numel/raw)
         (only-in "structs.rkt"
                  handle->repr
                  handle->string
                  tensor-handle
                  tensor-impl-shape
                  tensor?
                  wrap-tensor))

(provide device->type+index
         dims-rest/c
         dtype/c
         prop:to
         with-default-device)

(module+ unsafe
  (provide (contract-out
            [to! (->i ([t tensor?] [target (or/c device/c dtype/c)])
                      ([dtype (target) (dtype-after/c target)])
                      [result tensor?])])))

(define dims-rest/c (listof exact-nonnegative-integer?))

(define dtype-symbols '(float32 float64 int64 bool))

(define dtype/c (apply or/c dtype-symbols))

;; Python's argument order: a dtype target stands alone, a device target may
;; carry a dtype — the shape gets contract blame, not a runtime error
(define (dtype-after/c target)
  (if (memq target dtype-symbols) none/c dtype/c))

(define/contract-out (torch-version) (-> string?) ;; noqa
  (tr-version/raw))

(define/contract-out (manual-seed! seed) (-> exact-nonnegative-integer? void?)
  (check-ok (tr-manual-seed/raw seed) 'manual-seed!)
  (void))

(define/contract-out (randn . dims)
  (->* [] #:rest dims-rest/c tensor?)
  (wrap-tensor
   (check-handle 'randn (tr-randn/raw (list->s64vector dims) (length dims)))))

(define/contract-out (rand . dims)
  (->* [] #:rest dims-rest/c tensor?)
  (wrap-tensor
   (check-handle 'rand (tr-rand/raw (list->s64vector dims) (length dims)))))

(define/contract-out (uniform! t low high) (-> tensor? real? real? void?)
  (check-ok (tr-tensor-uniform!/raw t
                                    (exact->inexact low)
                                    (exact->inexact high))
            'uniform!)
  (void))

(define/checked-out (item t) (-> tensor? real?)
  (cond
    [(and (int64-tensor? t) (= 1 (tensor-numel t)))
     (define out (make-s64vector 1))
     (define-values (rc _n) (tr-tensor-copy-data-i64/raw t 1 out))
     (check-ok rc 'item)
     (s64vector-ref out 0)]
    [else
     (define-values (rc v) (tr-tensor-item/raw t))
     (check-ok rc 'item)
     v]))

(define/checked-out (to-dtype t dtype) (-> tensor? dtype/c tensor?) ;; noqa
  (to t dtype))

(define/checked-out (tensor-dtype t) (-> tensor? dtype/c)
  (define-values (rc code) (tr-tensor-dtype/raw t))
  (check-ok rc 'tensor-dtype)
  (or (dtype-code->symbol code)
      (error 'tensor-dtype "unsupported dtype code: ~a" code)))

(define (int64-tensor? t)
  (define-values (rc code) (tr-tensor-dtype/raw t))
  (and (zero? rc) (eq? (dtype-code->symbol code) 'int64)))

(define (device->type+index dev)
  (cond
    [(device? dev) (values (device-type dev) (device-index dev))]
    [(eq? dev 'cpu) (values 'cpu 0)]
    [(eq? dev 'cuda) (values 'cuda 0)]
    [(eq? dev 'mps) (values 'mps 0)]
    [(pair? dev) (values 'cuda (cadr dev))]
    [else (error 'device "unsupported device: ~e" dev)]))

(define (type+index->device type index)
  (case type
    [(cpu) (cpu-device)]
    [(mps) (mps-device)]
    [else (cuda-device index)]))

(define/contract-out (cuda-available?) (-> boolean?)
  (= 1 (tr-cuda-is-available/raw)))

(define/contract-out (cuda-if-available) (-> device?) ;; noqa
  (if (cuda-available?) (cuda-device) (cpu-device)))

(define/contract-out (mps-available?) (-> boolean?)
  (= 1 (tr-mps-is-available/raw)))

(define/contract-out (mps-if-available) (-> device?) ;; noqa
  (if (mps-available?) (mps-device) (cpu-device)))

;; CUDA before MPS only for determinism; the two never coexist (MPS is
;; darwin-only, and nixpkgs has no darwin CUDA libtorch).
(define/contract-out (accelerator-if-available) (-> device?) ;; noqa
  (cond
    [(cuda-available?) (cuda-device)]
    [(mps-available?) (mps-device)]
    [else (cpu-device)]))

(define/contract-out (cuda-device-count) (-> exact-nonnegative-integer?) ;; noqa
  (tr-cuda-device-count/raw))

(define/contract-out (cuda-memory-stats [dev (cuda-device)])
  (->* [] [device/c]
       (listof (cons/c (or/c 'allocated 'reserved 'peak-allocated)
                       exact-nonnegative-integer?)))
  (define-values (type index) (device->type+index dev))
  (unless (eq? type 'cuda)
    (error 'cuda-memory-stats "expected a CUDA device, given: ~e" dev))
  (define-values (rc allocated reserved peak)
    (tr-cuda-memory-stats/raw index))
  (check-ok rc 'cuda-memory-stats)
  (list (cons 'allocated allocated)
        (cons 'reserved reserved)
        (cons 'peak-allocated peak)))

(define/contract-out (cuda-empty-cache!) (-> void?)
  (check-ok (tr-cuda-empty-cache/raw) 'cuda-empty-cache!)
  (void))

(define/contract-out (mps-empty-cache!) (-> void?)
  (check-ok (tr-mps-empty-cache/raw) 'mps-empty-cache!)
  (void))

;; a handle-attributed estimate: views charge their full extents (shared
;; storage double-counts) and ATen-internal allocations are absent
(define/contract-out native-memory-use
  (-> (listof (cons/c device? exact-nonnegative-integer?)))
  raw:native-memory-use)

(define/contract-out finalizer-failures ;; noqa
  (-> exact-nonnegative-integer?)
  raw:finalizer-failures)

(define/contract-out finalizer-diagnostics ;; noqa
  (-> (list/c (cons/c 'runs exact-nonnegative-integer?)
              (cons/c 'failures exact-nonnegative-integer?)
              (cons/c 'messages (listof string?))
              (cons/c 'ledger-entries exact-nonnegative-integer?)))
  raw:finalizer-diagnostics)

(define/contract-out (reclaim-native-memory!) (-> void?) ;; noqa
  (let loop ([prev (ledger-total)] [rounds 4])
    (define drained? (collect-and-drain!))
    (define now (ledger-total))
    ;; a busy executor can make zero ledger progress in one round without
    ;; being done, so no-progress only terminates after a drained round
    (when (and (> rounds 1)
               (or (< now prev) (not drained?)))
      (loop now (sub1 rounds))))
  (cuda-empty-cache!)
  (mps-empty-cache!))

(define (ledger-total)
  (for/sum ([entry (in-list (native-memory-use))])
    (cdr entry)))

(define/contract-out (set-default-device! dev) (-> device/c void?)
  (define-values (type index) (device->type+index dev))
  (check-ok (tr-set-default-device/raw type index) 'set-default-device!)
  (void))

(define/contract-out (default-device) (-> device?)
  (define-values (rc type index) (tr-get-default-device/raw))
  (check-ok rc 'default-device)
  (type+index->device type index))

(define/contract-out (call-with-default-device dev thunk)
  (-> device/c (-> any) any)
  (define saved (default-device))
  (dynamic-wind (lambda () (set-default-device! dev))
                thunk
                (lambda () (set-default-device! saved))))

(define-syntax-parse-rule (with-default-device dev:expr body:expr ...+)
  (call-with-default-device dev (lambda () body ...)))

(define/checked-out (to-device t dev) (-> tensor? device/c tensor?) ;; noqa
  (to t dev))

;; 'keep is the C side's "leave this axis" sentinel; the argument-order
;; check backs the uncontracted in-package entry
(define (parse-target who target dtype)
  (cond
    [(memq target dtype-symbols)
     (when dtype
       (raise-arguments-error who "a dtype target takes no second argument"
                              "target" target "dtype" dtype))
     (values 'keep 0 target)]
    [else
     (define-values (type index) (device->type+index target))
     (values type index (or dtype 'keep))]))

(define (already-there? t type index dtype)
  (and (or (eq? type 'keep)
           (equal? (type+index->device type index) (tensor-device t)))
       (or (eq? dtype 'keep) (eq? dtype (tensor-dtype t)))))

(define-values (prop:to to-able?* to-ref)
  (make-struct-type-property
   'to
   (lambda (v _info)
     (unless (and (procedure? v) (procedure-arity-includes? v 3))
       (raise-argument-error 'prop:to "(procedure-arity-includes/c 3)" v))
     v)))

(define/contract-out (to x target [dtype #f]) ;; noqa
  (->i ([x (or/c tensor? to-able?)] [target (or/c device/c dtype/c)])
       ([dtype (target) (dtype-after/c target)])
       [result (or/c tensor? to-able?)])
  (define-values (type index dt) (parse-target 'to target dtype))
  (cond
    [(tensor? x)
     ;; x.to(...) is x when nothing changes; a fresh wrapper over the aliased
     ;; storage would charge the ledger twice
     (if (already-there? x type index dt)
         x
         (wrap-tensor (check-handle 'to (tr-tensor-to/raw x type index dt))))]
    [else
     ((to-ref x) x
                 (and (not (eq? type 'keep)) (type+index->device type index))
                 (and (not (eq? dt 'keep)) dt))]))

(define/contract-out (to-able? v) (-> any/c boolean?) ;; noqa
  (to-able?* v))

(define tensor-to!/retrying ((oom-retry/status) tr-tensor-to!/raw))

;; the status is judged first, while the native error is fresh, and the
;; handle is re-accounted whichever way that goes: the ledger entry
;; describes the handle as it now is
(define (to! t target [dtype #f])
  (define-values (type index dt) (parse-target 'to! target dtype))
  (define rc (tensor-to!/retrying t type index dt))
  (dynamic-wind void
                (lambda () (check-ok rc 'to!))
                (lambda () (reaccount! (tensor-handle t))))
  t)

(define/checked-out (tensor-device t) (-> tensor? device?)
  (define-values (rc type index) (tr-tensor-device/raw t))
  (check-ok rc 'tensor-device)
  (type+index->device type index))

(define/contract-out (tensor-numel t) (-> tensor? exact-nonnegative-integer?)
  (define-values (rc n) (tr-tensor-numel/raw t))
  (check-ok rc 'tensor-numel)
  n)

(define/checked-out (tensor-shape t)
  (-> tensor? (listof exact-nonnegative-integer?))
  (tensor-impl-shape t))

(define/contract-out shape ;; noqa
  (-> tensor? (listof exact-nonnegative-integer?))
  tensor-shape)
(define/contract-out dtype (-> tensor? dtype/c) tensor-dtype) ;; noqa
(define/contract-out numel (-> tensor? exact-nonnegative-integer?) tensor-numel) ;; noqa

(define/contract-out (device x [index #f])
  (->i ([target (or/c tensor? 'cpu 'cuda 'mps)])
       ([index (target)
               (case target
                 [(cuda) exact-nonnegative-integer?]
                 [(cpu mps) 0]
                 [else none/c])])
       [result device?])
  (cond
    [(tensor? x)
     (when index
       (error 'device "an ordinal makes no sense when querying a tensor: ~e"
              index))
     (tensor-device x)]
    [else (make-device x (or index 0))]))

(define/contract-out (tensor->vector t)
  (-> tensor? (or/c f32vector? f64vector? s64vector?))
  (define n (tensor-numel t))
  (define-values (dtype-rc code) (tr-tensor-dtype/raw t))
  (define dt (and (zero? dtype-rc) (dtype-code->symbol code)))
  (cond
    [(eq? dt 'float64)
     (define out (make-f64vector n))
     (define-values (rc _n) (tr-tensor-copy-data-f64/raw t n out))
     (check-ok rc 'tensor->vector)
     out]
    [(eq? dt 'int64)
     (define out (make-s64vector n))
     (define-values (rc _n) (tr-tensor-copy-data-i64/raw t n out))
     (check-ok rc 'tensor->vector)
     out]
    [else
     (define out (make-f32vector n))
     (define-values (rc _n) (tr-tensor-copy-data/raw t n out))
     (check-ok rc 'tensor->vector)
     out]))

(define/checked-out (tensor->list t) (-> tensor? (listof real?)) ;; noqa
  (define v (tensor->vector t))
  (cond
    [(s64vector? v) (s64vector->list v)]
    [(f64vector? v) (f64vector->list v)]
    [else (f32vector->list v)]))

(define/contract-out (tensor->string t) (-> tensor? string?) ;; noqa
  (handle->string (tensor-handle t)))

(define/contract-out (tensor->repr t) (-> tensor? string?) ;; noqa
  (handle->repr (tensor-handle t) (tensor-shape t)))
