#lang racket/base

(require (only-in ffi/unsafe prop:cpointer)
         (only-in ffi/vector
                  f32vector->list
                  f32vector-length
                  f32vector?
                  list->f32vector
                  list->s64vector
                  s64vector->list
                  s64vector-length
                  s64vector?)
         (only-in racket/contract/base
                  ->
                  ->*
                  ->i
                  any/c
                  flat-contract?
                  flat-named-contract
                  integer-in
                  list/c
                  or/c
                  unsupplied-arg?)
         (only-in racket/list append-map)
         (only-in "../private/contract.rkt"
                  define/checked-out
                  define/contract-out)
         (only-in "autograd-ops.rkt" requires-grad!)
         (only-in "device-type.rkt" device/c)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "ops.rkt"
                  device->type+index
                  dims-rest/c
                  dtype/c
                  tensor-device
                  tensor-dtype
                  tensor-shape)
         (only-in "raw/creation.rkt"
                  tr-arange-on/raw
                  tr-eye-on/raw
                  tr-from-data-i64-on-device/raw
                  tr-from-data-i64/raw
                  tr-from-data-on-device/raw
                  tr-from-data/raw
                  tr-full-on/raw
                  tr-ones-on/raw
                  tr-zeros-on/raw)
         (only-in "raw/random.rkt"
                  Generator?
                  tr-generator-draw-seed/raw
                  tr-generator-new/raw
                  tr-rand-on/raw
                  tr-randn-on/raw
                  tr-randperm/raw)
         (only-in "structs.rkt" tensor? wrap-tensor))

(define (wrap who h)
  (wrap-tensor (check-handle who h)))

(define shape-rest/c (or/c (list/c dims-rest/c) dims-rest/c))

(define (shape-of dims)
  (if (and (pair? dims) (list? (car dims))) (car dims) dims))

;; the device and dtype go into native construction — never a default-device
;; scope or a construct-then-move hop through another device
(define (placement device dtype)
  (define-values (type index)
    (if device (device->type+index device) (values 'keep 0)))
  (values type index (or dtype 'keep)))

(define (finish out requires-grad?)
  (if requires-grad? (requires-grad! out) out))

(define float-dtype/c (or/c 'float32 'float64))

(define (shaped who raw dims device dtype requires-grad? . extra)
  (define shape (shape-of dims))
  (define-values (type index dt) (placement device dtype))
  (finish (wrap who
                (apply raw (list->s64vector shape) (length shape)
                       (append extra (list type index dt))))
          requires-grad?))

(define/contract-out (zeros #:device [device #f] #:dtype [dtype #f]
                            #:requires-grad? [requires-grad? #f]
                            . dims)
  (->* [] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'zeros tr-zeros-on/raw dims device dtype requires-grad?))

(define/contract-out (ones #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->* [] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'ones tr-ones-on/raw dims device dtype requires-grad?))

;; the fill crosses the FFI as a double: an int64 fill outside the exact
;; range of a double would round silently, so the contract refuses it
(define (fill-crosses-exactly? value dtype)
  (or (unsupplied-arg? dtype)
      (not (eq? dtype 'int64))
      (not (exact-integer? value))
      (= (exact->inexact value) value)
      "an int64 fill value must be exactly representable as a double"))

(define/contract-out (full value #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->i ([value real?])
       (#:device [device device/c]
        #:dtype [dtype dtype/c]
        #:requires-grad? [requires-grad? boolean?])
       #:rest [dims shape-rest/c]
       #:pre/desc (value dtype) (fill-crosses-exactly? value dtype)
       [result tensor?])
  (shaped 'full tr-full-on/raw dims device dtype requires-grad?
          (exact->inexact value)))

(define/contract-out (randn #:device [device #f] #:dtype [dtype #f]
                            #:requires-grad? [requires-grad? #f]
                            . dims)
  (->* [] [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'randn tr-randn-on/raw dims device dtype requires-grad?))

(define/contract-out (rand #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->* [] [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'rand tr-rand-on/raw dims device dtype requires-grad?))

(struct generator-impl (handle)
  #:reflection-name 'generator
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (_g port _mode) (write-string "#<generator>" port)))

(define/contract-out seed/c flat-contract? ;; noqa
  (flat-named-contract 'seed (integer-in 0 (sub1 (expt 2 64)))))

(define/contract-out size/c flat-contract? ;; noqa
  (flat-named-contract 'size (integer-in 0 (sub1 (expt 2 63)))))

(define/contract-out (make-generator seed) ;; noqa
  (-> seed/c generator?)
  (generator-impl (check-handle 'make-generator (tr-generator-new/raw seed))))

(define/contract-out (generator? v) (-> any/c boolean?) ;; noqa
  (and (generator-impl? v) (Generator? (generator-impl-handle v))))

(define/contract-out (randperm n #:generator [generator #f]) ;; noqa
  (->* [size/c] [#:generator (or/c generator? #f)] tensor?)
  (wrap 'randperm (tr-randperm/raw n generator)))

(define/contract-out (draw-seed #:generator [generator #f]) ;; noqa
  (->* [] [#:generator (or/c generator? #f)] exact-nonnegative-integer?)
  (define-values (rc seed) (tr-generator-draw-seed/raw generator))
  (check-ok rc 'draw-seed)
  seed)

(define (like t device dtype)
  (values (or device (tensor-device t)) (or dtype (tensor-dtype t))))

(define/contract-out (zeros-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                 #:requires-grad? [requires-grad? #f])
  (->* [tensor?] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (zeros (tensor-shape t) #:device dev #:dtype dt
         #:requires-grad? requires-grad?))

(define/contract-out (ones-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor?] [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (ones (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (full-like t value #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor? real?]
       [#:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (full value (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (randn-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                 #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (randn (tensor-shape t) #:device dev #:dtype dt
         #:requires-grad? requires-grad?))

(define/contract-out (rand-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype float-dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (rand (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (arange a [b #f] [c #f]
                             #:device [device #f] #:dtype [dtype #f]
                             #:requires-grad? [requires-grad? #f])
  (->* [real?]
       [real? real?
        #:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (start end step)
    (cond
      [(not b) (values 0 a 1)]
      [(not c) (values a b 1)]
      [else (values a b c)]))
  (define-values (type index dt) (placement device dtype))
  (finish (wrap 'arange
                (tr-arange-on/raw (exact->inexact start)
                                  (exact->inexact end)
                                  (exact->inexact step)
                                  type index dt))
          requires-grad?))

(define/contract-out (eye n [m n]
                          #:device [device #f] #:dtype [dtype #f]
                          #:requires-grad? [requires-grad? #f])
  (->* [exact-nonnegative-integer?]
       [exact-nonnegative-integer?
        #:device device/c #:dtype dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (type index dt) (placement device dtype))
  (finish (wrap 'eye (tr-eye-on/raw n m type index dt)) requires-grad?))

(define (nested-dims data)
  (cond
    [(list? data)
     (if (null? data) '(0) (cons (length data) (nested-dims (car data))))]
    [(vector? data)
     (if (zero? (vector-length data))
         '(0)
         (cons (vector-length data) (nested-dims (vector-ref data 0))))]
    [(f32vector? data) (list (f32vector-length data))]
    [(s64vector? data) (list (s64vector-length data))]
    [else '()]))

(define (sequence-children data)
  (cond
    [(list? data) data]
    [(vector? data) (vector->list data)]
    [(f32vector? data) (f32vector->list data)]
    [(s64vector? data) (s64vector->list data)]
    [else #f]))

(define (sequence-flatten data)
  (define kids (sequence-children data))
  (if kids (append-map sequence-flatten kids) (list data)))

(define (check-regular data dims d)
  (define kids (sequence-children data))
  (cond
    [(null? dims)
     (when kids
       (error 'tensor
              "ragged nested sequence: unexpected sequence at dim ~a: ~e"
              d data))]
    [(not kids)
     (error 'tensor
            (string-append "ragged nested sequence: expected sequence of"
                           " length ~a at dim ~a, got ~e")
            (car dims) d data)]
    [(not (= (length kids) (car dims)))
     (error 'tensor
            (string-append "ragged nested sequence: expected sequence of"
                           " length ~a at dim ~a, got length ~a")
            (car dims) d (length kids))]
    [else
     (for ([kid (in-list kids)])
       (check-regular kid (cdr dims) (add1 d)))]))

(define (exact-int64 x)
  (cond
    [(exact-integer? x) x]
    [(rational? x) (inexact->exact (truncate x))]
    [else
     (error 'tensor "cannot convert non-finite value to int64: ~e" x)]))

(define (infer-dtype flat)
  (cond
    [(null? flat) 'float32]
    [(andmap exact-integer? flat) 'int64]
    [else 'float32]))

(define/checked-out (tensor data
                            #:requires-grad? [requires-grad? #f]
                            #:device [device #f]
                            #:dtype [dtype #f])
  (->* [(or/c real? list? vector? f32vector? s64vector?)]
       [#:requires-grad? boolean?
        #:device (or/c #f device/c)
        #:dtype (or/c #f 'float32 'int64)]
       tensor?)
  (unless (memq dtype '(#f float32 int64))
    (error 'tensor "unsupported #:dtype (float32 or int64): ~e" dtype))
  (define dims (nested-dims data))
  (define-values (chosen payload numel)
    (cond
      [(and (f32vector? data) (not (eq? dtype 'int64)))
       (values 'float32 data (f32vector-length data))]
      [(and (s64vector? data) (not (eq? dtype 'float32)))
       (values 'int64 data (s64vector-length data))]
      [else
       (check-regular data dims 0)
       (define flat (sequence-flatten data))
       (case (or dtype (infer-dtype flat))
         [(int64)
          (values 'int64
                  (list->s64vector (map exact-int64 flat))
                  (length flat))]
         [else
          (values 'float32
                  (list->f32vector (map exact->inexact flat))
                  (length flat))])]))
  (define dim-vec (list->s64vector dims))
  (define-values (type index)
    (if device (device->type+index device) (values #f #f)))
  (define out
    (wrap 'tensor
          (case chosen
            [(int64)
             (if device
                 (tr-from-data-i64-on-device/raw payload numel
                                          dim-vec (length dims)
                                          type index)
                 (tr-from-data-i64/raw payload numel
                                       dim-vec (length dims)))]
            [else
             (if device
                 (tr-from-data-on-device/raw payload numel
                                      dim-vec (length dims)
                                      type index)
                 (tr-from-data/raw payload numel
                                   dim-vec (length dims)))])))
  (if requires-grad? (requires-grad! out) out))
