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
                  any-float-dtype/c
                  default-device
                  device->type+index
                  dims-rest/c
                  dtype/c
                  placement
                  tensor-device
                  tensor-dtype
                  tensor-shape
                  to-device
                  to-dtype)
         (only-in "raw/creation.rkt"
                  tr-arange-on/raw
                  tr-eye-on/raw
                  tr-from-bytes-on-device/raw
                  tr-from-bytes/raw
                  tr-from-data-i64-on-device/raw
                  tr-from-data-i64/raw
                  tr-from-data-on-device/raw
                  tr-from-data-u8-on-device/raw
                  tr-from-data-u8/raw
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

(define (finish out requires-grad?)
  (if requires-grad? (requires-grad! out) out))

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
;; range of a double would round silently and a uint8 fill outside 0..255
;; would wrap, so the contract refuses both
(define (fill-crosses-exactly? value dtype)
  (cond
    [(unsupplied-arg? dtype) #t]
    [(eq? dtype 'int64)
     (or (not (exact-integer? value))
         (= (exact->inexact value) value)
         "an int64 fill value must be exactly representable as a double")]
    [(eq? dtype 'uint8)
     (or (and (integer? value) (<= 0 value 255))
         "a uint8 fill value must be an integer from 0 to 255")]
    [else #t]))

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
  (->* [] [#:device device/c #:dtype any-float-dtype/c #:requires-grad? boolean?]
       #:rest shape-rest/c tensor?)
  (shaped 'randn tr-randn-on/raw dims device dtype requires-grad?))

(define/contract-out (rand #:device [device #f] #:dtype [dtype #f]
                           #:requires-grad? [requires-grad? #f]
                           . dims)
  (->* [] [#:device device/c #:dtype any-float-dtype/c #:requires-grad? boolean?]
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

(define/checked-out (generator? v) (-> any/c boolean?) ;; noqa
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
  (define verdict (fill-crosses-exactly? value dt))
  (unless (eq? verdict #t)
    (raise-argument-error 'full-like verdict value))
  (full value (tensor-shape t) #:device dev #:dtype dt
        #:requires-grad? requires-grad?))

(define/contract-out (randn-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                 #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype any-float-dtype/c #:requires-grad? boolean?]
       tensor?)
  (define-values (dev dt) (like t device dtype))
  (randn (tensor-shape t) #:device dev #:dtype dt
         #:requires-grad? requires-grad?))

(define/contract-out (rand-like t #:device [device #f] #:dtype [dtype #f] ;; noqa
                                #:requires-grad? [requires-grad? #f])
  (->* [tensor?]
       [#:device device/c #:dtype any-float-dtype/c #:requires-grad? boolean?]
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
    [(bytes? data) (list (bytes-length data))]
    [else '()]))

(define (sequence-children data)
  (cond
    [(list? data) data]
    [(vector? data) (vector->list data)]
    [(f32vector? data) (f32vector->list data)]
    [(s64vector? data) (s64vector->list data)]
    [(bytes? data) (bytes->list data)]
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

(define (exact-byte x)
  (define v
    (cond
      [(exact-integer? x) x]
      [(rational? x) (inexact->exact (truncate x))]
      [else (error 'tensor "cannot convert non-finite value to uint8: ~e" x)]))
  (unless (byte? v)
    (error 'tensor "cannot convert value to uint8 (0 to 255): ~e" x))
  v)

(define (infer-dtype flat)
  (cond
    [(null? flat) 'float32]
    [(andmap exact-integer? flat) 'int64]
    [else 'float32]))

(define/checked-out (tensor data
                            #:requires-grad? [requires-grad? #f]
                            #:device [device #f]
                            #:dtype [dtype #f])
  (->* [(or/c real? list? vector? f32vector? s64vector? bytes?)]
       [#:requires-grad? boolean?
        #:device (or/c #f device/c)
        #:dtype (or/c #f 'float32 'int64 'uint8 'float16 'bfloat16)]
       tensor?)
  (unless (memq dtype '(#f float32 int64 uint8 float16 bfloat16))
    (error 'tensor
           "unsupported #:dtype (float32, int64, uint8, float16 or bfloat16): ~e"
           dtype))
  (define dims (nested-dims data))
  (define-values (chosen payload numel)
    (cond
      [(bytes? data) (values 'uint8 data (bytes-length data))]
      [(and (f32vector? data) (not (memq dtype '(int64 uint8))))
       (values 'float32 data (f32vector-length data))]
      [(and (s64vector? data) (not (memq dtype '(float32 uint8))))
       (values 'int64 data (s64vector-length data))]
      [else
       (check-regular data dims 0)
       (define flat (sequence-flatten data))
       (case (or dtype (infer-dtype flat))
         [(int64)
          (values 'int64
                  (list->s64vector (map exact-int64 flat))
                  (length flat))]
         [(uint8)
          (values 'uint8 (list->bytes (map exact-byte flat)) (length flat))]
         [else
          (values 'float32
                  (list->f32vector (map exact->inexact flat))
                  (length flat))])]))
  (define dim-vec (list->s64vector dims))
  (define ndim (length dims))
  (define narrow?
    (or (and (bytes? data) dtype (not (eq? dtype 'uint8)) #t)
        (and (memq dtype '(float16 bfloat16)) #t)))
  ;; only the half pair is built wide and cast down, so only it is staged on
  ;; the CPU, where no accelerator holds the wide copy and the narrow one at
  ;; once; bytes are built at their own size and only widen, so they are
  ;; built where they are wanted and widened there
  (define stage? (and narrow? (not (bytes? data))))
  ;; the destination is fixed before the build, as every other constructor
  ;; fixes it, so a default that changes meanwhile does not move the result
  (define lands-on (and stage? (or device (default-device))))
  (define build-on (if stage? 'cpu device))
  (define-values (type index)
    (if build-on (device->type+index build-on) (values #f #f)))
  (define out
    (wrap 'tensor
          (case chosen
            [(int64)
             (if build-on
                 (tr-from-data-i64-on-device/raw payload numel dim-vec ndim
                                                 type index)
                 (tr-from-data-i64/raw payload numel dim-vec ndim))]
            [(uint8)
             (if build-on
                 (tr-from-data-u8-on-device/raw payload numel dim-vec ndim
                                                type index)
                 (tr-from-data-u8/raw payload numel dim-vec ndim))]
            [else
             (if build-on
                 (tr-from-data-on-device/raw payload numel dim-vec ndim
                                             type index)
                 (tr-from-data/raw payload numel dim-vec ndim))])))
  ;; the half pair has no host vector type, so it is built as float32 and
  ;; narrowed natively, like a byte string asked for another dtype
  (define typed
    (cond
      [stage? (to-device (to-dtype out dtype) lands-on)]
      [narrow? (to-dtype out dtype)]
      [else out]))
  (if requires-grad? (requires-grad! typed) typed))

(define/contract-out (bytes->tensor bs dtype shape #:device [device #f]) ;; noqa
  (->* [bytes? dtype/c dims-rest/c] [#:device (or/c #f device/c)] tensor?)
  (define dims (list->s64vector shape))
  (wrap 'bytes->tensor
        (cond
          [device
           (define-values (type index) (device->type+index device))
           (tr-from-bytes-on-device/raw bs (bytes-length bs) dims (length shape)
                                        dtype type index)]
          [else
           (tr-from-bytes/raw bs (bytes-length bs) dims (length shape)
                              dtype)])))
