#lang racket/base

(require (for-syntax racket/base
                     (only-in racket/syntax format-id)
                     (only-in syntax/parse/pre
                              define-syntax-class id pattern))
         (only-in ffi/unsafe prop:cpointer)
         (only-in racket/contract/base
                  -> ->* ->i >=/c flat-contract? flat-named-contract
                  integer-in or/c parameter/c)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out supplied-or)
         (only-in "device-type.rkt" device-type device/c device?)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "ops.rkt"
                  default-device
                  device->type+index
                  float-dtype-max
                  tensor-device
                  tensor-dtype
                  tensor-shape
                  type+index->device)
         (only-in "raw/random.rkt"
                  current-generator/raw
                  tr-default-generator-set-state!/raw
                  tr-default-generator-state/raw
                  tr-generator-device/raw
                  tr-generator-new-on/raw
                  tr-generator-set-state!/raw
                  tr-generator-state/raw
                  tr-tensor-normal!/raw
                  tr-tensor-uniform!/raw)
         (only-in "raw/tensor.rkt" tr-tensor-copy-bytes/raw)
         (only-in "structs.rkt" tensor? wrap-tensor))

(provide check-generator-on!
         generator?
         generator-on?
         uniform-range?)

(struct generator (handle)
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (_g port _mode) (write-string "#<generator>" port)))

(define/contract-out seed/c flat-contract? ;; noqa
  (flat-named-contract 'seed (integer-in 0 (sub1 (expt 2 64)))))

(define/contract-out (make-generator seed #:device [device 'cpu]) ;; noqa
  (->* [seed/c] [#:device device/c] generator?)
  (define-values (type index) (device->type+index device))
  (generator
   (check-handle 'make-generator (tr-generator-new-on/raw seed type index))))

(define/contract-out (generator-device g) (-> generator? device?) ;; noqa
  (define-values (rc type index) (tr-generator-device/raw g))
  (check-ok rc 'generator-device)
  (type+index->device type index))

(define/contract-out cpu-generator/c flat-contract? ;; noqa
  (flat-named-contract
   'cpu-generator
   (lambda (g)
     (and (generator? g) (eq? 'cpu (device-type (generator-device g)))))))

(define/contract-out current-generator ;; noqa
  (parameter/c (or/c generator? #f))
  current-generator/raw)

(define (target-device target)
  (define-values (type index) (device->type+index (or target (default-device))))
  (type+index->device type index))

(define (generator-on? generator target)
  (define effective (or generator (current-generator/raw)))
  (or (not effective)
      (equal? (generator-device effective) (target-device target))))

(define (check-generator-on! who generator target
                             #:noun [noun "target"] #:label [label noun])
  (unless (generator-on? generator target)
    (raise-arguments-error
     who
     (string-append "contract violation;\n the generator lives on the " noun
                    " device")
     "generator device" (generator-device (or generator (current-generator/raw)))
     (string-append label " device") (target-device target))))

(define (host-bytes t n)
  (define out (make-bytes n))
  (define-values (rc _copied) (tr-tensor-copy-bytes/raw t n out))
  (and (zero? rc) out))

(define (byte-tensor-of? v n)
  (and (tensor? v)
       (= n (apply * (tensor-shape v)))
       (eq? 'uint8 (tensor-dtype v))
       (eq? 'cpu (device-type (tensor-device v)))))

(define (state-bytes v n)
  (and (byte-tensor-of? v n) (host-bytes v n)))

(define (field bs start end signed?)
  (integer-bytes->integer bs signed? #f start end))

(define (cpu-rng-state? v)
  (define bs (state-bytes v 5056))
  (and bs
       (<= 1 (field bs 8 12 #t) 624)
       (not (zero? (field bs 12 16 #t)))
       (<= (field bs 16 20 #f) 624)))

(define (cuda-rng-state? v)
  (define seed-only? (byte-tensor-of? v 8))
  (define bs (and (not seed-only?) (state-bytes v 16)))
  (cond
    [seed-only? #t]
    [bs (zero? (modulo (field bs 8 16 #f) 4))]
    [else #f]))

(define (mps-rng-state? v)
  (byte-tensor-of? v 44))

(begin-for-syntax
  (define-syntax-class state-of
    (pattern [type:id accepts?:id]
             #:with name (format-id #'type "~a-rng-state" #'type))))

(define-syntax-parse-rule (define-state-of/c lookup:id row:state-of ...)
  (define lookup
    (let ([contracts
           (hasheq (~@ 'row.type (flat-named-contract 'row.name row.accepts?))
                   ...)])
      (lambda (type) (hash-ref contracts type)))))

(define-state-of/c rng-state-of/c
  [cpu cpu-rng-state?]
  [cuda cuda-rng-state?]
  [mps mps-rng-state?])

(define (designated-type device)
  (define d (supplied-or device 'cpu))
  (cond
    [(device? d) (device-type d)]
    [(pair? d) 'cuda]
    [else d]))

(define (state-tensor who handle)
  (wrap-tensor (check-handle who handle)))

(define/contract-out (rng-state #:device [device 'cpu]) ;; noqa
  (->* () (#:device device/c) tensor?)
  (define-values (type index) (device->type+index device))
  (state-tensor 'rng-state (tr-default-generator-state/raw type index)))

(define/contract-out (set-rng-state! state #:device [device 'cpu]) ;; noqa
  (->i ([state (device) (rng-state-of/c (designated-type device))])
       (#:device [device device/c])
       [result void?])
  (define-values (type index) (device->type+index device))
  (check-ok (tr-default-generator-set-state!/raw type index state)
            'set-rng-state!)
  (void))

(define/contract-out (generator-state g) (-> generator? tensor?) ;; noqa
  (state-tensor 'generator-state (tr-generator-state/raw g)))

(define/contract-out (set-generator-state! g state) ;; noqa
  (->i ([g generator?]
        [state (g) (rng-state-of/c (device-type (generator-device g)))])
       [result void?])
  (check-ok (tr-generator-set-state!/raw g state) 'set-generator-state!)
  (void))

(define float-tensor/c
  (flat-named-contract
   'float-tensor
   (lambda (t)
     (and (tensor? t) (hash-has-key? float-dtype-max (tensor-dtype t))))))

(define (uniform-range? dtype low high)
  (define top (hash-ref float-dtype-max dtype #f))
  (define lo (exact->inexact low))
  (define hi (exact->inexact high))
  (and top (<= (- top) lo hi top) (<= (- hi lo) top)))

(define/contract-out (uniform! t [low 0.0] [high 1.0] ;; noqa
                               #:generator [generator #f])
  (->i ([t float-tensor/c])
       ([low real?]
        [high real?]
        #:generator [generator (or/c generator? #f)])
       #:pre/name (t generator) "the generator lives on the tensor's device"
       (generator-on? (supplied-or generator) (tensor-device t))
       #:pre/name (t low high)
       "low is at most high, both within the dtype's finite range and at most its largest value apart"
       (uniform-range? (tensor-dtype t) (supplied-or low 0.0) (supplied-or high 1.0))
       [result void?])
  (check-ok (tr-tensor-uniform!/raw t (exact->inexact low) (exact->inexact high)
                                    generator)
            'uniform!)
  (void))

(define/contract-out (normal! t [mean 0.0] [std 1.0] ;; noqa
                              #:generator [generator #f])
  (->i ([t float-tensor/c])
       ([mean real?]
        [std (>=/c 0)]
        #:generator [generator (or/c generator? #f)])
       #:pre/name (t generator) "the generator lives on the tensor's device"
       (generator-on? (supplied-or generator) (tensor-device t))
       [result void?])
  (check-ok (tr-tensor-normal!/raw t (exact->inexact mean) (exact->inexact std)
                                   generator)
            'normal!)
  (void))
