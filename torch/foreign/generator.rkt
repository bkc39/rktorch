#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre expr id))
         (only-in ffi/unsafe prop:cpointer)
         (only-in racket/contract/base
                  -> ->* ->i and/c flat-contract? flat-named-contract
                  integer-in or/c parameter/c)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "device-type.rkt" device-type device/c device?)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "ops.rkt"
                  default-device
                  device->type+index
                  tensor-device
                  type+index->device)
         (only-in "raw/random.rkt"
                  current-generator/raw
                  tr-generator-device/raw
                  tr-generator-new-on/raw
                  tr-tensor-normal!/raw
                  tr-tensor-uniform!/raw)
         (only-in "structs.rkt" tensor?))

(provide generator?
         generator-on?)

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

(define (generator-on? generator target)
  (define effective (or generator (current-generator/raw)))
  (cond
    [(not effective) #t]
    [else
     (define-values (type index)
       (device->type+index (or target (default-device))))
     (equal? (generator-device effective)
             (type+index->device type index))]))

(define-syntax-parse-rule (define-fill! name:id raw:id first:id second:id
                                        second/c:expr)
  (define/contract-out (name t first second) ;; noqa
    (->i ([t tensor?] [first real?] [second second/c])
         #:pre/name (t) "the generator lives on the tensor's device"
         (generator-on? #f (tensor-device t))
         [result void?])
    (check-ok (raw t (exact->inexact first) (exact->inexact second) #f) 'name)
    (void)))

(define-fill! uniform! tr-tensor-uniform!/raw low high real?)
(define-fill! normal! tr-tensor-normal!/raw mean std (and/c real? positive?))
