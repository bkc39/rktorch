#lang racket/base

(require (only-in ffi/unsafe prop:cpointer)
         (only-in racket/contract/base
                  -> ->* any/c flat-contract? flat-named-contract integer-in
                  or/c parameter/c)
         (only-in "../private/contract.rkt"
                  define/checked-out
                  define/contract-out)
         (only-in "device-type.rkt" device-type device/c device?)
         (only-in "error.rkt" check-handle check-ok)
         (only-in "ops.rkt"
                  default-device
                  device->type+index
                  type+index->device)
         (only-in "raw/random.rkt"
                  Generator?
                  current-generator/raw
                  tr-generator-device/raw
                  tr-generator-new-on/raw))

(provide generator-on?)

(struct generator-impl (handle)
  #:reflection-name 'generator
  #:property prop:cpointer 0
  #:property prop:custom-write
  (lambda (_g port _mode) (write-string "#<generator>" port)))

(define/contract-out seed/c flat-contract? ;; noqa
  (flat-named-contract 'seed (integer-in 0 (sub1 (expt 2 64)))))

(define/contract-out (make-generator seed #:device [device 'cpu]) ;; noqa
  (->* [seed/c] [#:device device/c] generator?)
  (define-values (type index) (device->type+index device))
  (generator-impl
   (check-handle 'make-generator (tr-generator-new-on/raw seed type index))))

(define/checked-out (generator? v) (-> any/c boolean?) ;; noqa
  (and (generator-impl? v) (Generator? (generator-impl-handle v))))

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
  (define-values (type index) (device->type+index (or target (default-device))))
  (or (not effective)
      (equal? (generator-device effective) (type+index->device type index))))
