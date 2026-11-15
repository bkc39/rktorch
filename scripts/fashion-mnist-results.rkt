#lang racket/base

(require (only-in racket/contract/base -> list/c listof)
         (only-in torch device?)
         (only-in "../torch/private/contract.rkt" define/contract-out))

(define row/c (list/c exact-positive-integer? real? real? real? real?))

(define/contract-out (fashion-mnist-record racket-rows lightning-rows device
                                           #:commit commit #:seconds seconds)
  (-> (listof row/c) (listof row/c) device? #:commit string? #:seconds (list/c real? real?) hash?)
  (error 'fashion-mnist-record "unimplemented"))
