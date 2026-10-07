#lang racket/base

(provide allocator-reading
         install-device-queries!
         queried-capacity
         release-query)

(struct queries (capacity allocated release) #:mutable)

(define the-queries (queries #f #f #f))

;; raw/device.rkt owns the device bindings and requires the ledger, so it
;; hands these over at instantiation instead of being required from here.
;; Each takes a device and answers in bytes, or #f where it cannot say;
;; `release` empties the device's allocator cache and answers whether it had
;; one to empty.
(define (install-device-queries! #:capacity capacity
                                 #:allocated allocated
                                 #:release [release #f])
  (set-queries-capacity! the-queries capacity)
  (set-queries-allocated! the-queries allocated)
  (set-queries-release! the-queries release))

(define (queried-capacity dev)
  (define query (queries-capacity the-queries))
  (define total (and query (query dev)))
  (and total (positive? total) total))

(define (release-query)
  (queries-release the-queries))

;; The allocator's own allocated bytes: the ledger double-counts views and
;; cannot see storage only the autograd graph holds. #f when unknown.
(define allocator-reading
  (make-parameter
   (lambda (dev)
     (define query (queries-allocated the-queries))
     (and query (query dev)))))
