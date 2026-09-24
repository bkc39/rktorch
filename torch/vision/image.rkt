#lang racket/base

(require (only-in racket/contract/base ->* flat-named-contract or/c)
         (only-in racket/file file->bytes)
         (only-in "../foreign.rkt" default-device device/c tensor? to-device)
         (only-in "../foreign/error.rkt" check-handle)
         (only-in "../foreign/raw/image.rkt" tr-image-decode/raw)
         (only-in "../foreign/structs.rkt" wrap-tensor)
         (only-in "../private/contract.rkt" define/contract-out))

(define image-modes
  '((unchanged . 0) (gray . 1) (gray-alpha . 2) (rgb . 3) (rgba . 4)))

(define image-mode/c
  (flat-named-contract 'image-mode
                       (lambda (m) (and (assq m image-modes) #t))))

(define encoded-image/c
  (flat-named-contract 'non-empty-bytes
                       (lambda (bs)
                         (and (bytes? bs) (positive? (bytes-length bs))))))

(define (decode who bs mode device)
  (define decoded
    (wrap-tensor
     (check-handle who
                   (tr-image-decode/raw bs (bytes-length bs)
                                        (cdr (assq mode image-modes))))))
  (to-device decoded (or device (default-device))))

(define/contract-out (decode-image bs ;; noqa
                                   #:mode [mode 'unchanged]
                                   #:device [device #f])
  (->* [encoded-image/c]
       [#:mode image-mode/c #:device (or/c #f device/c)]
       tensor?)
  (decode 'decode-image bs mode device))

(define/contract-out (read-image path ;; noqa
                                 #:mode [mode 'unchanged]
                                 #:device [device #f])
  (->* [path-string?]
       [#:mode image-mode/c #:device (or/c #f device/c)]
       tensor?)
  (decode 'read-image (file->bytes path) mode device))
