#lang racket/base

;; safetensors save/load for a model's parameters. The format is a small
;; binary: an 8-byte little-endian header length, a UTF-8 JSON header mapping
;; each (dotted) parameter name to {dtype, shape, data_offsets}, then the raw
;; little-endian f32 bytes laid out contiguously in named-parameters order.
;; That's exactly the layout Python's `safetensors` reads, so a file written
;; here loads there (and vice versa). load-state! copies into the existing
;; parameters in place via copy_ (under with-no-grad, like torch's
;; load_state_dict), so the model tree is preserved.

(require (only-in racket/file file->bytes)
         (only-in json jsexpr->string string->jsexpr)
         (only-in "module.rkt" named-parameters)
         (only-in "../foreign.rkt"
                  reshape
                  tensor
                  tensor->list
                  tensor-shape
                  with-no-grad)
         (only-in "../generated.rkt" copy!))

(provide state-dict
         save-state!
         load-state!)

;; The model's parameters as (dotted-name . tensor) pairs, in PyTorch order.
(define (state-dict model)
  (named-parameters model))

;; --- f32 <-> little-endian bytes ----------------------------------------
(define (floats->bytes floats)
  (apply bytes-append
         (for/list ([f (in-list floats)])
           (real->floating-point-bytes (exact->inexact f) 4 #f))))

(define (bytes->floats bs)
  (define n (quotient (bytes-length bs) 4))
  (for/list ([i (in-range n)])
    (floating-point-bytes->real bs #f (* i 4) (* (+ i 1) 4))))

;; --- save ---------------------------------------------------------------
(define (save-state! model path)
  (define-values (fields chunks total)
    (for/fold ([fields '()] [chunks '()] [offset 0])
              ([e (in-list (named-parameters model))])
      (define bs (floats->bytes (tensor->list (cdr e))))
      (define end (+ offset (bytes-length bs)))
      (values (cons (cons (string->symbol (car e))
                          (hasheq 'dtype "F32"
                                  'shape (tensor-shape (cdr e))
                                  'data_offsets (list offset end)))
                    fields)
              (cons bs chunks)
              end)))
  (define header-bytes
    (string->bytes/utf-8
     (jsexpr->string (make-immutable-hasheq (reverse fields)))))
  (call-with-output-file path #:exists 'replace
    (lambda (out)
      ;; 8-byte little-endian unsigned header length, header, then data.
      (write-bytes (integer->integer-bytes (bytes-length header-bytes) 8 #f #f)
                   out)
      (write-bytes header-bytes out)
      (for ([bs (in-list (reverse chunks))]) (write-bytes bs out)))))

;; --- load ---------------------------------------------------------------
(define (load-state! model path)
  (define raw (file->bytes path))
  (define header-len (integer-bytes->integer raw #f #f 0 8))
  (define data-start (+ 8 header-len))
  (define header
    (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (with-no-grad
    (for ([e (in-list (named-parameters model))])
      (define name (car e))
      (define param (cdr e))
      (define meta
        (hash-ref header (string->symbol name)
                  (lambda ()
                    (error 'load-state! "no entry for parameter ~s" name))))
      (define offsets (hash-ref meta 'data_offsets))
      (define floats
        (bytes->floats (subbytes raw
                                 (+ data-start (car offsets))
                                 (+ data-start (cadr offsets)))))
      (copy! param (apply reshape (tensor floats) (tensor-shape param)) #f))))
