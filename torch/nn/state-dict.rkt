#lang racket/base

(require (only-in racket/contract/base -> cons/c listof)
         (only-in racket/file file->bytes)
         (only-in json jsexpr->string string->jsexpr)
         (only-in "../foreign.rkt"
                  reshape
                  tensor
                  tensor-dtype
                  tensor->list
                  tensor-shape
                  tensor?
                  to-dtype
                  with-no-grad)
         (only-in "../generated.rkt" copy!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "module.rkt" layer? named-buffers named-parameters))

(define/contract-out (state-dict model) ;; noqa
  (-> layer? (listof (cons/c string? tensor?)))
  (append (named-parameters model) (named-buffers model)))

(define (encode name t)
  (define vals (tensor->list t))
  (case (tensor-dtype t)
    [(float32)
     (values "F32"
             (apply bytes-append
                    (for/list ([f (in-list vals)])
                      (real->floating-point-bytes (exact->inexact f) 4 #f))))]
    [(int64)
     (values "I64"
             (apply bytes-append
                    (for/list ([i (in-list vals)])
                      (integer->integer-bytes i 8 #t #f))))]
    [(bool)
     (values "BOOL"
             (apply bytes (for/list ([v (in-list vals)]) (if (zero? v) 0 1))))]
    [else
     (raise-arguments-error 'save-state! "unsupported dtype"
                            "entry" name
                            "dtype" (tensor-dtype t))]))

(define (decode dtype bs)
  (define n (bytes-length bs))
  (case dtype
    [("F32")
     (tensor (for/list ([i (in-range 0 n 4)])
               (floating-point-bytes->real bs #f i (+ i 4))))]
    [("I64")
     (tensor (for/list ([i (in-range 0 n 8)])
               (integer-bytes->integer bs #t #f i (+ i 8))))]
    [("BOOL")
     (to-dtype (tensor (for/list ([b (in-bytes bs)]) b)) 'bool)]
    [else
     (raise-arguments-error 'load-state! "unsupported dtype" "dtype" dtype)]))

(define/contract-out (save-state! model path) ;; noqa
  (-> layer? path-string? void?)
  (define-values (fields chunks total)
    (for/fold ([fields '()] [chunks '()] [offset 0])
              ([e (in-list (state-dict model))])
      (define-values (dtype bs) (encode (car e) (cdr e)))
      (define end (+ offset (bytes-length bs)))
      (values (cons (cons (string->symbol (car e))
                          (hasheq 'dtype dtype
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
      (write-bytes (integer->integer-bytes (bytes-length header-bytes) 8 #f #f)
                   out)
      (write-bytes header-bytes out)
      (for ([bs (in-list (reverse chunks))]) (write-bytes bs out)))))

(define/contract-out (load-state! model path) ;; noqa
  (-> layer? path-string? void?)
  (define raw (file->bytes path))
  (define header-len (integer-bytes->integer raw #f #f 0 8))
  (define data-start (+ 8 header-len))
  (define header
    (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (with-no-grad
    (for ([e (in-list (state-dict model))])
      (define name (car e))
      (define target (cdr e))
      (define meta
        (hash-ref header (string->symbol name)
                  (lambda ()
                    (error 'load-state! "no entry for ~s" name))))
      (define offsets (hash-ref meta 'data_offsets))
      (define loaded
        (decode (hash-ref meta 'dtype)
                (subbytes raw
                          (+ data-start (car offsets))
                          (+ data-start (cadr offsets)))))
      (copy! target (apply reshape loaded (tensor-shape target)) #f))))
