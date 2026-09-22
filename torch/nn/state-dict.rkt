#lang racket/base

(require (only-in racket/contract/base -> cons/c listof)
         (only-in racket/file file->bytes)
         (only-in json jsexpr->string string->jsexpr)
         (only-in "../foreign.rkt"
                  bytes->tensor
                  tensor-dtype
                  tensor->bytes
                  tensor-shape
                  tensor?
                  with-default-device
                  with-no-grad)
         (only-in "../generated.rkt" copy!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt"
                  layer-named-buffers layer-named-parameters layer?))

(define/contract-out (state-dict model) ;; noqa
  (-> layer? (listof (cons/c string? tensor?)))
  (append (layer-named-parameters model "") (layer-named-buffers model "")))

;; safetensors' dtype tags; the payload is the element bytes little-endian,
;; which is the host's order on every platform the shim builds for
(define tags
  '((float32 . "F32") (float64 . "F64") (float16 . "F16") (bfloat16 . "BF16")
    (int64 . "I64") (bool . "BOOL") (uint8 . "U8")))

(define (encode name t)
  (define tag (assq (tensor-dtype t) tags))
  (unless tag
    (raise-arguments-error 'save-state! "unsupported dtype"
                           "entry" name
                           "dtype" (tensor-dtype t)))
  (values (cdr tag) (tensor->bytes t)))

(define (decode name dtype shape bs)
  (define entry
    (for/first ([e (in-list tags)] #:when (string=? (cdr e) dtype))
      (car e)))
  (unless entry
    (raise-arguments-error 'load-state! "unsupported dtype"
                           "entry" name
                           "dtype" dtype))
  ;; the payload decodes on the host and `copy!` moves it: an F64 entry
  ;; loading into a float32 model could not land on an MPS default device
  (with-default-device 'cpu (bytes->tensor bs entry shape)))

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
      (define (field key)
        (hash-ref meta key
                  (lambda ()
                    (raise-arguments-error 'load-state! "entry has no field"
                                           "entry" name
                                           "field" key))))
      (define offsets (field 'data_offsets))
      (define shape (field 'shape))
      (unless (equal? shape (tensor-shape target))
        (raise-arguments-error 'load-state! "shape mismatch"
                               "entry" name
                               "file" shape
                               "model" (tensor-shape target)))
      (define loaded
        (decode name (field 'dtype) shape
                (subbytes raw
                          (+ data-start (car offsets))
                          (+ data-start (cadr offsets)))))
      ;; copy_ converts, so a file in one dtype loads into a model in another
      (copy! target loaded #f))))
