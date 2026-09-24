#lang racket/base

(require (only-in racket/contract/base -> ->* any cons/c listof)
         (only-in racket/file file->bytes)
         (only-in json jsexpr->string string->jsexpr)
         (only-in "../foreign.rkt"
                  bytes->tensor
                  tensor-dtype
                  tensor->bytes
                  tensor-shape
                  tensor?
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
  (bytes->tensor bs entry shape #:device 'cpu))

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

(define (field meta name key)
  (hash-ref meta key
            (lambda ()
              (raise-arguments-error 'load-state! "entry has no field"
                                     "entry" name
                                     "field" key))))

(define (mismatch-report strict? missing unexpected mismatched)
  (append
   (if (and strict? (pair? missing)) (list "missing keys" missing) '())
   (if (and strict? (pair? unexpected))
       (list "unexpected keys" unexpected)
       '())
   (if (pair? mismatched)
       (list "shape mismatches (key, file, model)" mismatched)
       '())))

(define/contract-out (load-state! model path #:strict? [strict? #t]) ;; noqa
  (->* [layer? path-string?] [#:strict? boolean?] any)
  (define raw (file->bytes path))
  (define header-len (integer-bytes->integer raw #f #f 0 8))
  (define data-start (+ 8 header-len))
  (define header
    (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (define entries (state-dict model))
  (define in-model
    (for/hash ([e (in-list entries)]) (values (car e) #t)))
  (define (meta-of name) (hash-ref header (string->symbol name) #f))
  (define missing
    (for/list ([e (in-list entries)] #:unless (meta-of (car e))) (car e)))
  (define unexpected
    (sort (for/list ([k (in-hash-keys header)]
                     #:unless (eq? k '__metadata__)
                     #:unless (hash-ref in-model (symbol->string k) #f))
            (symbol->string k))
          string<?))
  (define mismatched
    (for*/list ([e (in-list entries)]
                [meta (in-value (meta-of (car e)))]
                #:when meta
                [shape (in-value (field meta (car e) 'shape))]
                #:unless (equal? shape (tensor-shape (cdr e))))
      (list (car e) shape (tensor-shape (cdr e)))))
  (define report (mismatch-report strict? missing unexpected mismatched))
  (unless (null? report)
    (apply raise-arguments-error 'load-state!
           "checkpoint does not match the model" report))
  (with-no-grad
    (for ([e (in-list entries)])
      (define meta (meta-of (car e)))
      (when meta
        (define offsets (field meta (car e) 'data_offsets))
        (define loaded
          (decode (car e)
                  (field meta (car e) 'dtype)
                  (field meta (car e) 'shape)
                  (subbytes raw
                            (+ data-start (car offsets))
                            (+ data-start (cadr offsets)))))
        ;; copy_ converts, so a file in one dtype loads into a model in another
        (copy! (cdr e) loaded #f))))
  (unless strict?
    (values missing unexpected)))
