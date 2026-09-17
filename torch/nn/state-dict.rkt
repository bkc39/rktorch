#lang racket/base

(require (only-in ffi/vector
                  f32vector-length
                  f32vector-ref
                  f32vector-set!
                  f64vector-length
                  f64vector-ref
                  f64vector-set!
                  make-f32vector
                  make-f64vector
                  make-s64vector
                  s64vector-length
                  s64vector-ref
                  s64vector-set!)
         (only-in racket/contract/base -> ->* any cons/c listof)
         (only-in racket/file file->bytes)
         (only-in json jsexpr->string string->jsexpr)
         (only-in "../foreign.rkt"
                  reshape
                  tensor
                  tensor-dtype
                  tensor->vector
                  tensor-shape
                  tensor?
                  to-dtype
                  with-no-grad)
         (only-in "../generated.rkt" copy!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt"
                  layer-named-buffers layer-named-parameters layer?))

(define/contract-out (state-dict model) ;; noqa
  (-> layer? (listof (cons/c string? tensor?)))
  (append (layer-named-parameters model "") (layer-named-buffers model "")))

;; Little-endian, as safetensors stores every dtype.
(define (pack n width write-at!)
  (define bs (make-bytes (* n width)))
  (for ([i (in-range n)])
    (write-at! bs i (* i width)))
  bs)

(define (encode name t)
  (case (tensor-dtype t)
    [(float32)
     (define v (tensor->vector t))
     (values "F32"
             (pack (f32vector-length v) 4
                   (lambda (bs i at)
                     (real->floating-point-bytes (f32vector-ref v i) 4 #f
                                                 bs at))))]
    [(float64)
     (define v (tensor->vector t))
     (values "F64"
             (pack (f64vector-length v) 8
                   (lambda (bs i at)
                     (real->floating-point-bytes (f64vector-ref v i) 8 #f
                                                 bs at))))]
    [(int64)
     (define v (tensor->vector t))
     (values "I64"
             (pack (s64vector-length v) 8
                   (lambda (bs i at)
                     (integer->integer-bytes (s64vector-ref v i) 8 #t #f
                                             bs at))))]
    [(uint8) (values "U8" (tensor->vector t))]
    [(bool) (values "BOOL" (tensor->vector (to-dtype t 'uint8)))]
    [else
     (raise-arguments-error 'save-state! "unsupported dtype"
                            "entry" name
                            "dtype" (tensor-dtype t))]))

(define (unpack bs width make set-at!)
  (define n (quotient (bytes-length bs) width))
  (define v (make n))
  (for ([i (in-range n)])
    (set-at! v i (* i width)))
  v)

(define (decode name dtype bs)
  (case dtype
    [("F32")
     (tensor
      (unpack bs 4 make-f32vector
              (lambda (v i at)
                (f32vector-set! v i
                                (floating-point-bytes->real bs #f
                                                            at (+ at 4))))))]
    [("F64")
     (tensor
      (unpack bs 8 make-f64vector
              (lambda (v i at)
                (f64vector-set! v i
                                (floating-point-bytes->real bs #f
                                                            at (+ at 8))))))]
    [("I64")
     (tensor
      (unpack bs 8 make-s64vector
              (lambda (v i at)
                (s64vector-set! v i
                                (integer-bytes->integer bs #t #f
                                                        at (+ at 8))))))]
    [("U8") (tensor bs)]
    [("BOOL") (to-dtype (tensor bs) 'bool)]
    [else
     (raise-arguments-error 'load-state! "unsupported dtype"
                            "entry" name
                            "dtype" dtype)]))

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
                #:unless (equal? (hash-ref meta 'shape)
                                 (tensor-shape (cdr e))))
      (list (car e) (hash-ref meta 'shape) (tensor-shape (cdr e)))))
  (define report (mismatch-report strict? missing unexpected mismatched))
  (unless (null? report)
    (apply raise-arguments-error 'load-state!
           "checkpoint does not match the model" report))
  (with-no-grad
    (for ([e (in-list entries)])
      (define meta (meta-of (car e)))
      (when meta
        (define offsets (hash-ref meta 'data_offsets))
        (define loaded
          (decode (car e)
                  (hash-ref meta 'dtype)
                  (subbytes raw
                            (+ data-start (car offsets))
                            (+ data-start (cadr offsets)))))
        (copy! (cdr e) (apply reshape loaded (tensor-shape (cdr e))) #f))))
  (unless strict?
    (values missing unexpected)))
