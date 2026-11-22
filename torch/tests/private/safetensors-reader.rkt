#lang racket/base

(require (only-in json string->jsexpr)
         (only-in racket/file file->bytes)
         (only-in racket/match match-define)
         (only-in "../../main.rkt" bytes->tensor))

(provide read-safetensors
         safetensors-metadata)

(define dtypes
  (hash "F32" 'float32 "F64" 'float64 "F16" 'float16 "BF16" 'bfloat16 "I64" 'int64
        "U8" 'uint8 "BOOL" 'bool))

(define (header-of raw)
  (define data-start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
  (values data-start (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start))))

(define (read-safetensors path)
  (define raw (file->bytes path))
  (define-values (data-start header) (header-of raw))
  (for/hash ([(key entry) (in-hash header)] #:unless (eq? key '__metadata__))
    (match-define (list lo hi) (map (lambda (o) (+ data-start o)) (hash-ref entry 'data_offsets)))
    (values (symbol->string key)
            (bytes->tensor (subbytes raw lo hi)
                           (hash-ref dtypes (hash-ref entry 'dtype))
                           (hash-ref entry 'shape)))))

(define (safetensors-metadata path)
  (define-values (_start header) (header-of (file->bytes path)))
  (hash-ref header '__metadata__ (hasheq)))
