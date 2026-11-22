#lang racket/base

(require (only-in json string->jsexpr)
         (only-in racket/file file->bytes)
         (only-in "../../main.rkt" bytes->tensor))

(provide read-safetensors)

(define dtypes
  (hash "F32" 'float32 "F64" 'float64 "F16" 'float16 "BF16" 'bfloat16 "I64" 'int64
        "U8" 'uint8 "BOOL" 'bool))

(define (read-safetensors path)
  (define raw (file->bytes path))
  (define data-start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
  (define header (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (for/hash ([(key entry) (in-hash header)] #:unless (eq? key '__metadata__))
    (define offsets (hash-ref entry 'data_offsets))
    (values (symbol->string key)
            (bytes->tensor (subbytes raw (+ data-start (car offsets)) (+ data-start (cadr offsets)))
                           (hash-ref dtypes (hash-ref entry 'dtype))
                           (hash-ref entry 'shape)))))
