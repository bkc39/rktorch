#lang racket/base

(require (only-in json jsexpr->string string->jsexpr)
         (only-in racket/string string-prefix?))

(provide safetensors-subset)

;; `raw` is the start of a safetensors file: its header and at least the
;; data of the entries named with `prefix`. The result is a whole file of
;; just those entries, their bytes at the offsets the header already gives.
(define (safetensors-subset who raw prefix)
  (define data-start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
  (define header
    (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (define kept
    (for/hasheq ([(key entry) (in-hash header)]
                 #:when (string-prefix? (symbol->string key) prefix))
      (values key entry)))
  (when (zero? (hash-count kept))
    (raise-arguments-error who "no entry has the prefix" "prefix" prefix))
  (define end
    (for/fold ([end 0]) ([entry (in-hash-values kept)])
      (max end (cadr (hash-ref entry 'data_offsets)))))
  (define available (- (bytes-length raw) data-start))
  (unless (<= end available)
    (raise-arguments-error who "the entries run past the bytes fetched"
                           "prefix" prefix
                           "bytes needed" end
                           "bytes fetched" available))
  (define text (string->bytes/utf-8 (jsexpr->string kept)))
  (define padded (bytes-append text (make-bytes (modulo (- (bytes-length text)) 8)
                                                (char->integer #\space))))
  (bytes-append (integer->integer-bytes (bytes-length padded) 8 #f #f)
                padded
                (subbytes raw data-start (+ data-start end))))
