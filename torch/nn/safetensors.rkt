#lang racket/base

(require (only-in json jsexpr->string string->jsexpr)
         (only-in racket/file file->bytes make-temporary-file)
         (only-in racket/match match match-define)
         (only-in racket/path normalize-path)
         (only-in "../foreign.rkt" bytes->tensor dtype tensor->bytes tensor-shape))

(provide (struct-out safetensors)
         call-with-atomic-output-file
         read-safetensors
         safetensors-payload-size
         safetensors-tensor
         tag-dtype
         tag-width
         write-safetensors!)

;; tensor->bytes is host order, little-endian on every platform the shim builds for
(define tags
  '((float32 "F32" 4) (float64 "F64" 8) (float16 "F16" 2) (bfloat16 "BF16" 2)
    (int64 "I64" 8) (bool "BOOL" 1) (uint8 "U8" 1)))

(define (tag-entry tag)
  (for/first ([e (in-list tags)] #:when (match e [(list _ t _) (equal? t tag)])) e))

(define (tag-dtype tag)
  (match (tag-entry tag)
    [(list dt _ _) dt]
    [#f #f]))

(define (tag-width tag)
  (match-define (list _ _ width) (tag-entry tag))
  width)

(define (encode who name t)
  (match (assq (dtype t) tags)
    [(list _ tag _) (values tag (tensor->bytes t))]
    [#f (raise-arguments-error who "unsupported dtype" "entry" name "dtype" (dtype t))]))

(define (delete-if-there path)
  (when (file-exists? path)
    (delete-file path)))

(define (call-with-atomic-output-file path write!)
  (define target (normalize-path path))
  (define-values (dir name _) (split-path target))
  (define temporary
    (make-temporary-file (string-append "." (path->string name) ".~a.tmp") #:base-dir dir))
  (with-handlers ([(lambda (_) #t)
                   (lambda (e)
                     (delete-if-there temporary)
                     (raise e))])
    (call-with-output-file temporary #:exists 'truncate
      (lambda (out)
        (write! out)
        (flush-output out)))
    (rename-file-or-directory temporary target #t)))

(define (write-safetensors! who path entries #:metadata [metadata #f])
  (define-values (fields chunks)
    (for/fold ([fields '()] [chunks '()] [offset 0] #:result (values fields chunks))
              ([e (in-list entries)])
      (match-define (cons name t) e)
      (define-values (tag bs) (encode who name t))
      (define end (+ offset (bytes-length bs)))
      (values (cons (cons (string->symbol name)
                          (hasheq 'dtype tag
                                  'shape (tensor-shape t)
                                  'data_offsets (list offset end)))
                    fields)
              (cons bs chunks)
              end)))
  (define header
    (make-immutable-hasheq
     (if metadata
         (cons (cons '__metadata__ (for/hasheq ([(k v) (in-hash metadata)])
                                     (values (string->symbol k) v)))
               (reverse fields))
         (reverse fields))))
  (define header-bytes (string->bytes/utf-8 (jsexpr->string header)))
  (call-with-atomic-output-file
   path
   (lambda (out)
     (write-bytes (integer->integer-bytes (bytes-length header-bytes) 8 #f #f) out)
     (write-bytes header-bytes out)
     (for ([bs (in-list (reverse chunks))])
       (write-bytes bs out)))))

(struct safetensors (header metadata raw data-start))

(define (read-safetensors path)
  (define raw (file->bytes path))
  (define data-start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
  (define all (string->jsexpr (bytes->string/utf-8 raw #f 8 data-start)))
  (safetensors (hash-remove all '__metadata__)
               (match (hash-ref all '__metadata__ #f)
                 [(? hash? m) (for/hash ([(k v) (in-hash m)]) (values (symbol->string k) v))]
                 [_ #f])
               raw
               data-start))

(define (safetensors-payload-size f)
  (- (bytes-length (safetensors-raw f)) (safetensors-data-start f)))

;; the payload decodes on the host: an F64 entry bound for a float32 model
;; could not land on an MPS default device
(define (safetensors-tensor f tag shape start end)
  (define base (safetensors-data-start f))
  (bytes->tensor (subbytes (safetensors-raw f) (+ base start) (+ base end))
                 (tag-dtype tag)
                 shape
                 #:device 'cpu))
