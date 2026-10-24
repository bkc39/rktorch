#lang racket/base

(module+ test
  (require (only-in json bytes->jsexpr)
           (only-in racket/file file->bytes make-temporary-file)
           (only-in racket/string string-replace)
           (only-in rackunit check-equal?)
           (only-in "../main.rkt" dtype item randn)
           (only-in "../nn.rkt"
                    BatchNorm2d buffers in-eval-mode load-state! save-state!
                    state-dict)
           (only-in "private/python-env.rkt"
                    python-check python-torch-available?)
           (only-in "private/within.rkt" bounded-test))

  (define (ours key)
    (string-replace key "_" "-"))

  (define (saved-entry path key)
    (define raw (file->bytes path))
    (define header-len (integer-bytes->integer raw #f #f 0 8))
    (define header (bytes->jsexpr (subbytes raw 8 (+ 8 header-len))))
    (define meta (hash-ref header (string->symbol key)))
    (define offsets (hash-ref meta 'data_offsets))
    (values (hash-ref meta 'dtype)
            (hash-ref meta 'shape)
            (bytes->list (subbytes raw
                                   (+ 8 header-len (car offsets))
                                   (+ 8 header-len (cadr offsets))))))

  (cond
    [(not (python-torch-available?))
     (displayln "[batch-norm-parity-test] skipped: no python3 `torch`")]
    [else
     (bounded-test "num_batches_tracked round-trips torch's state past 2^24"
       (define j (python-check "batch_norm_counter.py"))
       (define in (make-temporary-file "rkt-bn-torch-~a.safetensors"))
       (define out (make-temporary-file "rkt-bn-ours-~a.safetensors"))
       (call-with-output-file in #:exists 'truncate
         (lambda (port) (write-bytes (list->bytes (hash-ref j 'file)) port)))
       (define bn (BatchNorm2d 3))
       (load-state! bn in #:rename ours)
       (check-equal? (map car (state-dict bn)) (map ours (hash-ref j 'keys)))
       (check-equal? (for/list ([e (in-list (state-dict bn))])
                       (format "torch.~a" (dtype (cdr e))))
                     (hash-ref j 'dtypes))
       (define counter (caddr (buffers bn)))
       (check-equal? (item counter) (hash-ref j 'saved_counter))
       (define x (randn 2 3 4 4))
       (bn x)
       (in-eval-mode bn (bn x))
       (check-equal? (item counter) (hash-ref j 'counter))
       (check-equal? (format "torch.~a" (dtype counter))
                     (hash-ref j 'counter_dtype))
       (save-state! bn out)
       (define-values (tag shape payload)
         (saved-entry out "num-batches-tracked"))
       (check-equal? (list tag shape) '("I64" ()))
       (check-equal? payload (hash-ref j 'counter_bytes))
       (delete-file in)
       (delete-file out))]))
