#lang racket/base

(module+ test
  (require (only-in json string->jsexpr)
           (only-in racket/file file->bytes)
           (only-in rackunit check-equal? check-exn test-case)
           (only-in "../main.rkt" manual-seed! tensor->list)
           (only-in "../nn.rkt" Linear Sequential load-state! named-parameters
                    save-state!)
           (only-in "../private/safetensors.rkt" safetensors-subset)
           (only-in "../private/util.rkt" with-temporary-directory))

  (define (header raw)
    (define end (+ 8 (integer-bytes->integer raw #f #f 0 8)))
    (string->jsexpr (bytes->string/utf-8 raw #f 8 end)))

  (define (data-offsets raw key)
    (hash-ref (hash-ref (header raw) key) 'data_offsets))

  (manual-seed! 0)
  (define two (Sequential (Linear 3 4) (Linear 4 2)))

  (with-temporary-directory (dir)
    (define whole (build-path dir "two.safetensors"))
    (save-state! two whole)
    (define raw (file->bytes whole))
    (define data-start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
    (define first-end
      (max (cadr (data-offsets raw '|0.weight|))
           (cadr (data-offsets raw '|0.bias|))))
    (define prefix (subbytes raw 0 (+ data-start first-end)))

    (test-case "the entries with the prefix, from the start of the file"
      (define subset (safetensors-subset 'test prefix "0."))
      (check-equal? (sort (map symbol->string (hash-keys (header subset)))
                          string<?)
                    '("0.bias" "0.weight"))
      (check-equal? (modulo (integer-bytes->integer subset #f #f 0 8) 8) 0
                    "the header is padded to eight bytes")
      (define path (build-path dir "first.safetensors"))
      (call-with-output-file path (lambda (out) (write-bytes subset out)))
      (define one (Sequential (Linear 3 4)))
      (load-state! one path)
      (check-equal? (for/list ([p (in-list (named-parameters one))])
                      (cons (car p) (tensor->list (cdr p))))
                    (for/list ([p (in-list (named-parameters two))]
                               #:when (regexp-match? #rx"^0[.]" (car p)))
                      (cons (car p) (tensor->list (cdr p))))))

    (test-case "a prefix no entry has is an error"
      (check-exn #rx"no entry has the prefix"
                 (lambda () (safetensors-subset 'test prefix "2."))))

    (test-case "entries that run past the bytes fetched are an error"
      (check-exn #rx"run past the bytes fetched.*bytes needed"
                 (lambda () (safetensors-subset 'test prefix "1."))))))
