#lang racket/base

(module+ test
  (require (only-in ffi/vector f64vector)
           (only-in json jsexpr->string string->jsexpr)
           (only-in racket/file file->bytes make-temporary-file)
           (only-in rackunit check-equal? check-exn check-not-exn test-case)
           (only-in "../main.rkt" manual-seed! tensor tensor->list tensor-dtype
                    tensor-shape zeros)
           (only-in "../nn.rkt" Buffer Linear buffers define-layer load-state!
                    named-parameters parameters save-state!))

  (define (values-of layer)
    (map tensor->list (parameters layer)))

  (define-layer Pair (a b)
    #:init (out)
    (set! a (Linear 2 out))
    (set! b (Linear 2 2))
    #:forward (x) x)

  (define-layer Other (a c)
    #:init (out)
    (set! a (Linear 2 out))
    (set! c (Linear 2 2))
    #:forward (x) x)

  (define (saved-pair out)
    (manual-seed! 0)
    (define model (Pair out))
    (define path (make-temporary-file "rkt-state-~a.safetensors"))
    (save-state! model path)
    (values model path))

  (test-case "tensor builds float64 exactly, from a list or an f64vector"
    (define wide (+ 1.0 (expt 2.0 -52)))
    (define t (tensor (list 0.1 wide) #:dtype 'float64))
    (check-equal? (tensor-dtype t) 'float64)
    (check-equal? (tensor->list t) (list 0.1 wide))
    (define v (tensor (f64vector 0.1 wide)))
    (check-equal? (tensor-dtype v) 'float64 "an f64vector is float64 data")
    (check-equal? (tensor->list v) (list 0.1 wide))
    (check-equal? (tensor-dtype (tensor (f64vector 1.5) #:dtype 'float32))
                  'float32)
    (check-equal? (tensor-shape (tensor '((1 2) (3 4)) #:dtype 'float64))
                  '(2 2)))

  (define (rewrite-header! path update)
    (define raw (file->bytes path))
    (define len (integer-bytes->integer raw #f #f 0 8))
    (define header
      (string->jsexpr (bytes->string/utf-8 raw #f 8 (+ 8 len))))
    (define rewritten (string->bytes/utf-8 (jsexpr->string (update header))))
    (call-with-output-file path #:exists 'truncate
      (lambda (out)
        (write-bytes (integer->integer-bytes (bytes-length rewritten) 8 #f #f)
                     out)
        (write-bytes rewritten out)
        (write-bytes raw out (+ 8 len)))))

  (test-case "a tag or payload the file cannot back fails before any copy"
    (for ([damage (in-list (list (lambda (m) (hash-set m 'dtype "I32"))
                                 (lambda (m) (hash-set m 'data_offsets '(0 4)))
                                 (lambda (m)
                                   (hash-set m 'data_offsets '(0 1000)))
                                 (lambda (m) (hash-set m 'data_offsets '(0)))))]
          [expected
           (in-list
            (list #rx"unsupported dtypes \\(key, dtype\\): '\\(\\(\"b.weight\" \"I32\"\\)\\)"
                  #rx"payloads out of place.*\"b.weight\" \\(0 4\\) 16"
                  #rx"payloads out of place.*\"b.weight\" \\(0 1000\\) 16"
                  #rx"payloads out of place.*\"b.weight\" \\(0\\) 16"))])
      (define-values (_model path) (saved-pair 2))
      (rewrite-header! path (lambda (h) (hash-update h 'b.weight damage)))
      (define target (Pair 2))
      (define before (values-of target))
      (check-exn expected (lambda () (load-state! target path)))
      (check-equal? (values-of target) before
                    "a.weight and a.bias precede the damage and stay put")
      (delete-file path)))

  (test-case "strict loading names every missing and unexpected key at once"
    (define-values (_model path) (saved-pair 2))
    (define target (Other 2))
    (define before (values-of target))
    (check-exn
     (lambda (e)
       (define m (exn-message e))
       (and (regexp-match? #rx"^load-state!: checkpoint does not match" m)
            (regexp-match? #rx"missing keys: '\\(\"c.weight\" \"c.bias\"\\)" m)
            (regexp-match? #rx"unexpected keys: '\\(\"b.bias\" \"b.weight\"\\)"
                           m)))
     (lambda () (load-state! target path)))
    (check-equal? (values-of target) before
                  "nothing is copied when the check fails")
    (delete-file path))

  (test-case "non-strict loading takes the intersection and returns both lists"
    (define-values (model path) (saved-pair 2))
    (define target (Other 2))
    (define c-before (map tensor->list (list-tail (parameters target) 2)))
    (define-values (missing unexpected)
      (load-state! target path #:strict? #f))
    (check-equal? missing '("c.weight" "c.bias"))
    (check-equal? unexpected '("b.bias" "b.weight"))
    (check-equal? (map tensor->list (list (car (parameters target))
                                          (cadr (parameters target))))
                  (map tensor->list (list (car (parameters model))
                                          (cadr (parameters model)))))
    (check-equal? (map tensor->list (list-tail (parameters target) 2))
                  c-before "keys the file lacks keep their values")
    (check-equal? (map car (named-parameters target))
                  '("a.weight" "a.bias" "c.weight" "c.bias"))
    (delete-file path))

  (define (b->c key) (regexp-replace #rx"^b[.]" key "c."))

  (test-case "#:rename maps each file key to the model key it loads into"
    (define-values (model path) (saved-pair 2))
    (define target (Other 2))
    (check-exn #rx"missing keys" (lambda () (load-state! target path)))
    (load-state! target path #:rename b->c)
    (check-equal? (values-of target) (values-of model))
    (delete-file path))

  (test-case "#:rename to #f leaves a key out; the rest report in their own names"
    (define-values (_model path) (saved-pair 2))
    (define-values (missing unexpected)
      (load-state! (Other 2) path #:strict? #f
                   #:rename (lambda (k) (and (regexp-match? #rx"^a[.]" k) k))))
    (check-equal? missing '("c.weight" "c.bias"))
    (check-equal? unexpected '() "a dropped key is not unexpected")
    (define-values (missing* unexpected*)
      (load-state! (Other 2) path #:strict? #f
                   #:rename (lambda (k) (regexp-replace #rx"^b[.]" k "z."))))
    (check-equal? missing* '("c.weight" "c.bias") "missing: the model's names")
    (check-equal? unexpected* '("b.bias" "b.weight")
                  "unexpected: the file's names")
    (define narrow (Other 3))
    (check-exn #rx"shape mismatches.*\"a.weight\" \\(2 2\\) \\(3 2\\)"
               (lambda () (load-state! narrow path #:rename b->c))
               "a mismatch names the model's key")
    (delete-file path))

  (test-case "two file keys renamed to one model key fail before any copy"
    (define-values (_model path) (saved-pair 2))
    (define target (Pair 2))
    (define before (values-of target))
    (check-exn
     #rx"#:rename maps two file keys to one model key.*a.weight.*a.bias"
     (lambda ()
       (load-state! target path
                    #:rename (lambda (k) (regexp-replace #rx"bias$" k
                                                         "weight")))))
    (check-equal? (values-of target) before)
    (delete-file path))

  (test-case "a shape mismatch is an error in either mode, per key"
    (define-values (_model path) (saved-pair 3))
    (define narrow (Pair 2))
    (for ([strict? (in-list '(#t #f))])
      (check-exn
       #rx"shape mismatches.*\"a.weight\" \\(3 2\\) \\(2 2\\).*\"a.bias\" \\(3\\) \\(2\\)"
       (lambda () (load-state! narrow path #:strict? strict?))))
    (delete-file path)
    (manual-seed! 0)
    (define wide-first (Linear 2 3))
    (define square-path (make-temporary-file "rkt-state-~a.safetensors"))
    (save-state! wide-first square-path)
    (check-exn #rx"shape mismatches.*\"weight\" \\(3 2\\) \\(2 3\\)"
               (lambda () (load-state! (Linear 3 2) square-path))
               "equal element counts do not make shapes equal")
    (delete-file square-path))

  (test-case "every dtype a buffer can hold round-trips"
    (define-layer Typed (raw wide tally)
      #:init (seed)
      (set! raw (Buffer (tensor (if seed #"\0\1\377" #"\0\0\0"))))
      (set! wide (Buffer (tensor (list (if seed 0.1 0.0)) #:dtype 'float64)))
      (set! tally (Buffer (tensor (list (if seed 9007199254740993 0)))))
      #:forward (x) x)
    (define path (make-temporary-file "rkt-state-~a.safetensors"))
    (save-state! (Typed #t) path)
    (define loaded (Typed #f))
    (check-not-exn (lambda () (load-state! loaded path)))
    (check-equal? (map tensor-dtype (buffers loaded)) '(uint8 float64 int64))
    (check-equal? (map tensor->list (buffers loaded))
                  '((0 1 255) (0.1) (9007199254740993)))
    (delete-file path)
    (define empty-path (make-temporary-file "rkt-state-~a.safetensors"))
    (define-layer Hollow (none)
      #:init ()
      (set! none (Buffer (zeros 0)))
      #:forward (x) x)
    (save-state! (Hollow) empty-path)
    (check-not-exn (lambda () (load-state! (Hollow) empty-path)))
    (delete-file empty-path)))
