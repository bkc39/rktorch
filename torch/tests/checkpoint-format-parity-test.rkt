#lang racket/base

(module+ test
  (require (only-in json read-json)
           (only-in racket/list remove-duplicates)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/string string-prefix?)
           (only-in racket/system system*)
           rackunit
           (only-in "../main.rkt" ones tensor? zeros)
           (only-in "../nn.rkt" state-dict)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run Small stored)
           (only-in "private/comparable.rkt" host-bytes)
           (only-in "private/python-env.rkt" python with-python-env))

  (define-runtime-path script "python/read_checkpoint.py")

  (define tags
    (hash "F32" 'float32 "F64" 'float64 "F16" 'float16 "BF16" 'bfloat16 "I64" 'int64
          "I32" 'int32 "I16" 'int16 "I8" 'int8 "U8" 'uint8 "BOOL" 'bool))

  (define (hex->bytes s)
    (apply bytes (for/list ([i (in-range 0 (string-length s) 2)])
                   (string->number (substring s i (+ i 2)) 16))))

  (define (entry-form e)
    (list (hash-ref tags (hash-ref e 'dtype)) (hash-ref e 'shape) (hex->bytes (hash-ref e 'hex))))

  (define (tensors-in v)
    (cond
      [(tensor? v) (list (host-bytes v))]
      [(hash? v) (for*/list ([(k x) (in-hash v)] [t (in-list (append (tensors-in k) (tensors-in x)))])
                   t)]
      [(pair? v) (append (tensors-in (car v)) (tensors-in (cdr v)))]
      [else '()]))

  (define (sorted forms) (sort forms string<? #:key (lambda (f) (format "~s" f))))

  (define (python-read path out)
    (define log (open-output-string))
    (define ok?
      (parameterize ([current-output-port log] [current-error-port log])
        (with-python-env (system* python "-I" script (path->string path) (path->string out)))))
    (unless ok? (error 'read_checkpoint.py "failed:\n~a" (get-output-string log)))
    (call-with-input-file out read-json))

  (define probe-value
    (list (ones 2 3) (zeros 0) (ones 2 #:dtype 'bfloat16) (ones 1 #:dtype 'bool)
          (ones 3 #:dtype 'int64) (ones 2 #:dtype 'uint8) (ones 2 #:dtype 'float64)
          +inf.0 +nan.0 'sym '(1 . 2) (hasheq 'k (ones 1)) (expt 2 80)))

  (test-case "T3-L7 (PS): Python reads the metadata and every entry of a checkpoint"
    (cond
      [(not (file-exists? script)) (displayln "skipped: no read_checkpoint.py (installed package)")]
      [(not python) (displayln "skipped: no python3")]
      [else
       (with-temporary-directory (dir #:template "format-parity-~a")
         (define path (build-path dir "p.ckpt"))
         (define tr (Small))
         (void (call-with-live-run #:trainer tr #:callbacks (list (stored (box probe-value)))
                                   (lambda (r) (save-checkpoint! r path))))
         (define seen (python-read path (build-path dir "p.json")))
         (define tree (read-checkpoint path))
         (check-equal? (hash-ref seen 'metadata) (hasheq 'format "rktorch-checkpoint" 'version "1"))
         (check-equal? (hash-ref seen 'keys)
                       (sort (map symbol->string (hash-keys tree)) string<?)
                       "the tree's keys")
         (define entries (hash-ref seen 'entries))
         (define names (map symbol->string (hash-keys entries)))
         (define references (hash-ref seen 'references))
         (check-equal? (remove-duplicates references) references "each entry is named once")
         (for ([name (in-list names)])
           (check-true (or (member name references) (string-prefix? name "state_dict.")) name))
         (for ([name (in-list references)])
           (check-true (and (member name names) #t) name))
         (for ([entry (in-list (state-dict tr))])
           (define name (string-append "state_dict." (car entry)))
           (check-equal? (entry-form (hash-ref entries (string->symbol name)))
                         (host-bytes (cdr entry))
                         name))
         (check-equal? (sorted (for/list ([(name e) (in-hash entries)]
                                          #:unless (string-prefix? (symbol->string name) "state_dict."))
                                 (entry-form e)))
                       (sorted (tensors-in (hash-remove tree "state_dict")))
                       "the other tensors, byte for byte"))])))
