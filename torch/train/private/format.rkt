#lang racket/base

(require (only-in json jsexpr->string string->jsexpr)
         (only-in racket/dict in-dict)
         (only-in racket/file make-directory*)
         (only-in racket/match match)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "../../nn/safetensors.rkt"
                  read-safetensors safetensors-header safetensors-metadata
                  safetensors-payload-size safetensors-tensor tag-dtype tag-width
                  write-safetensors!)
         (only-in "value.rkt" checkpoint-value? decode-value encode-value string-keyed?))

(provide current-header-limit
         current-write-fault
         read-checkpoint-file
         write-checkpoint!)

(define format-name "rktorch-checkpoint")
(define format-version "1")

(define current-header-limit (make-parameter 100000000))

(define current-write-fault (make-parameter void))

(define (model-entries tree)
  (match (hash-ref tree "state_dict" #f)
    [(? hash? model)
     (for/list ([name (in-list (sort (filter string? (hash-keys model)) string<?))]
                #:when (tensor? (hash-ref model name)))
       (cons (string-append "state_dict." name) (hash-ref model name)))]
    [_ '()]))

(define decimal-digits-per-bit 3/10)

(define (written-size j)
  (define sizes (make-hasheq))
  (let size ([j j])
    (match j
      [(? string?) (+ 2 (string-length j))]
      [(? exact-integer?) (add1 (floor (* decimal-digits-per-bit (integer-length j))))]
      [(? list?)
       (hash-ref! sizes j (lambda () (for/fold ([n 1]) ([x (in-list j)]) (+ n 1 (size x)))))]
      [(? hash?)
       (hash-ref! sizes j
                  (lambda ()
                    (for/fold ([n 1]) ([(k x) (in-hash j)])
                      (+ n 2 (size (symbol->string k)) (size x)))))]
      [_ 1])))

(define (check-header-size! who n)
  (when (> n (current-header-limit))
    (raise-arguments-error who "the checkpoint is over safetensors' 100 MB header limit"
                           "limit in bytes" (current-header-limit))))

(define (header-text who j)
  (check-header-size! who (written-size j))
  (define text (jsexpr->string j))
  (check-header-size! who (string-utf-8-length text))
  text)

(define (write-checkpoint! who path tree)
  (define model (model-entries tree))
  (define model-names (for/hasheq ([(name t) (in-dict model)]) (values t name)))
  (define numbered (make-hasheq))
  (define (name-tensor t)
    (or (hash-ref model-names t #f)
        (hash-ref! numbered t (lambda () (format "$~a" (hash-count numbered))))))
  (define text (header-text who (encode-value tree name-tensor)))
  (define entries
    (append model
            (sort (for/list ([(t name) (in-hash numbered)]) (cons name t)) string<? #:key car)))
  (define-values (dir _name _dir?) (split-path (path->complete-path path)))
  (make-directory* dir)
  ;; make-temporary-file draws from `random` when a name is taken, and a save
  ;; must not move the run's Racket stream
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (write-safetensors! who path entries
                        #:metadata (hash "format" format-name
                                         "version" format-version
                                         "checkpoint" text)
                        #:on-header (lambda (header) (check-header-size! who (bytes-length header)))
                        #:on-written (lambda () ((current-write-fault))))))

(define (not-a-checkpoint who path)
  (raise-arguments-error who "not a rktorch checkpoint" "path" path))

(define (read-checkpoint-file who path)
  (define f
    (with-handlers ([(lambda (e) (and (exn:fail? e) (not (exn:fail:filesystem? e))))
                     (lambda (_) (not-a-checkpoint who path))])
      (read-safetensors path)))
  (define meta (or (safetensors-metadata f) (hash)))
  (unless (and (equal? (hash-ref meta "format" #f) format-name)
               (string? (hash-ref meta "checkpoint" #f)))
    (not-a-checkpoint who path))
  (define version (hash-ref meta "version" #f))
  (unless (equal? version format-version)
    (raise-arguments-error who "unsupported checkpoint version"
                           "file's version" version
                           "supported version" format-version
                           "path" path))
  (define header (safetensors-header f))
  (define tensors (make-hash))
  (define (malformed) (not-a-checkpoint who path))
  (define (tensor-named name)
    (hash-ref! tensors name
               (lambda ()
                 (match (hash-ref header (string->symbol name) #f)
                   [#f (raise-arguments-error who "the checkpoint names a missing tensor entry"
                                              "entry" name "path" path)]
                   [(hash-table ('dtype (? tag-dtype tag))
                                ('shape (list (? exact-nonnegative-integer? dims) ...))
                                ('data_offsets (list (? exact-nonnegative-integer? start)
                                                     (? exact-nonnegative-integer? end))))
                    #:when (and (<= start end (safetensors-payload-size f))
                                (= (- end start) (* (tag-width tag) (apply * dims))))
                    (safetensors-tensor f tag dims start end)]
                   [_ (malformed)]))))
  (define text
    (with-handlers ([exn:fail? (lambda (_) (malformed))])
      (string->jsexpr (hash-ref meta "checkpoint"))))
  (define tree (decode-value text tensor-named malformed))
  (unless (and (string-keyed? tree) (checkpoint-value? tree))
    (malformed))
  tree)
