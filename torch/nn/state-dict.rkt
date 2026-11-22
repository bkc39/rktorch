#lang racket/base

(require (only-in racket/contract/base -> ->* any cons/c listof or/c)
         (only-in racket/match match match-define)
         (only-in "../foreign.rkt" tensor-shape tensor? with-no-grad)
         (only-in "../generated.rkt" copy!)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "layer.rkt" layer-named-buffers layer-named-parameters layer?)
         (only-in "safetensors.rkt"
                  read-safetensors safetensors-header safetensors-payload-size safetensors-tensor
                  tag-dtype tag-width write-safetensors!))

(define/contract-out (state-dict model) ;; noqa
  (-> layer? (listof (cons/c string? tensor?)))
  (append (layer-named-parameters model "") (layer-named-buffers model "")))

(define/contract-out (save-state! model path) ;; noqa
  (-> layer? path-string? void?)
  (write-safetensors! 'save-state! path (state-dict model)))

;; an entry the load would copy, read from the header before anything is
(struct pending (name target tag shape offsets))

(define (field meta name key)
  (hash-ref meta key
            (lambda ()
              (raise-arguments-error 'load-state! "entry has no field"
                                     "entry" name
                                     "field" key))))

(define (read-pending name target meta)
  (pending name target
           (field meta name 'dtype)
           (field meta name 'shape)
           (field meta name 'data_offsets)))

(define (shape-matches? p)
  (equal? (pending-shape p) (tensor-shape (pending-target p))))

(define (expected-bytes p)
  (* (apply * (pending-shape p)) (tag-width (pending-tag p))))

(define (payload-fits? p payload-size)
  (match (pending-offsets p)
    [(list (? exact-nonnegative-integer? start)
           (? exact-nonnegative-integer? end))
     (and (<= start end payload-size)
          (= (- end start) (expected-bytes p)))]
    [_ #f]))

(define (mismatch-report strict? missing unexpected mismatched unsupported
                         misplaced)
  (append
   (if (and strict? (pair? missing)) (list "missing keys" missing) '())
   (if (and strict? (pair? unexpected))
       (list "unexpected keys" unexpected)
       '())
   (if (pair? mismatched)
       (list "shape mismatches (key, file, model)" mismatched)
       '())
   (if (pair? unsupported)
       (list "unsupported dtypes (key, dtype)" unsupported)
       '())
   (if (pair? misplaced)
       (list "payloads out of place (key, data_offsets, expected bytes)"
             misplaced)
       '())))

(define (entries-by-model-key header rename)
  (for/fold ([by-key (hash)])
            ([file-key (in-list (sort (map symbol->string (hash-keys header))
                                      string<?))])
    (define model-key (rename file-key))
    (cond
      [(not model-key) by-key]
      [(hash-ref by-key model-key #f)
       => (lambda (prior)
            (raise-arguments-error 'load-state!
                                   "#:rename maps two file keys to one model key"
                                   "model key" model-key
                                   "file keys" (list (car prior) file-key)))]
      [else
       (hash-set by-key model-key
                 (cons file-key
                       (hash-ref header (string->symbol file-key))))])))

(define/contract-out (load-state! model path ;; noqa
                                  #:strict? [strict? #t]
                                  #:rename [rename values])
  (->* [layer? path-string?]
       [#:strict? boolean? #:rename (-> string? (or/c string? #f))]
       any)
  (define file (read-safetensors path))
  (define payload-size (safetensors-payload-size file))
  (define header (safetensors-header file))
  (define by-key (entries-by-model-key header rename))
  (define entries (state-dict model))
  (define in-model
    (for/hash ([e (in-list entries)]) (values (car e) #t)))
  (define (meta-of name)
    (define entry (hash-ref by-key name #f))
    (and entry (cdr entry)))
  (define missing
    (for/list ([e (in-list entries)] #:unless (meta-of (car e))) (car e)))
  (define unexpected
    (sort (for/list ([(model-key entry) (in-hash by-key)]
                     #:unless (hash-ref in-model model-key #f))
            (car entry))
          string<?))
  (define pendings
    (for*/list ([e (in-list entries)]
                [meta (in-value (meta-of (car e)))]
                #:when meta)
      (read-pending (car e) (cdr e) meta)))
  (define mismatched
    (for/list ([p (in-list pendings)] #:unless (shape-matches? p))
      (list (pending-name p) (pending-shape p)
            (tensor-shape (pending-target p)))))
  (define unsupported
    (for/list ([p (in-list pendings)] #:unless (tag-dtype (pending-tag p)))
      (list (pending-name p) (pending-tag p))))
  (define misplaced
    (for/list ([p (in-list pendings)]
               #:when (shape-matches? p)
               #:when (tag-dtype (pending-tag p))
               #:unless (payload-fits? p payload-size))
      (list (pending-name p) (pending-offsets p) (expected-bytes p))))
  (define report
    (mismatch-report strict? missing unexpected mismatched unsupported
                     misplaced))
  (unless (null? report)
    (apply raise-arguments-error 'load-state!
           "checkpoint does not match the model" report))
  (with-no-grad
    (for ([p (in-list pendings)])
      (match-define (pending _ target tag shape (list start end)) p)
      ;; copy_ converts, so a file in one dtype loads into a model in another
      (copy! target (safetensors-tensor file tag shape start end) #f)))
  (unless strict?
    (values missing unexpected)))
