#lang racket/base

(require (only-in json jsexpr->string string->jsexpr)
         (only-in racket/file make-directory*)
         (only-in racket/match match)
         (only-in "../../foreign.rkt" tensor?)
         (only-in "../../nn/safetensors.rkt"
                  read-safetensors safetensors-header safetensors-metadata
                  safetensors-payload-size safetensors-tensor write-safetensors!)
         (only-in "value.rkt" decode-value encode-value))

(provide current-write-fault
         read-checkpoint-file
         write-checkpoint!)

(define format-name "rktorch-checkpoint")
(define format-version "1")

(define safetensors-header-limit 100000000)

(define current-write-fault (make-parameter void))

(define (model-entries tree)
  (match (hash-ref tree "state_dict" #f)
    [(? hash? model)
     (for/list ([name (in-list (sort (filter string? (hash-keys model)) string<?))]
                #:when (tensor? (hash-ref model name)))
       (cons (string-append "state_dict." name) (hash-ref model name)))]
    [_ '()]))

(define (written-size j)
  (define sizes (make-hasheq))
  (let size ([j j])
    (match j
      [(or (? list?) (? hash?))
       (hash-ref! sizes j
                  (lambda ()
                    (for/fold ([n 1]) ([x (if (hash? j) (in-hash-values j) (in-list j))])
                      (+ n 1 (size x)))))]
      [_ 1])))

(define (header-text who j)
  (define (over-limit)
    (raise-arguments-error who "the checkpoint is over safetensors' 100 MB header limit"
                           "limit in bytes" safetensors-header-limit))
  (when (> (written-size j) safetensors-header-limit) (over-limit))
  (define text (jsexpr->string j))
  (when (> (string-utf-8-length text) safetensors-header-limit) (over-limit))
  text)

(define (write-checkpoint! who path tree)
  (define model (model-entries tree))
  (define model-names (for/hasheq ([e (in-list model)]) (values (cdr e) (car e))))
  (define numbered (make-hasheq))
  (define (name-tensor t)
    (or (hash-ref model-names t #f)
        (format "$~a" (hash-ref! numbered t (lambda () (hash-count numbered))))))
  (define text (header-text who (encode-value tree name-tensor)))
  (define others
    (for/list ([(t k) (in-hash numbered)])
      (cons k t)))
  (define entries
    (append model
            (for/list ([e (in-list (sort others < #:key car))])
              (cons (format "$~a" (car e)) (cdr e)))))
  (define-values (dir _name _dir?) (split-path (path->complete-path path)))
  (make-directory* dir)
  ;; make-temporary-file draws from `random` when a name is taken, and a save
  ;; must not move the run's Racket stream
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (write-safetensors! who path entries
                        #:metadata (hash "format" format-name
                                         "version" format-version
                                         "checkpoint" text)
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
  (define (tensor-named name)
    (hash-ref! tensors name
               (lambda ()
                 (match (hash-ref header (string->symbol name) #f)
                   [(hash-table ('dtype tag) ('shape shape) ('data_offsets (list start end)))
                    (unless (<= 0 start end (safetensors-payload-size f))
                      (not-a-checkpoint who path))
                    (safetensors-tensor f tag shape start end)]
                   [_ (raise-arguments-error who "the checkpoint names a missing tensor entry"
                                             "entry" name "path" path)]))))
  (define tree (decode-value (string->jsexpr (hash-ref meta "checkpoint")) tensor-named))
  (unless (hash? tree)
    (not-a-checkpoint who path))
  tree)
