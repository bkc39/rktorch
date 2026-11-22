#lang racket/base

(require (only-in racket/file make-directory*)
         (only-in racket/match match)
         (only-in "../../nn/safetensors.rkt"
                  read-safetensors safetensors-header safetensors-metadata safetensors-tensor
                  write-safetensors!)
         (only-in "json.rkt" json-text text->json)
         (only-in "value.rkt" decode-value encode-value))

(provide current-write-fault
         read-checkpoint-file
         write-checkpoint!)

(define format-name "rktorch-checkpoint")
(define format-version "1")

(define current-write-fault (make-parameter void))

(define (model-entry? path)
  (match path
    [(list "state_dict" (? string?)) #t]
    [_ #f]))

(define (write-checkpoint! who path tree)
  (define entries '())
  (define (name-tensor t at)
    (define name
      (if (model-entry? at)
          (string-append "state_dict." (cadr at))
          (format "$~a" (length entries))))
    (set! entries (cons (cons name t) entries))
    name)
  (define text (json-text (encode-value tree name-tensor)))
  (define-values (dir _name _dir?) (split-path (path->complete-path path)))
  (make-directory* dir)
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (write-safetensors! who path (reverse entries)
                        #:metadata (hash "format" format-name
                                         "version" format-version
                                         "checkpoint" text)
                        #:on-written (lambda () ((current-write-fault))))))

(define (read-checkpoint-file who path)
  (define f (read-safetensors path))
  (define meta (or (safetensors-metadata f) (hash)))
  (unless (and (equal? (hash-ref meta "format" #f) format-name)
               (string? (hash-ref meta "checkpoint" #f)))
    (raise-arguments-error who "not a rktorch checkpoint" "path" path))
  (define version (hash-ref meta "version" #f))
  (unless (equal? version format-version)
    (raise-arguments-error who "unsupported checkpoint version"
                           "file's version" version
                           "supported version" format-version
                           "path" path))
  (define header (safetensors-header f))
  (define (tensor-named name)
    (match (hash-ref header (string->symbol name) #f)
      [(hash-table ('dtype tag) ('shape shape) ('data_offsets (list start end)))
       (safetensors-tensor f tag shape start end)]
      [_ (raise-arguments-error who "the checkpoint names a missing tensor entry"
                                "entry" name "path" path)]))
  (define tree (decode-value (text->json (hash-ref meta "checkpoint")) tensor-named))
  (unless (hash? tree)
    (raise-arguments-error who "not a rktorch checkpoint" "path" path))
  tree)
