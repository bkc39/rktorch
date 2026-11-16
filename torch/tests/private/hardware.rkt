#lang racket/base

(require (only-in racket/port open-output-nowhere)
         (only-in racket/string string-split)
         (only-in racket/system system*))

(provide tf32-hardware?)

(define (smi-command)
  (define smi (find-executable-path "nvidia-smi"))
  (and smi (list smi "--query-gpu=compute_cap" "--format=csv,noheader")))

(define (command-output command seconds)
  (define out (open-output-string))
  (define ok? #f)
  (define started (make-custodian))
  (define probe
    (parameterize ([current-custodian started]
                   [current-subprocess-custodian-mode 'kill])
      (thread (lambda ()
                (parameterize ([current-output-port out]
                               [current-error-port (open-output-nowhere)])
                  (set! ok? (apply system* command)))))))
  (define finished? (sync/timeout seconds probe))
  (custodian-shutdown-all started)
  (and finished? ok? (get-output-string out)))

(define (compute-capabilities command seconds)
  (define text (and command (command-output command seconds)))
  (define caps (and text (map string->number (string-split text))))
  (and caps (pair? caps) (andmap real? caps) caps))

(define (tf32-hardware? #:command [command (smi-command)] #:seconds [seconds 30])
  (define caps (compute-capabilities command seconds))
  (cond
    [(not caps) 'unknown]
    [(andmap (lambda (c) (>= c 8)) caps) #t]
    [(andmap (lambda (c) (< c 8)) caps) #f]
    [else 'unknown]))
