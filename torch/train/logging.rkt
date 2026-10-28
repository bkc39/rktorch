#lang racket/base

(require (only-in json jsexpr->string)
         (only-in racket/contract/base
                  -> ->* ->i and/c any/c contract-out listof or/c property/c
                  unsupplied-arg?)
         (only-in racket/file make-directory*)
         (only-in racket/generic define-generics)
         (only-in racket/match match match*)
         (only-in racket/math infinite? nan?)
         (only-in racket/string string-join)
         (only-in "../foreign.rkt" dtype numel tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "private/book.rkt" accumulators host-reads note-epoch! note-step!)
         (only-in "private/hooks.rkt"
                  log-rule-default-on-epoch? log-rule-default-on-step?
                  log-rule-on-epoch log-rule-on-step)
         (only-in "private/state.rkt" run-batch-size run-book run-stage)
         (only-in "private/types.rkt" define-named-contract)
         (only-in "run.rkt" hook-log-rule logging-run/c))

(provide gen:training-logger
         logger-finalize!
         logger-log-hyperparams!
         logger-log-metrics!
         (contract-out [training-logger? (-> any/c boolean?)]))

(define-generics training-logger
  (logger-log-hyperparams! training-logger params)
  (logger-log-metrics! training-logger metrics step)
  (logger-finalize! training-logger status))

(define-named-contract reduce/c (or/c 'mean 'sum 'max 'min))

(define-named-contract scalar-tensor/c
  (and/c tensor? (property/c numel 1)))

(define-named-contract metric-value/c (or/c real? scalar-tensor/c))

(define-named-contract loss/c
  (and/c scalar-tensor/c
         (property/c dtype (or/c 'float32 'float64 'float16 'bfloat16))))

(define-named-contract logger-spec/c
  (or/c #f training-logger? (listof training-logger?)))

(module+ private
  (provide epoch-accumulators host-reads))

(define (epoch-accumulators r)
  (accumulators (run-book r) 'train))

(define (allowed? flag rule-values r)
  (or (unsupplied-arg? flag)
      (not (hook-log-rule r))
      (memq flag (rule-values (hook-log-rule r)))))

(define/contract-out (run-log! r key value
                               #:progress? [progress? #f]
                               #:on-step? [on-step? (log-rule-default-on-step?
                                                     (hook-log-rule r))]
                               #:on-epoch? [on-epoch? (log-rule-default-on-epoch?
                                                       (hook-log-rule r))]
                               #:reduce [reduce 'mean]
                               #:batch-size [batch-size #f]
                               #:logger? [logger? #t])
  (->i ([r logging-run/c] [key symbol?] [value metric-value/c])
       (#:progress? [progress? boolean?]
        #:on-step? [on-step? boolean?]
        #:on-epoch? [on-epoch? boolean?]
        #:reduce [reduce reduce/c]
        #:batch-size [batch-size (or/c #f exact-positive-integer?)]
        #:logger? [logger? boolean?])
       #:pre/name (r on-step?) "this hook logs only per epoch"
       (allowed? on-step? log-rule-on-step r)
       #:pre/name (r on-epoch?) "this hook logs only per epoch"
       (allowed? on-epoch? log-rule-on-epoch r)
       [result void?])
  (define b (run-book r))
  (define both? (and on-step? on-epoch?))
  (define loop (if (eq? (run-stage r) 'train) 'train 'eval))
  (define weight (or batch-size (run-batch-size r)))
  (cond
    [(eq? (run-stage r) 'sanity-check) (void)]
    [else
     (when on-step? (note-step! b key value logger? both?))
     (when on-epoch? (note-epoch! b loop key value weight reduce logger? both?))]))

(define (metric-text metrics)
  (string-join (for/list ([k (in-list (sort (hash-keys metrics) symbol<?))])
                 (format "~a=~a" k (hash-ref metrics k)))
               " "))

(struct stdout-sink (out)
  #:methods gen:training-logger
  [(define (logger-log-hyperparams! l params)
     (fprintf (stdout-sink-out l) "hparams ~a\n" (metric-text params)))
   (define (logger-log-metrics! l metrics step)
     (fprintf (stdout-sink-out l) "step ~a ~a\n" step (metric-text metrics)))
   (define (logger-finalize! l status)
     (fprintf (stdout-sink-out l) "~a\n" status))])

(define/contract-out (stdout-logger #:out [out (current-output-port)])
  (->* () (#:out output-port?) training-logger?)
  (stdout-sink out))

(define (json-value v)
  (cond
    [(exact-integer? v) v]
    [(or (nan? v) (infinite? v)) (format "~a" v)]
    [else (exact->inexact v)]))

(define (readable v)
  (if (or (real? v) (string? v) (symbol? v) (boolean? v)) v (format "~a" v)))

(define (append-line path line)
  (with-output-to-file path #:exists 'append (lambda () (displayln line))))

(struct file-sink (metrics hparams write-metrics)
  #:methods gen:training-logger
  [(define (logger-log-hyperparams! l params)
     (with-output-to-file (file-sink-hparams l) #:exists 'truncate
       (lambda () (writeln (for/hasheq ([(k v) (in-hash params)]) (values k (readable v)))))))
   (define (logger-log-metrics! l metrics step)
     ((file-sink-write-metrics l) (file-sink-metrics l) metrics step))
   (define (logger-finalize! l status) (void))])

(define (file-logger who dir name exists write-metrics)
  (make-directory* dir)
  (define metrics (build-path dir name))
  (define hparams (build-path dir "hparams.rktd"))
  (define present (filter file-exists? (list metrics hparams)))
  (match* (exists present)
    [('error (cons path _))
     (raise (exn:fail:filesystem:exists
             (format "~a: ~a already exists" who path)
             (current-continuation-marks)))]
    [('replace _) (for-each delete-file present)]
    [(_ _) (void)])
  (file-sink metrics hparams write-metrics))

(define (jsonl-line path metrics step)
  (append-line path
               (jsexpr->string
                (for/fold ([h (hasheq 'step step)]) ([(k v) (in-hash metrics)])
                  (hash-set h k (json-value v))))))

(define (csv-rows path metrics step)
  (with-output-to-file path #:exists 'append
    (lambda ()
      (when (zero? (file-size path)) (displayln "step,key,value"))
      (for ([k (in-list (sort (hash-keys metrics) symbol<?))])
        (printf "~a,~a,~a\n" step k (hash-ref metrics k))))))

(define/contract-out (jsonl-logger dir #:exists [exists 'error])
  (->* (path-string?) (#:exists (or/c 'error 'append 'replace)) training-logger?)
  (file-logger 'jsonl-logger dir "metrics.jsonl" exists jsonl-line))

(define/contract-out (csv-logger dir #:exists [exists 'error])
  (->* (path-string?) (#:exists (or/c 'error 'append 'replace)) training-logger?)
  (file-logger 'csv-logger dir "metrics.csv" exists csv-rows))
