#lang racket/base

(require (for-syntax racket/base)
         (only-in compiler/find-exe find-exe)
         (only-in racket/cmdline command-line)
         (only-in racket/file make-directory*)
         (only-in racket/format ~a)
         (only-in racket/list append* shuffle)
         (only-in racket/match match)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-split)
         (only-in "harness.rkt" write-record)
         (only-in "m0.rkt" cell-args run-blocks)
         (only-in "op-cost.rkt" default-out-dir stamp)
         (only-in "op-cost-pipelines.rkt" single-threaded-environment)
         (only-in "pytorch.rkt" child-records require-pytorch twin-records)
         (only-in "session.rkt" fresh-run-id)
         (only-in "summary.rkt" read-records)
         (only-in "summary-m1.rkt" m1-markdown))

(provide m1-arms
         m1-plan
         m1-settings)

(define-runtime-path cell-module "m1-cell.rkt")

(define (m1-arms workers trip)
  (cond
    [trip '(rktorch)]
    [(zero? workers) '(rktorch pytorch)]
    [else '(rktorch pytorch-persistent)]))

(define (m1-plan #:rounds rounds #:pipelines pipelines #:workers workers #:trips trips #:seed seed)
  (define cells
    (for*/list ([p (in-list pipelines)] [w (in-list workers)] [t (in-list trips)])
      (list p w t)))
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed seed)
    (append*
     (for/list ([round (in-range rounds)])
       (for/list ([c (in-list (shuffle cells))] [i (in-naturals)])
         (match c
           [(list p w t)
            (hasheq 'round round 'block i 'pipeline p 'workers w 'trip t
                    'arms (shuffle (m1-arms w t)))]))))))

(define (one-record arm records)
  (match records
    [(list r) r]
    [_ (raise-user-error 'm1 "~a printed ~a records, not one" arm (length records))]))

(define (m1-child block arm settings #:log log)
  (define args (cell-args block arm settings))
  (define limit (hash-ref settings 'time-limit))
  (define trip (hash-ref block 'trip))
  (parameterize ([current-environment-variables (single-threaded-environment)])
    (one-record
     arm
     (match arm
       ['rktorch
        (child-records (find-exe)
                       (append (list cell-module) (if trip (list "--trip" (~a trip)) '()) args)
                       #:log log #:time-limit limit #:who 'm1-cell)]
       [_ (twin-records "loaders.py" args #:log log #:time-limit limit)]))))

(define (run-m1 settings #:log [log (current-error-port)])
  (require-pytorch 'm1)
  (define dir (hash-ref settings 'out-dir))
  (make-directory* dir)
  (define jsonl (build-path dir (format "m1-~a-~a-~a.jsonl" (hash-ref settings 'tree)
                                        (hash-ref settings 'racket) (stamp))))
  (define run-id (fresh-run-id))
  (define plan
    (m1-plan #:rounds (hash-ref settings 'rounds) #:pipelines (hash-ref settings 'pipelines)
             #:workers (hash-ref settings 'workers) #:trips (hash-ref settings 'trips)
             #:seed (hash-ref settings 'seed)))
  (call-with-output-file jsonl #:exists 'error
    (lambda (out)
      (define (emit r)
        (write-record (hash-set* r 'run_id run-id 'tree (hash-ref settings 'tree)) out)
        (flush-output out))
      (run-blocks plan settings #:emit emit #:log log #:run m1-child)))
  (fprintf log "records: ~a\n" jsonl))

(define (default-settings)
  (hasheq 'rounds 3 'racket (version) 'tree "master" 'out-dir (default-out-dir) 'max-load 2.0
          'pipelines '(cifar-png jpeg) 'workers '(0 4 7) 'trips '(#f)
          'samples #f 'warmup 1 'epochs 2 'dataset "full" 'seed 589 'time-limit 1800))

(define (m1-settings argv)
  (define settings (default-settings))
  (define (put! key value) (set! settings (hash-set settings key value)))
  (define (numbers text)
    (for/list ([t (in-list (string-split text ","))])
      (match t
        ["-" #f]
        [_ (or (string->number t) (raise-user-error 'm1 "expected a number, got ~s" t))])))
  (define summarize
    (command-line
     #:program "m1"
     #:argv argv
     #:once-each
     [("--rounds") n "Rounds, each block once per round (default 3)"
                   (put! 'rounds (string->number n))]
     [("--tree") t "Label of the checkout measured, e.g. master or int" (put! 'tree t)]
     [("--out-dir") d "Directory for m1-<tree>-<racket>-<stamp>.jsonl" (put! 'out-dir d)]
     [("--max-load") l "Wait for fewer busy cores before a block" (put! 'max-load (string->number l))]
     [("--pipelines") s "Comma-separated: cifar-png,jpeg"
                      (put! 'pipelines (map string->symbol (string-split s ",")))]
     [("--workers") s "Comma-separated loader worker counts (default 0,4,7)"
                    (put! 'workers (numbers s))]
     [("--trips") s "Comma-separated collect trips in MB, - for Racket's (default -)"
                  (put! 'trips (numbers s))]
     [("--samples") n "Samples per epoch for every pipeline" (put! 'samples (string->number n))]
     [("--warmup") n "Untimed epochs" (put! 'warmup (string->number n))]
     [("--epochs") n "Timed epochs per arm" (put! 'epochs (string->number n))]
     [("--dataset") d "The CIFAR-10 PNG tree: full or fixture" (put! 'dataset d)]
     [("--seed") n "Seed of the block and arm order" (put! 'seed (string->number n))]
     [("--time-limit") s "Kill an arm after s seconds" (put! 'time-limit (string->number s))]
     #:args paths paths))
  (hash-set settings 'summarize (and (pair? summarize) summarize)))

(module+ main
  (define settings (m1-settings (current-command-line-arguments)))
  (match (hash-ref settings 'summarize)
    [#f (run-m1 settings)]
    [paths (write-string (m1-markdown (read-records paths) #:limit (hash-ref settings 'max-load)))]))
