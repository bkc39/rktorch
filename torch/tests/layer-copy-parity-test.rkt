#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           rackunit
           "../main.rkt"
           "../nn.rkt"
           (only-in "private/python-env.rkt" python-check python-torch-available?)
           (only-in "private/within-memory.rkt" within-memory))

  (define-layer leaf (weight up)
    #:init ()
    (set! weight (Parameter (ones 2)))
    (set! up (vector #f))
    #:forward ()
    (list weight up))

  (define-layer holder (kid loop)
    #:init (k)
    (set! kid k)
    (set! loop (let ([v (vector #f)]) (vector-set! v 0 v) v))
    #:forward ()
    (list kid loop))

  (define (kind-of x)
    (cond
      [(leaf? x) "leaf"]
      [(holder? x) "holder"]
      [(tensor? x) "tensor"]
      [else "list"]))

  (define (parts-of x)
    (cond
      [(layer? x) (x)]
      [(tensor? x) '()]
      [else (vector->list x)]))

  (define (shape root)
    (define ids (make-hasheq))
    (define nodes (make-hasheqv))
    (let visit ([x root])
      (cond
        [(hash-ref ids x #f) => values]
        [else
         (define id (hash-count ids))
         (hash-set! ids x id)
         (define edges (map visit (parts-of x)))
         (hash-set! nodes id (hasheq 'kind (kind-of x) 'edges edges))
         id]))
    (values (for/list ([i (in-range (hash-count nodes))]) (hash-ref nodes i))
            (hash-keys ids)))

  (cond
    [(not (python-torch-available?))
     (printf "[layer-copy-parity-test] skipped: python3 `torch` not available ~a\n"
             "(run inside the default `nix develop`)")]
    [else
     (define j (python-check "deepcopy_cycles.py"))
     (define h (holder (leaf)))
     (match-define (list kid _) (h))
     (match-define (list _ up) (kid))
     (vector-set! up 0 h)
     (define-values (original original-nodes) (shape h))
     (define-values (copied copied-nodes) (shape (within-memory 512 (lambda () (layer-copy h)))))
     (test-case "the original graph has the shape of PyTorch's"
       (check-equal? original (hash-ref j 'original)))
     (test-case "layer-copy's copy has the shape of copy.deepcopy's"
       (check-equal? copied (hash-ref j 'copy)))
     (test-case "the copy shares no node with the original, as copy.deepcopy's"
       (check-equal? (length (filter (lambda (n) (memq n original-nodes)) copied-nodes))
                     (hash-ref j 'shared)))]))
