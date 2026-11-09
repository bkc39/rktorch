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

  (define (tuple-in-list)
    (define l (vector #f))
    (define t (vector-immutable l))
    (vector-set! l 0 t)
    t)

  (define-layer holder (kid loop twice const)
    #:init (k)
    (set! kid k)
    (set! loop (let ([v (vector #f)]) (vector-set! v 0 v) v))
    (set! twice (let ([t (tuple-in-list)]) (vector-immutable t t)))
    (set! const (vector-immutable 1 2))
    #:forward ()
    (list kid loop twice const))

  (define (kind-of x)
    (cond
      [(leaf? x) "leaf"]
      [(holder? x) "holder"]
      [(tensor? x) "tensor"]
      [(immutable? x) "tuple"]
      [else "list"]))

  (define (parts-of x)
    (cond
      [(layer? x) (x)]
      [(tensor? x) '()]
      [else (filter (lambda (p) (not (number? p))) (vector->list x))]))

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
     (match-define (list kid _ _ _) (h))
     (match-define (list _ up) (kid))
     (vector-set! up 0 h)
     (define-values (original original-nodes) (shape h))
     (define-values (copied copied-nodes) (shape (within-memory 512 (lambda () (layer-copy h)))))
     (test-case "the original graph has the shape of PyTorch's"
       (check-equal? original (hash-ref j 'original)))
     (test-case "layer-copy's copy has the shape of copy.deepcopy's"
       (check-equal? copied (hash-ref j 'copy)))
     (test-case "the copy shares with the original what copy.deepcopy's does"
       (define originals (for/hasheq ([n (in-list original-nodes)]) (values n #t)))
       (check-equal? (for/sum ([n (in-list copied-nodes)]) (if (hash-ref originals n #f) 1 0))
                     (hash-ref j 'shared)))]))
