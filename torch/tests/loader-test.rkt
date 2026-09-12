#lang racket/base

(module+ test
  (require rackunit
           "../data/loader.rkt"
           "../main.rkt")

  (define xs (reshape (arange 12) 6 2))
  (define ys (arange 6 #:dtype 'int64))

  (test-case "generators: seeded streams replay and continue"
    (define a (make-generator 3))
    (define b (make-generator 3))
    (check-true (generator? a))
    (check-false (generator? (ones 1)))
    (define a1 (tensor->list (randperm 16 #:generator a)))
    (define a2 (tensor->list (randperm 16 #:generator a)))
    (check-equal? (sort a1 <) (for/list ([i (in-range 16)]) i))
    (check-false (equal? a1 a2) "the stream continues across draws")
    (check-equal? (tensor->list (randperm 16 #:generator b)) a1)
    (check-equal? (tensor->list (randperm 16 #:generator b)) a2)
    (check-equal? (tensor-dtype (randperm 4)) 'int64)
    (check-equal? (tensor-shape (randperm 0)) '(0))
    (check-exn exn:fail:contract? (lambda () (randperm -1)))
    (check-true (generator? (make-generator (sub1 (expt 2 64)))))
    (check-exn #rx"seed" (lambda () (make-generator (expt 2 64)))))

  (test-case "generators are released by the guarded finalizer"
    (define (runs) (cdr (assq 'runs (finalizer-diagnostics))))
    (define before (runs))
    (for ([_i (in-range 32)]) (void (make-generator 1)))
    (let loop ([i 0])
      (collect-garbage)
      (unless (or (> (runs) before) (>= i 50))
        (sleep 0.01)
        (loop (add1 i))))
    (check-true (> (runs) before) "a collected generator ran the finalizer"))

  (test-case "generator draws leave the global stream alone"
    (manual-seed! 5)
    (define expected (tensor->list (randn 4)))
    (manual-seed! 5)
    (void (randperm 64 #:generator (make-generator 9)))
    (check-equal? (tensor->list (randn 4)) expected))

  (test-case "tensor-dataset: length, items, and batch paths"
    (define ds (tensor-dataset xs ys))
    (check-true (dataset? ds))
    (check-true (tensor-dataset? ds))
    (check-equal? (dataset-length ds) 6)
    (define-values (x2 y2) (dataset-ref ds 2))
    (check-equal? (tensor->list x2) '(4.0 5.0))
    (check-equal? (tensor->list y2) '(2))
    (define-values (xr yr) (dataset-batch ds '(1 2 3) default-collate))
    (check-equal? (tensor-shape xr) '(3 2))
    (check-equal? (tensor->list yr) '(1 2 3) "a run narrows")
    (define-values (xg yg) (dataset-batch ds '(5 0 3) default-collate))
    (check-equal? (tensor->list yg) '(5 0 3) "a permutation gathers")
    (check-equal? (tensor->list xg) '(10.0 11.0 0.0 1.0 6.0 7.0))
    (define-values (_xt yt)
      (dataset-batch ds (tensor '(4 1) #:dtype 'int64) default-collate))
    (check-equal? (tensor->list yt) '(4 1) "an index tensor gathers too")
    (check-exn exn:fail:contract? (lambda () (tensor-dataset xs (ones 5))))
    (check-exn #rx"same-leading-dimension" (lambda () (tensor-dataset xs (ones 5))))
    (check-exn #rx"batched-tensor" (lambda () (tensor-dataset (tensor 1.0))))
    (check-exn #rx"int64-vector"
               (lambda () (dataset-batch ds (tensor '(1.0 2.0)) default-collate)))
    (check-exn #rx"int64-vector"
               (lambda () (dataset-batch ds (reshape (arange 4 #:dtype 'int64) 2 2)
                                         default-collate)))
    (check-exn #rx"int64-vector"
               (lambda () (dataset-batch ds (arange 0 #:dtype 'int64) default-collate)))
    (check-exn exn:fail:contract? (lambda () (dataset-batch ds '() default-collate)))
    (check-exn #rx"rectangular-tensor-items"
               (lambda () (default-collate (list (list (ones 2) (ones 2)) (list (ones 2))))))
    (check-exn #rx"rectangular-tensor-items" (lambda () (default-collate '(()))))
    (check-exn exn:fail:contract?
               (lambda () (dataset-batch ds '(-6 -5 -4) default-collate))
               "indices are natural numbers, not end-relative")
    (check-exn exn:fail:contract? (lambda () (dataset-ref ds -1)))
    (check-exn #rx"same-leading-dimension"
               (lambda () (tensor-dataset xs (ones 5 2))))
    (check-equal? (format "~a" ds) "#<tensor-dataset>")
    ;; the native path hands out views: a run's batch follows the source
    (define src (ones 6 2))
    (define-values (view) (dataset-batch (tensor-dataset src) '(1 2 3) default-collate))
    (mul! src 3.0)
    (check-equal? (tensor->list view) '(3.0 3.0 3.0 3.0 3.0 3.0)
                  "default-collate from outside the module takes the native path")
    ;; a custom collate sees the items, as DataLoader's collate_fn does
    (define counted
      (dataloader ds #:batch-size 4
                  #:collate (lambda (items)
                              (values (length items)
                                      (map (lambda (it) (item (cadr it))) items)))))
    (check-equal? (for/list ([(n ids) (in-dataloader counted)]) (cons n ids))
                  '((4 0 1 2 3) (2 4 5))))

  (test-case "a hand-written dataset goes through dataset-ref and collate"
    (struct Squares (n)
      #:methods gen:dataset
      [(define (dataset-length self) (Squares-n self))
       (define (dataset-ref _self i) (values (full (* i i) 2) (tensor i)))])
    (define ds (Squares 5))
    (define-values (xb yb) (dataset-batch ds '(1 3) default-collate))
    (check-equal? (tensor-shape xb) '(2 2))
    (check-equal? (tensor->list xb) '(1.0 1.0 9.0 9.0))
    (check-equal? (tensor->list yb) '(1 3))
    (define loader (dataloader ds #:batch-size 2))
    (check-equal? (dataloader-length loader) 3)
    (check-equal? (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb))
                  '((0 1) (2 3) (4)))
    (define summed
      (dataloader ds #:batch-size 2
                  #:collate (lambda (items)
                              (values (apply + (map (lambda (it) (item (cadr it)))
                                                    items))))))
    (check-equal? (for/list ([s (in-dataloader summed)]) s) '(1 5 4)
                  "a custom collate returns whatever the trainer wants"))

  (test-case "dataloader: sequential batches are narrow views in order"
    (define ds (tensor-dataset xs ys))
    (define loader (dataloader ds #:batch-size 4))
    (check-equal? (dataloader-length loader) 2)
    (check-equal? (for/list ([(xb yb) (in-dataloader loader)])
                    (cons (tensor-shape xb) (tensor->list yb)))
                  '(((4 2) 0 1 2 3) ((2 2) 4 5)))
    (check-equal? (dataloader-length (dataloader ds #:batch-size 4 #:drop-last? #t))
                  1)
    (check-equal? (for/list ([(_xb yb) (in-dataloader
                                        (dataloader ds #:batch-size 4
                                                    #:drop-last? #t))])
                    (tensor->list yb))
                  '((0 1 2 3)))
    ;; full batch, no shuffle: exactly the dataset's tensors, as the oracles use
    (define full
      (for/first ([(xb yb) (in-dataloader (dataloader ds #:batch-size 6))])
        (list xb yb)))
    (check-equal? (tensor->list (car full)) (tensor->list xs))
    (check-equal? (tensor->list (cadr full)) (tensor->list ys))
    ;; an unshuffled traversal still draws the iterator's base seed
    (define g (make-generator 5))
    (for* ([_e (in-range 2)]
           [(_xb _yb) (in-dataloader (dataloader ds #:batch-size 4 #:generator g))])
      (void))
    (define twin (make-generator 5))
    (void (draw-seed #:generator twin))
    (void (draw-seed #:generator twin))
    (check-equal? (tensor->list (randperm 6 #:generator g))
                  (tensor->list (randperm 6 #:generator twin))
                  "one word per unshuffled epoch"))

  (test-case "the remainder permutation waits until the first is used up"
    (define ds (tensor-dataset xs ys))
    (define (run batch-size take)
      (define g (make-generator 11))
      (define loader
        (dataloader ds #:batch-size batch-size #:shuffle? #t #:generator g))
      (define n-first (if (eq? take 'all) (dataloader-length loader) 1))
      ;; in-range first: the loader's exhaustion check never runs
      (define first-orders
        (for/list ([_i (in-range n-first)] [(_xb yb) (in-dataloader loader)])
          (tensor->list yb)))
      (define second-orders
        (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb)))
      (list first-orders second-orders (tensor->list (randperm 6 #:generator g))))
    (define (replay batch-size take)
      (define g (make-generator 11))
      (define (perm) (tensor->list (randperm 6 #:generator g)))
      (define (batches p)
        (for/list ([k (in-range (quotient (+ 5 batch-size) batch-size))])
          (for/list ([i (in-list p)] [j (in-naturals)]
                     #:when (and (>= j (* k batch-size))
                                 (< j (* (add1 k) batch-size))))
            i)))
      (void (draw-seed #:generator g))
      (define first-perm (batches (perm)))
      ;; batch 4 leaves a partial last batch, so a full first epoch drew the
      ;; remainder before yielding it; batch 3 divides 6 and drew nothing
      (when (and (eq? take 'all) (positive? (remainder 6 batch-size)))
        (void (perm)))
      (void (draw-seed #:generator g))
      (define second-perm (batches (perm)))
      (void (perm))
      (list (if (eq? take 'all) first-perm (list (car first-perm)))
            second-perm
            (perm)))
    (check-equal? (run 4 'one) (replay 4 'one) "one batch of four, abandoned")
    (check-equal? (run 4 'all) (replay 4 'all) "all batches of four, not exhausted")
    (check-equal? (run 3 'all) (replay 3 'all) "all batches of three, not exhausted")
    (define g (make-generator 11))
    (define exhausted
      (for/list ([(_xb yb) (in-dataloader
                            (dataloader ds #:batch-size 3 #:shuffle? #t
                                        #:generator g))])
        (tensor->list yb)))
    (define twin (make-generator 11))
    (void (draw-seed #:generator twin))
    (void (randperm 6 #:generator twin))
    (void (randperm 6 #:generator twin))
    (check-equal? (length exhausted) 2)
    (check-equal? (tensor->list (randperm 6 #:generator g))
                  (tensor->list (randperm 6 #:generator twin))
                  "exhaustion draws the remainder")
    ;; a traversal started but never asked for a batch drew only its base seed
    (define g-idle (make-generator 11))
    (define-values (_next _more?)
      (sequence-generate
       (in-dataloader (dataloader ds #:batch-size 3 #:shuffle? #t #:generator g-idle))))
    (define twin-idle (make-generator 11))
    (void (draw-seed #:generator twin-idle))
    (check-equal? (tensor->list (randperm 6 #:generator g-idle))
                  (tensor->list (randperm 6 #:generator twin-idle))
                  "the permutation waits for the first batch"))

  (test-case "dataloader: shuffle draws a permutation per traversal"
    (define ds (tensor-dataset xs ys))
    (define (orders g)
      (define loader
        (dataloader ds #:batch-size 4 #:shuffle? #t #:generator g))
      (for/list ([_e (in-range 2)])
        (for/list ([(_xb yb) (in-dataloader loader)]) (tensor->list yb))))
    (define first-run (orders (make-generator 1)))
    (check-equal? first-run (orders (make-generator 1)) "seeded replay")
    (for ([epoch (in-list first-run)])
      (check-equal? (sort (apply append epoch) <) '(0 1 2 3 4 5)))
    (check-false (equal? (car first-run) (cadr first-run))
                 "the second epoch is a fresh draw from the same stream")
    ;; one epoch draws what a DataLoader epoch draws: a seed word, then the
    ;; permutation, then the sampler's discarded remainder permutation
    (define g (make-generator 1))
    (void (draw-seed #:generator g))
    (define expected (tensor->list (randperm 6 #:generator g)))
    (define loader
      (dataloader ds #:batch-size 6 #:shuffle? #t #:generator (make-generator 1)))
    (define batch
      (for/first ([(xb yb) (in-dataloader loader)]) (list xb yb)))
    (check-equal? (tensor->list (cadr batch)) expected
                  "the batch is the permutation")
    (check-equal? (tensor->list (car batch))
                  (apply append (for/list ([i (in-list expected)])
                                  (tensor->list (select xs 0 i))))))

  (test-case "shuffle without a generator takes two words of the global stream"
    (define ds (tensor-dataset xs ys))
    (define (epoch-order)
      (for/list ([(_xb yb) (in-dataloader
                            (dataloader ds #:batch-size 6 #:shuffle? #t))])
        (tensor->list yb)))
    (manual-seed! 11)
    (define seen (epoch-order))
    (define after (tensor->list (randn 4)))
    ;; the replay: a base seed, then a seed for a fresh generator
    (manual-seed! 11)
    (void (draw-seed))
    (define g (make-generator (draw-seed)))
    (check-equal? seen (list (tensor->list (randperm 6 #:generator g))))
    (check-equal? (tensor->list (randn 4)) after
                  "exactly two words leave the global stream per epoch")
    (manual-seed! 11)
    (check-equal? (epoch-order) seen "the global stream replays under a seed"))

  (test-case "a device-resident dataset batches on its device"
    (for ([dev (in-list (list (and (cuda-available?) (cuda-device))
                              (and (mps-available?) (mps-device))))]
          #:when dev)
      (check-exn #rx"same-device" (lambda () (tensor-dataset (to xs dev) ys)))
      (define ds (tensor-dataset (to xs dev) (to ys dev)))
      (define loader
        (dataloader ds #:batch-size 4 #:shuffle? #t #:generator (make-generator 3)))
      (define seen
        (for/list ([(xb yb) (in-dataloader loader)])
          (check-equal? (tensor-device xb) dev)
          (check-equal? (tensor-device yb) dev)
          (tensor->list yb)))
      (check-equal? (sort (apply append seen) <) '(0 1 2 3 4 5))
      (check-equal? seen
                    (for/list ([(_xb yb) (in-dataloader
                                          (dataloader (tensor-dataset xs ys)
                                                      #:batch-size 4 #:shuffle? #t
                                                      #:generator (make-generator 3)))])
                      (tensor->list yb))
                    "the permutation is drawn on the CPU whatever the device")))

  (test-case "in-epochs numbers epochs and continues the stream"
    (define ds (tensor-dataset xs ys))
    (define loader
      (dataloader ds #:batch-size 4 #:shuffle? #t #:generator (make-generator 2)))
    (define seen
      (for/list ([(epoch _xb yb) (in-epochs loader 3)])
        (cons epoch (tensor->list yb))))
    (check-equal? (map car seen) '(0 0 1 1 2 2))
    (define by-epoch
      (for/list ([e (in-range 3)])
        (apply append (for/list ([s (in-list seen)] #:when (= (car s) e))
                        (cdr s)))))
    (define twin
      (dataloader ds #:batch-size 4 #:shuffle? #t #:generator (make-generator 2)))
    (check-equal? by-epoch
                  (for/list ([_e (in-range 3)])
                    (apply append
                           (for/list ([(_xb yb) (in-dataloader twin)])
                             (tensor->list yb))))
                  "in-epochs and repeated in-dataloader traversals agree")
    (check-equal? (for/list ([(epoch _xb _yb) (in-epochs loader 0)]) epoch)
                  '())
    ;; an epoch with no batches still draws, as a drained DataLoader does
    (define g (make-generator 4))
    (define empty (dataloader ds #:batch-size 8 #:shuffle? #t #:drop-last? #t
                              #:generator g))
    (check-equal? (dataloader-length empty) 0)
    (check-equal? (for/list ([(epoch _xb _yb) (in-epochs empty 3)]) epoch) '())
    (define g2 (make-generator 4))
    (for* ([_e (in-range 3)]
           [(_xb _yb) (in-dataloader (dataloader ds #:batch-size 8 #:shuffle? #t
                                                 #:drop-last? #t #:generator g2))])
      (void))
    (check-equal? (tensor->list (randperm 6 #:generator g))
                  (tensor->list (randperm 6 #:generator g2))
                  "three empty epochs consumed three epochs of the stream")))
