#lang racket/base

(module+ test
  (require (only-in racket/dict in-dict)
           (only-in racket/list append-map check-duplicates first last range remove-duplicates)
           (only-in racket/match match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../data/loader.rkt"
                    dataloader dataset-batch dataset-length dataset-ref default-collate
                    in-dataloader tensor-dataset)
           (only-in "../main.rkt"
                    * + - / @ argmax cuda-available? cuda-device device device-type dtype
                    has-grad? index-select item manual-seed! mps-available? mps-device narrow
                    native-memory-use randn reclaim-native-memory! requires-grad? reshape select
                    shape tensor tensor->list to to-dtype zeros)
           (only-in "../nn.rkt"
                    Dropout Flatten Linear Sequential children eval! layer-mode layer-set-mode!
                    parameters train!)
           (only-in "../train.rkt"
                    knn-accuracy linear-probe linear-probe-bias linear-probe-classes
                    linear-probe-mean linear-probe-std linear-probe-sweep linear-probe-weight
                    linear-probe-weight-decay probe-accuracy probe-dataset probe-dataset-classes
                    probe-dataset-features probe-dataset-indices probe-dataset-labels
                    probe-dataset? probe-split tensors->probe-dataset)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "probe-test"))

  (define accelerators
    (append (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (flat t)
    (~> t (to 'cpu) (reshape -1) tensor->list))

  (define (rows t)
    (for/list ([i (in-range (car (shape t)))])
      (flat (select t 0 i))))

  (define (seeded-ints n bound seed)
    (define prng (vector->pseudo-random-generator (vector seed 2 3 4 5 6)))
    (for/list ([_ (in-range n)]) (random bound prng)))

  (define (integer-tokens n t d)
    (define draws (seeded-ints (* n t d) 17 1))
    (tensor (for/list ([i (in-range n)])
              (for/list ([j (in-range t)])
                (for/list ([k (in-range d)])
                  (exact->inexact (- (list-ref draws (+ (* i t d) (* j d) k)) 8)))))
            #:dtype 'float32))

  (define (cycle-labels n classes)
    (tensor (for/list ([i (in-range n)]) (modulo i classes))))

  (define (token-dataset n t d)
    (tensor-dataset (integer-tokens n t d) (cycle-labels n 4)))

  (define (doubled x)
    (* x 2.0))

  (define (cls-token x)
    (select x 1 0))

  (define (mean-of xs)
    (/ (apply + xs) (length xs)))

  (define (pooled-oracle tokens pool)
    (for/list ([row (in-list (tensor->list-nested tokens))])
      (define cls (first row))
      (define rest-mean (apply map (lambda xs (mean-of xs)) (cdr row)))
      (match pool
        ['cls cls]
        ['mean rest-mean]
        ['cls+mean (append cls rest-mean)])))

  (define (tensor->list-nested t)
    (for/list ([i (in-range (car (shape t)))])
      (rows (select t 0 i))))

  (define pools '(cls mean cls+mean))

  (bounded-test "P1: features are float32 [N D'] beside the dataset's labels"
    (define images (token-dataset 10 5 3))
    (for ([pool (in-list (cons #f pools))])
      (define ds
        (probe-dataset (if pool doubled (lambda (x) (cls-token (doubled x)))) images
                       #:pool pool #:batch-size 4 #:device 'cpu))
      (define fs (probe-dataset-features ds))
      (check-equal? (shape fs) (list 10 (if (eq? pool 'cls+mean) 6 3)) (format "~a" pool))
      (check-eq? (dtype fs) 'float32)
      (check-eq? (device-type (device fs)) 'cpu)
      (check-eq? (dtype (probe-dataset-labels ds)) 'int64)
      (check-equal? (flat (probe-dataset-labels ds)) (flat (cycle-labels 10 4)))
      (check-equal? (probe-dataset-classes ds) 4)
      (check-false (probe-dataset-indices ds))
      (check-equal? (dataset-length ds) 10)
      (for ([i (in-range 10)])
        (define-values (row label) (dataset-ref ds i))
        (check-equal? (flat row) (flat (select fs 0 i)))
        (check-equal? (item label) (modulo i 4)))
      (define-values (batch labels) (dataset-batch ds '(2 3 4) default-collate))
      (check-equal? (flat batch) (flat (narrow fs 0 2 3)))
      (check-equal? (flat labels) '(2 3 0))
      (define seen
        (for/list ([(xb yb) (in-dataloader (dataloader ds #:batch-size 3))])
          (cons (rows xb) (flat yb))))
      (check-equal? (append-map car seen) (rows fs))
      (check-equal? (append-map cdr seen) (flat (cycle-labels 10 4)))))

  (bounded-test "P1: #:classes overrides 1 + the largest label"
    (define ds (probe-dataset doubled (token-dataset 6 5 3) #:classes 9))
    (check-equal? (probe-dataset-classes ds) 9))

  (bounded-test "P2: pooling of known tokens, whatever the batch size"
    (define tokens (integer-tokens 9 5 3))
    (define images (tensor-dataset tokens (cycle-labels 9 4)))
    (for ([pool (in-list pools)])
      (define want (pooled-oracle (doubled tokens) pool))
      (for ([batch-size (in-list '(1 4 1024))])
        (define ds (probe-dataset doubled images #:pool pool #:batch-size batch-size))
        (check-equal? (rows (probe-dataset-features ds)) want
                      (format "~a at batch size ~a" pool batch-size))))
    (define flat-images (tensor-dataset (cls-token tokens) (cycle-labels 9 4)))
    (check-equal? (rows (probe-dataset-features (probe-dataset doubled flat-images #:pool #f)))
                  (rows (doubled (cls-token tokens)))))

  (define (dropout-encoder)
    (manual-seed! 0)
    (Sequential (Flatten) (Linear 4 8) (Dropout 0.5)))

  (define (image-dataset n)
    (manual-seed! 1)
    (tensor-dataset (randn n 1 2 2) (cycle-labels n 4)))

  (define (modes layer)
    (cons (layer-mode layer) (append-map modes (children layer))))

  (bounded-test "P3: an eval pass under no grad that restores every mode"
    (define images (image-dataset 16))
    (define starts
      (list train!
            eval!
            (lambda (enc)
              (train! enc)
              (layer-set-mode! (last (children enc)) 'eval))))
    (for ([start! (in-list starts)])
      (define enc (dropout-encoder))
      (start! enc)
      (define before (modes enc))
      (define a (probe-dataset enc images #:pool #f #:batch-size 5))
      (check-equal? (modes enc) before)
      (define b (probe-dataset enc images #:pool #f #:batch-size 5))
      (check-equal? (modes enc) before)
      (check-equal? (flat (probe-dataset-features a)) (flat (probe-dataset-features b)))
      (check-false (requires-grad? (probe-dataset-features a)))
      (check-false (ormap has-grad? (parameters enc)))))

  (define (cpu-bytes)
    (for/sum ([(place bytes) (in-dict (native-memory-use))]
              #:when (eq? (device-type place) 'cpu))
      bytes))

  (bounded-test "P4: the pass keeps exactly its features and labels"
    (define images (image-dataset 24))
    (define enc (dropout-encoder))
    (reclaim-native-memory!)
    (define base (cpu-bytes))
    (define ds (probe-dataset enc images #:pool #f #:batch-size 5))
    (reclaim-native-memory!)
    (check-equal? (- (cpu-bytes) base) (+ (* 24 8 4) (* 24 8)))
    (check-true (probe-dataset? ds)))

  (define split-labels '(0 1 2 0 1 2 0 1 2 0 1 2 0 1 0 2 2 0))

  (define (labelled features)
    (tensors->probe-dataset features (tensor split-labels)))

  (define (split-indices ds #:per-class [n 3] #:seed [seed 1])
    (define-values (fit select) (probe-split ds #:select-per-class n #:seed seed))
    (list (flat (probe-dataset-indices fit)) (flat (probe-dataset-indices select)) fit select))

  (define (ascending? xs)
    (or (null? xs) (null? (cdr xs)) (and (< (car xs) (cadr xs)) (ascending? (cdr xs)))))

  (bounded-test "P5: a stratified split of the labels alone"
    (define features (randn 18 2))
    (match-define (list fit-ix select-ix fit select) (split-indices (labelled features)))
    (check-false (check-duplicates (append fit-ix select-ix)))
    (check-equal? (sort (append fit-ix select-ix) <) (range 18))
    (check-true (ascending? fit-ix))
    (check-true (ascending? select-ix))
    (for ([c (in-range 3)])
      (check-equal? (for/sum ([i (in-list select-ix)]) (if (= c (list-ref split-labels i)) 1 0))
                    3))
    (check-equal? (rows (probe-dataset-features select))
                  (rows (index-select features 0 (tensor select-ix))))
    (check-equal? (flat (probe-dataset-labels fit))
                  (for/list ([i (in-list fit-ix)]) (list-ref split-labels i)))
    (check-equal? (take-2 (split-indices (labelled features)))
                  (list fit-ix select-ix))
    (check-equal? (take-2 (split-indices (labelled (randn 18 2))))
                  (list fit-ix select-ix))
    (check-true (< 1 (length (remove-duplicates
                              (for/list ([seed (in-range 5)])
                                (cadr (split-indices (labelled features) #:seed seed))))))))

  (define (take-2 xs)
    (list (car xs) (cadr xs)))

  (bounded-test "P5: a class short of select-per-class raises"
    (check-exn exn:fail:contract?
               (lambda () (split-indices (labelled (randn 18 2)) #:per-class 6))))

  (bounded-test "P5: the split leaves the global stream alone"
    (define ds (labelled (randn 18 2)))
    (manual-seed! 7)
    (define without (flat (randn 4)))
    (manual-seed! 7)
    (split-indices ds)
    (check-equal? (flat (randn 4)) without))

  (define (column-means xss)
    (apply map (lambda xs (mean-of xs)) xss))

  (define (column-stds xss)
    (define n (length xss))
    (for/list ([xs (in-list (apply map list xss))] [mu (in-list (column-means xss))])
      (max 1e-6 (sqrt (/ (for/sum ([x (in-list xs)]) (* (- x mu) (- x mu))) (- n 1))))))

  (define (blob-features n seed)
    (manual-seed! seed)
    (define xs (randn n 3 #:dtype 'float64))
    (tensor (for/list ([row (in-list (rows xs))] [i (in-naturals)])
              (append (list (+ (car row) (* 2.0 (modulo i 3)))) (cdr row) '(4.0)))
            #:dtype 'float64))

  (define (blob-set n seed)
    (tensors->probe-dataset (blob-features n seed) (cycle-labels n 3)))

  (define (single ds)
    (tensors->probe-dataset (to-dtype (probe-dataset-features ds) 'float32)
                            (probe-dataset-labels ds)))

  (bounded-test "P7: the standardiser is the fit split's mean and unbiased std"
    (define fit (blob-set 40 2))
    (define x (rows (probe-dataset-features fit)))
    (define probe (linear-probe fit #:weight-decay 1e-3 #:max-iter 50))
    (for ([got (in-list (flat (linear-probe-mean probe)))] [want (in-list (column-means x))])
      (check-= got want 1e-6))
    (for ([got (in-list (flat (linear-probe-std probe)))] [want (in-list (column-stds x))])
      (check-= got want 1e-6))
    (define (statistics select)
      (define p (linear-probe fit select #:weight-decays '(1e-3 1e-2) #:max-iter 50))
      (list (flat (linear-probe-mean p)) (flat (linear-probe-std p))))
    (check-equal? (statistics (blob-set 30 3)) (statistics (blob-set 30 4))))

  (define (argmax-accuracy probe ds)
    (define x (probe-dataset-features ds))
    (define predicted
      (~> (- x (linear-probe-mean probe))
          (/ (linear-probe-std probe))
          (@ (linear-probe-weight probe))
          (+ (linear-probe-bias probe))
          (argmax 1)
          flat))
    (/ (for/sum ([p (in-list predicted)] [y (in-list (flat (probe-dataset-labels ds)))])
         (if (= p y) 1 0))
       (length predicted)))

  (bounded-test "P7: probe-accuracy is the argmax accuracy of x W + b on standardised x"
    (define fit (blob-set 40 2))
    (define select (blob-set 30 3))
    (define probe (linear-probe fit #:weight-decay 1e-2 #:max-iter 50))
    (check-= (probe-accuracy probe select) (argmax-accuracy probe select) 1e-6)
    (check-equal? (shape (linear-probe-weight probe)) '(4 3))
    (check-equal? (shape (linear-probe-bias probe)) '(3))
    (check-equal? (linear-probe-classes probe) 3))

  (define grid '(1e-4 1e-3 1e-2 1e-1 1.0))

  (define (first-best sweep)
    (define best (apply max (map cdr sweep)))
    (car (findf (lambda (entry) (= best (cdr entry))) sweep)))

  (define (separated n seed)
    (manual-seed! seed)
    (tensors->probe-dataset (+ (* 0.1 (randn n 2 #:dtype 'float64))
                               (tensor (for/list ([i (in-range n)])
                                         (if (even? i) '(5.0 0.0) '(-5.0 0.0)))
                                       #:dtype 'float64))
                            (cycle-labels n 2)))

  (bounded-test "P8: the sweep lists the grid in order and keeps the first best"
    (for ([fit+select (in-list (list (cons (blob-set 40 2) (blob-set 30 3))
                                     (cons (separated 40 5) (separated 20 6))))])
      (define probe (linear-probe (car fit+select) (cdr fit+select)
                                  #:weight-decays grid #:max-iter 50))
      (define sweep (linear-probe-sweep probe))
      (check-equal? (map car sweep) grid)
      (check-equal? (linear-probe-weight-decay probe) (first-best sweep)))
    (define separable (linear-probe (separated 40 5) (separated 20 6)
                                    #:weight-decays grid #:max-iter 50))
    (check-equal? (linear-probe-weight-decay separable) (car grid)))

  (bounded-test "P8: one #:weight-decay fits on the fit split alone"
    (define fit (blob-set 40 2))
    (define alone (linear-probe fit #:weight-decay 1e-2 #:max-iter 50))
    (check-false (linear-probe-sweep alone))
    (check-equal? (linear-probe-weight-decay alone) 1e-2)
    (define swept (linear-probe fit (blob-set 30 3) #:weight-decays '(1e-2) #:max-iter 50))
    (check-equal? (flat (linear-probe-weight alone)) (flat (linear-probe-weight swept)))
    (check-equal? (flat (linear-probe-bias alone)) (flat (linear-probe-bias swept))))

  (bounded-test "P15: probe-dataset blames its caller"
    (define images (image-dataset 4))
    (check-exn blames-this-test? (lambda () (probe-dataset doubled 5)))
    (check-exn blames-this-test? (lambda () (probe-dataset doubled images #:pool 'max)))
    (check-exn blames-this-test? (lambda () (probe-dataset doubled images #:batch-size 0))))

  (bounded-test "P15: tensors->probe-dataset blames its caller"
    (define y (tensor '(0 1 0 1)))
    (for ([bad (in-list (list (lambda () (tensors->probe-dataset (randn 4) y))
                              (lambda () (tensors->probe-dataset (zeros 4 2 #:dtype 'int64) y))
                              (lambda () (tensors->probe-dataset (randn 4 2) (randn 4)))
                              (lambda () (tensors->probe-dataset (randn 5 2) y))
                              (lambda () (tensors->probe-dataset (randn 4 2) (tensor '(0 -1 0 1))))
                              (lambda () (tensors->probe-dataset (randn 4 2) y #:classes 1))))]
          [i (in-naturals)])
      (check-exn blames-this-test? bad (format "case ~a" i)))
    (for ([dev (in-list accelerators)])
      (check-exn blames-this-test? (lambda () (tensors->probe-dataset (randn 4 2) (to y dev))))))

  (bounded-test "P15: probe-split and linear-probe blame their caller"
    (define fit (blob-set 40 2))
    (define select (blob-set 30 3))
    (define narrow-select (tensors->probe-dataset (randn 30 2 #:dtype 'float64) (cycle-labels 30 3)))
    (check-exn blames-this-test? (lambda () (probe-split fit #:select-per-class 0)))
    (for ([bad (in-list (list (lambda () (linear-probe fit narrow-select))
                              (lambda () (linear-probe fit))
                              (lambda () (linear-probe fit select #:weight-decay 1e-3))
                              (lambda () (linear-probe fit #:weight-decay 1e-3
                                                       #:weight-decays '(1e-3)))
                              (lambda () (linear-probe fit select #:weight-decays '()))
                              (lambda () (linear-probe (blob-set 1 2) #:weight-decay 1e-3))))]
          [i (in-naturals)])
      (check-exn blames-this-test? bad (format "case ~a" i)))
    (for ([dev (in-list accelerators)])
      (define moved (tensors->probe-dataset (to (to-dtype (blob-features 30 3) 'float32) dev)
                                            (to (cycle-labels 30 3) dev)))
      (check-exn blames-this-test? (lambda () (linear-probe (single fit) moved)))))

  (bounded-test "P15: probe-accuracy and knn-accuracy blame their caller"
    (define fit (blob-set 40 2))
    (define probe (linear-probe fit #:weight-decay 1e-3 #:max-iter 20))
    (define narrow-set (tensors->probe-dataset (randn 30 2 #:dtype 'float64) (cycle-labels 30 3)))
    (define wide-classes (tensors->probe-dataset (blob-features 30 3) (cycle-labels 30 5)))
    (define float32-set (tensors->probe-dataset (to-dtype (blob-features 30 3) 'float32)
                                                (cycle-labels 30 3)))
    (for ([bad (in-list (list (lambda () (probe-accuracy probe narrow-set))
                              (lambda () (probe-accuracy probe float32-set))
                              (lambda () (probe-accuracy probe wide-classes))
                              (lambda () (knn-accuracy fit narrow-set))
                              (lambda () (knn-accuracy fit wide-classes))
                              (lambda () (knn-accuracy fit #:k 0))
                              (lambda () (knn-accuracy fit #:k 40))
                              (lambda () (knn-accuracy fit (blob-set 30 3) #:k 41))
                              (lambda () (knn-accuracy fit #:temperature 0))))]
          [i (in-naturals)])
      (check-exn blames-this-test? bad (format "case ~a" i)))
    (for ([dev (in-list accelerators)])
      (define moved (tensors->probe-dataset (to (to-dtype (blob-features 30 3) 'float32) dev)
                                            (to (cycle-labels 30 3) dev)))
      (define single-fit (single fit))
      (define single-probe (linear-probe single-fit #:weight-decay 1e-3 #:max-iter 20))
      (check-exn blames-this-test? (lambda () (probe-accuracy single-probe moved)))
      (check-exn blames-this-test? (lambda () (knn-accuracy single-fit moved)))))

  (bounded-test "P15: a wrong-rank encoder output raises, naming probe-dataset"
    (define (raised-by-probe-dataset? e)
      (and (exn:fail:contract? e)
           (regexp-match? #rx"^probe-dataset: " (exn-message e))
           (not (blames-this-test? e))))
    (check-exn raised-by-probe-dataset?
               (lambda () (probe-dataset doubled (image-dataset 4) #:pool #f)))
    (check-exn raised-by-probe-dataset?
               (lambda () (probe-dataset cls-token (token-dataset 4 5 3) #:pool 'cls))))

  (for ([dev (in-list accelerators)])
    (bounded-test (format "P17: on ~a the probes match the CPU within 1/n" dev)
      (define (on-device ds)
        (tensors->probe-dataset (to (probe-dataset-features ds) dev)
                                (to (probe-dataset-labels ds) dev)))
      (define fit (single (blob-set 60 2)))
      (define select (single (blob-set 30 3)))
      (define-values (cpu-fit cpu-select) (probe-split (single (blob-set 60 7)) #:select-per-class 5))
      (define-values (dev-fit dev-select) (probe-split (on-device (single (blob-set 60 7)))
                                                       #:select-per-class 5))
      (check-equal? (flat (probe-dataset-indices dev-select))
                    (flat (probe-dataset-indices cpu-select)))
      (check-equal? (flat (probe-dataset-indices dev-fit)) (flat (probe-dataset-indices cpu-fit)))
      (define cpu-probe (linear-probe fit #:weight-decay 1e-2 #:max-iter 50))
      (define dev-probe (linear-probe (on-device fit) #:weight-decay 1e-2 #:max-iter 50))
      (check-= (probe-accuracy dev-probe (on-device select)) (probe-accuracy cpu-probe select)
               (/ 1.0 30))
      (check-= (knn-accuracy (on-device fit)) (knn-accuracy fit) (/ 1.0 60))
      (check-= (knn-accuracy (on-device fit) (on-device select)) (knn-accuracy fit select)
               (/ 1.0 30))
      (define images (image-dataset 12))
      (define enc (dropout-encoder))
      (check-equal? (device (probe-dataset-features (probe-dataset (to enc dev) images #:pool #f)))
                    (device (to (zeros 1) dev))))))
