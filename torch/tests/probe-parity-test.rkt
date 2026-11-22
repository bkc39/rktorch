#lang racket/base

(module+ test
  (require (only-in json json-null)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + - / @ abs argmax backward! dtype eq eye item masked-fill! max min ones
                    requires-grad! reshape select shape sqrt sum tensor tensor->list to-dtype
                    topk transpose zero-grad! zeros)
           (only-in "../train.rkt"
                    knn-accuracy linear-probe linear-probe-bias linear-probe-mean linear-probe-std
                    linear-probe-weight linear-probe-weight-decay probe-accuracy
                    probe-dataset-features probe-dataset-indices probe-dataset-labels probe-split
                    tensors->probe-dataset)
           (only-in "../train/private/lbfgs.rkt" lbfgs-minimize!)
           (only-in "../vision/cifar10.rkt" cifar10-cached? load-cifar10)
           (only-in "private/python-env.rkt" python-check python-module-available? unpack))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (check-close got want tolerance label)
    (define worst (item (max (abs (- got want)))))
    (check-true (<= worst tolerance) (format "~a: off by ~a" label worst)))

  (define (labelled j features-key labels-key dtype)
    (tensors->probe-dataset (to-dtype (unpack (hash-ref j features-key) 'float32) dtype)
                            (unpack (hash-ref j labels-key) 'int64)))

  (define (standardised probe ds)
    (~> (- (probe-dataset-features ds) (linear-probe-mean probe))
        (/ (linear-probe-std probe))))

  (define (logits probe ds)
    (~> (standardised probe ds)
        (@ (linear-probe-weight probe))
        (+ (linear-probe-bias probe))))

  (define (smallest-margin probe ds)
    (define-values (top _) (topk (logits probe ds) 2 #:dim 1))
    (item (min (- (select top 1 0) (select top 1 1)))))

  (define (row-normalised x)
    (/ x (sqrt (@ (* x x) (ones (cadr (shape x)) 1 #:dtype (dtype x))))))

  (define (smallest-gap fit-x query-x)
    (define f (row-normalised fit-x))
    (define sims (@ (row-normalised (or query-x fit-x)) (transpose f 0 1)))
    (unless query-x
      (masked-fill! sims (eq (eye (car (shape fit-x)) #:dtype 'int64) 1) -inf.0))
    (define-values (top _) (topk sims 21 #:dim 1))
    (item (min (- (select top 1 19) (select top 1 20)))))

  (define (hits-ratio predicted labels)
    (/ (for/sum ([p (in-list (flat predicted))] [y (in-list (flat labels))]) (if (= p y) 1 0))
       (length (flat labels))))

  (define (first-best accuracies)
    (define best (apply max accuracies))
    (for/first ([a (in-list accuracies)] [i (in-naturals)] #:when (= a best)) i))

  (define (check-split j)
    (define labels256 (unpack (hash-ref j 'labels256) 'int64))
    (define-values (fit select)
      (probe-split (tensors->probe-dataset (zeros 256 1) labels256) #:select-per-class 2))
    (check-equal? (flat (probe-dataset-indices fit)) (flat (unpack (car (hash-ref j 'split)) 'int64))
                  "P6: fit indices of the fixture")
    (check-equal? (flat (probe-dataset-indices select))
                  (flat (unpack (cadr (hash-ref j 'split)) 'int64))
                  "P6: select indices of the fixture")
    (define full (hash-ref j 'full_split))
    (cond
      [(or (eq? full (json-null)) (not (cifar10-cached?)))
       (displayln "[probe-parity-test] P6 full split skipped: CIFAR-10 archive not cached")]
      [else
       (define-values (_images labels) (load-cifar10 'train))
       (define-values (fit* select*)
         (probe-split (tensors->probe-dataset (zeros (car (shape labels)) 1) labels)))
       (check-equal? (flat (probe-dataset-indices fit*)) (flat (unpack (car full) 'int64))
                     "P6: CIFAR-10 fit indices")
       (check-equal? (flat (probe-dataset-indices select*)) (flat (unpack (cadr full) 'int64))
                     "P6: CIFAR-10 select indices")]))

  (define (check-standardiser j fit select test)
    (define probe (linear-probe fit #:weight-decay 1e-4))
    (for ([ds (in-list (list fit select test))]
          [want (in-list (hash-ref j 'std))]
          [name (in-list '(fit select test))])
      (check-close (standardised probe ds) (unpack want 'float32) 1e-6
                   (format "P16a: standardised ~a" name))))

  (define (check-fit-linear j fit)
    (for ([wd (in-list '(1e-4 1e-2))] [key (in-list '(|0.0001| |0.01|))])
      (define probe (linear-probe fit #:weight-decay wd))
      (define want (hash-ref (hash-ref j 'fit_linear) key))
      (check-close (linear-probe-weight probe) (unpack (hash-ref want 'weight) 'float32) 1e-4
                   (format "P16b: weight at ~a" wd))
      (check-close (linear-probe-bias probe) (unpack (hash-ref want 'bias) 'float32) 1e-4
                   (format "P16b: bias at ~a" wd))))

  (define (check-suite j fit select test)
    (define suite (hash-ref j 'suite))
    (define linear (hash-ref suite 'linear))
    (define sweep (sort (hash-values (hash-ref linear 'sweep)) >))
    (check-true (> (- (car sweep) (cadr sweep)) (/ 1 600))
                "P16c: the twin's best weight decay leads the runner-up by more than 1/n")
    (define probe (linear-probe fit select))
    (check-equal? (linear-probe-weight-decay probe) (hash-ref linear 'weight_decay)
                  "P16c: chosen weight decay")
    (for ([ds (in-list (list fit select test))]
          [key (in-list '(train val test))]
          [n (in-list '(3000 600 1000))])
      (check-= (probe-accuracy probe ds) (hash-ref linear key) (/ 1.0 n)
               (format "P16c: linear ~a" key)))
    (define (std-of k) (unpack (list-ref (hash-ref j 'std) k) 'float32))
    (define knn (hash-ref suite 'knn))
    (for ([query (in-list (list #f select test))]
          [query-x (in-list (list #f (std-of 1) (std-of 2)))]
          [key (in-list '(train val test))])
      (check-true (< 1e-5 (smallest-gap (std-of 0) query-x))
                  (format "P16c: similarity gap precondition, ~a" key))
      (check-= (knn-accuracy fit query) (hash-ref knn key) 1e-6 (format "P16c: knn ~a" key))))

  (define (check-sklearn j fit select test)
    (define learn (hash-ref j 'sklearn))
    (check-equal? (hash-ref learn 'version) "1.8.0" "P16f: sklearn version of record")
    (define logistic (hash-ref learn 'logistic))
    (define select-accuracies
      (for/list ([entry (in-list logistic)])
        (define wd (hash-ref entry 'weight_decay))
        (define probe (linear-probe fit #:weight-decay wd))
        (define w (linear-probe-weight probe))
        (check-close w (unpack (hash-ref entry 'coef) 'float64)
                     (* 1e-3 (item (max (abs w))))
                     (format "P16d: weights at ~a" wd))
        (for ([ds (in-list (list select test))] [key (in-list '(select test))])
          (check-true (< 1e-3 (smallest-margin probe ds))
                      (format "P16d: logit margin precondition, ~a at ~a" key wd))
          (check-equal? (flat (argmax (logits probe ds) 1)) (flat (unpack (hash-ref entry key) 'int64))
                        (format "P16d: ~a predictions at ~a" key wd)))
        (hits-ratio (unpack (hash-ref entry 'select) 'int64) (probe-dataset-labels select))))
    (check-equal? (linear-probe-weight-decay (linear-probe fit select))
                  (hash-ref (list-ref logistic (first-best select-accuracies)) 'weight_decay)
                  "P16d: the sweep's choice")
    (define knn (hash-ref learn 'knn))
    (for ([query (in-list (list select test #f))]
          [key (in-list '(select test loo))])
      (define labels (probe-dataset-labels (or query fit)))
      (check-= (knn-accuracy fit query) (hash-ref-ratio knn key labels) 1e-6
               (format "P16e: knn ~a" key))))

  (define (hash-ref-ratio knn key labels)
    (exact->inexact (hits-ratio (unpack (hash-ref knn key) 'int64) labels)))

  (define (check-rosenbrock j)
    (define want (hash-ref j 'lbfgs_rosenbrock))
    (define x (requires-grad! (tensor '(-1.5 2.0) #:dtype 'float64)))
    (define losses '())
    (define (closure)
      (zero-grad! x)
      (define x0 (select x 0 0))
      (define x1 (select x 0 1))
      (define loss (+ (* (- 1.0 x0) (- 1.0 x0))
                      (* 100.0 (* (- x1 (* x0 x0)) (- x1 (* x0 x0))))))
      (backward! loss)
      (set! losses (cons (item loss) losses))
      loss)
    (lbfgs-minimize! (list x) closure #:max-iter 100 #:history-size 20 #:line-search 'strong-wolfe)
    (check-equal? (length losses) (length (hash-ref want 'losses)) "P18: closure evaluations")
    (for ([got (in-list (reverse losses))] [expected (in-list (hash-ref want 'losses))] [i (in-naturals)])
      (check-= got expected 1e-10 (format "P18: loss at evaluation ~a" i)))
    (for ([got (in-list (flat x))] [expected (in-list (hash-ref want 'x))])
      (check-= got expected 1e-10 "P18: final x")))

  (cond
    [(not (python-module-available? "sklearn"))
     (displayln "[probe-parity-test] skipped: python3 `sklearn` not available")]
    [else
     (define j (python-check "probe_twin.py"))
     (define (split dtype)
       (list (labelled j 'fit 'fit_labels dtype)
             (labelled j 'select 'select_labels dtype)
             (labelled j 'test 'test_labels dtype)))
     (check-split j)
     (apply check-standardiser j (split 'float32))
     (check-fit-linear j (car (split 'float32)))
     (apply check-suite j (split 'float32))
     (apply check-sklearn j (split 'float64))
     (check-rosenbrock j)]))
