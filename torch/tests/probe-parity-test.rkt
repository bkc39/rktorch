#lang racket/base

(module+ test
  (require (only-in json json-null)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + - / @ abs argmax backward! dtype eq eye item masked-fill! max min narrow
                    ones pow requires-grad! reshape select shape sqrt sum take tensor
                    tensor->list to-dtype topk transpose zero-grad! zeros)
           (only-in "../train.rkt"
                    knn-accuracy linear-probe linear-probe-bias linear-probe-mean linear-probe-std
                    linear-probe-weight linear-probe-weight-decay probe-accuracy
                    probe-dataset-features probe-dataset-indices probe-dataset-labels probe-split
                    tensors->probe-dataset)
           (only-in "../train/private/lbfgs.rkt" lbfgs-minimize!)
           (only-in (submod "../train/probe.rkt" private) knn-predictions probe-objective)
           (only-in "../vision/cifar10.rkt" cifar10-cached? load-cifar10)
           (only-in "private/python-env.rkt" python-check python-module-available? unpack))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (check-close got want tolerance label)
    (define worst (~> (- got want) abs max item))
    (check-true (<= worst tolerance) (format "~a: off by ~a" label worst)))

  (define (labelled j features-key labels-key dtype)
    (tensors->probe-dataset (~> (hash-ref j features-key) (unpack 'float32) (to-dtype dtype))
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
    (~> (- (select top 1 0) (select top 1 1)) min item))

  (define (row-normalised x)
    (/ x (sqrt (@ (* x x) (ones (cadr (shape x)) 1 #:dtype (dtype x))))))

  (define (split-gaps fit-x fit-labels query-x)
    (define f (row-normalised fit-x))
    (define sims (@ (row-normalised (or query-x fit-x)) (transpose f 0 1)))
    (unless query-x
      (masked-fill! sims (~> (shape fit-x) car (eye #:dtype 'int64) (eq 1)) -inf.0))
    (define-values (top idx) (topk sims 21 #:dim 1))
    (define labels (take fit-labels idx))
    (for/list ([gap (in-list (flat (- (select top 1 19) (select top 1 20))))]
               [a (in-list (flat (select labels 1 19)))]
               [b (in-list (flat (select labels 1 20)))]
               #:unless (= a b))
      gap))

  (define (hits-ratio predicted labels)
    (/ (for/sum ([p (in-list (flat predicted))] [y (in-list (flat labels))]) (if (= p y) 1 0))
       (length (flat labels))))

  (define (first-best accuracies)
    (define best (apply max accuracies))
    (for/first ([a (in-list accuracies)] [i (in-naturals)] #:when (= a best)) i))

  (define (check-split j)
    (define labels256 (unpack (hash-ref j 'labels256) 'int64))
    (define-values (fit select)
      (~> (zeros 256 1) (tensors->probe-dataset labels256) (probe-split #:select-per-class 2)))
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
         (~> (shape labels) car (zeros 1) (tensors->probe-dataset labels) probe-split))
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
    (define sweep (~> (hash-ref linear 'sweep) hash-values (sort >)))
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
    (define (std-of k) (~> (hash-ref j 'std) (list-ref k) (unpack 'float32)))
    (define knn (hash-ref suite 'knn))
    (for ([query (in-list (list #f select test))]
          [query-x (in-list (list #f (std-of 1) (std-of 2)))]
          [key (in-list '(train val test))])
      (define gaps (split-gaps (std-of 0) (probe-dataset-labels fit) query-x))
      (check-equal? (for/sum ([gap (in-list gaps)] #:unless (< 1e-5 gap)) 1) 0
                    (format (string-append "P16c: similarity gap precondition, ~a, on the ~a rows"
                                           " whose 20th and 21st neighbours differ in label")
                            key (length gaps)))
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
                     (* 1e-3 (~> (abs w) max item))
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
      (check-equal? (flat (knn-predictions fit query #:k 20 #:temperature 0.07 #:standardize? #t))
                    (flat (unpack (hash-ref knn key) 'int64))
                    (format "P16e: knn ~a predictions" key))))

  (define (traced params objective minimize!)
    (define losses '())
    (define (closure)
      (for-each zero-grad! params)
      (define loss (objective))
      (backward! loss)
      (set! losses (cons (item loss) losses))
      loss)
    (minimize! params closure)
    (reverse losses))

  (define (rosenbrock-trace start dtype)
    (define x (requires-grad! (tensor start #:dtype dtype)))
    (define (rosenbrock)
      (define x0 (select x 0 0))
      (define x1 (select x 0 1))
      (+ (pow (- 1 x0) 2) (* 100 (pow (- x1 (pow x0 2)) 2))))
    (values (traced (list x) rosenbrock
                    (lambda (params closure)
                      (lbfgs-minimize! params closure #:max-iter 100 #:history-size 20
                                       #:line-search 'strong-wolfe)))
            (flat x)))

  (define (check-rosenbrock j)
    (define want (hash-ref j 'lbfgs_rosenbrock))
    (define-values (losses x) (rosenbrock-trace '(-1.5 2.0) 'float64))
    (check-equal? (length losses) (length (hash-ref want 'losses)) "P18: closure evaluations")
    (for ([got (in-list losses)] [expected (in-list (hash-ref want 'losses))] [i (in-naturals)])
      (check-= got expected 1e-10 (format "P18: loss at evaluation ~a" i)))
    (for ([got (in-list x)] [expected (in-list (hash-ref want 'x))])
      (check-= got expected 1e-10 "P18: final x"))
    (for ([run (in-list (hash-ref j 'lbfgs_rosenbrock32))])
      (define start (hash-ref run 'start))
      (define-values (losses x) (rosenbrock-trace start 'float32))
      (check-equal? losses (hash-ref run 'losses)
                    (format "P18: float32 losses from ~a, evaluation for evaluation" start))
      (check-equal? x (hash-ref run 'x) (format "P18: float32 x from ~a" start))))

  (define (check-probe-trace j fit)
    (define want (hash-ref j 'lbfgs_probe))
    (define n (hash-ref want 'n))
    (define x (~> (hash-ref j 'std) car (unpack 'float32) (narrow 0 0 n) (* (hash-ref want 'scale))))
    (define y (narrow (probe-dataset-labels fit) 0 0 n))
    (define w (requires-grad! (zeros 24 10)))
    (define b (requires-grad! (zeros 10)))
    (check-equal? (traced (list w b)
                          (lambda () (probe-objective x y w b (hash-ref want 'weight_decay)))
                          (lambda (params closure)
                            (lbfgs-minimize! params closure #:lr 1.0 #:max-iter 1000
                                             #:tolerance-grad 1e-7 #:tolerance-change 1e-10
                                             #:history-size 20 #:line-search 'strong-wolfe)))
                  (hash-ref want 'losses)
                  "P14: float32 probe objective losses, evaluation for evaluation"))

  (define (check-quadratic j)
    (define want (hash-ref j 'lbfgs_quadratic))
    (define a (tensor (hash-ref want 'a) #:dtype 'float64))
    (define b (tensor (hash-ref want 'b) #:dtype 'float64))
    (for ([line-search (in-list '(#f strong-wolfe))] [key (in-list '(none strong_wolfe))])
      (define x (requires-grad! (zeros 6 #:dtype 'float64)))
      (define losses
        (traced (list x)
                (lambda () (- (* 0.5 (sum (* x (@ a x)))) (sum (* b x))))
                (lambda (params closure)
                  (lbfgs-minimize! params closure #:max-iter 100 #:tolerance-change 1e-12
                                   #:line-search line-search))))
      (define run (hash-ref want key))
      (check-equal? losses (hash-ref run 'losses)
                    (format "P14: ~a losses, evaluation for evaluation" key))
      (check-equal? (flat x) (hash-ref run 'x) (format "P14: ~a final x" key))))

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
     (check-rosenbrock j)
     (check-quadratic j)
     (check-probe-trace j (car (split 'float32)))]))
