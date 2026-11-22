#lang racket/base

(module+ test
  (require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr))
           (only-in racket/list argmax drop range take)
           rackunit
           (only-in syntax/parse/define define-syntax-parse-rule)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * + - / @ backward! grad-enabled? index-select manual-seed! randn
                    requires-grad! requires-grad? reshape select shape sum tensor tensor->list
                    topk with-no-grad zero-grad! zeros)
           (only-in "../train.rkt"
                    knn-accuracy linear-probe linear-probe-bias linear-probe-mean
                    linear-probe-std linear-probe-weight linear-probe-weight-decay probe-accuracy
                    probe-dataset-features probe-dataset-labels tensors->probe-dataset)
           (only-in "../train/private/copy.rkt" copy-of)
           (only-in "../train/private/lbfgs.rkt" lbfgs-minimize!)
           (only-in (submod "../train/probe.rkt" private) probe-objective)
           (only-in "private/gradcheck.rkt" gradcheck)
           (only-in "private/within.rkt" bounded-test within))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (rows t)
    (for/list ([i (in-range (car (shape t)))])
      (flat (select t 0 i))))

  (define (seeded-ints n bound seed)
    (define prng (vector->pseudo-random-generator (vector seed 7 11 13 17 19)))
    (for/list ([_ (in-range n)]) (random bound prng)))

  (define (one-hot labels classes)
    (tensor (for/list ([y (in-list labels)])
              (for/list ([c (in-range classes)]) (if (= c y) 1.0 0.0)))
            #:dtype 'float32))

  (define-syntax-parse-rule (slow-test name:expr body:expr ...+)
    (test-case name (within 900 (lambda () body ...))))

  (define (separable n seed)
    (define labels (seeded-ints n 10 seed))
    (manual-seed! seed)
    (tensors->probe-dataset (+ (* 8.0 (one-hot labels 10)) (* 0.1 (randn n 10)))
                            (tensor labels)))

  (slow-test "P9: separable classes are scored perfectly by both probes"
    (define fit (separable 2000 1))
    (define select (separable 500 2))
    (define test (separable 500 3))
    (define probe (linear-probe fit select))
    (for ([ds (in-list (list fit select test))])
      (check-= (probe-accuracy probe ds) 1.0 0.0))
    (check-= (knn-accuracy fit) 1.0 0.0)
    (check-= (knn-accuracy fit select) 1.0 0.0)
    (check-= (knn-accuracy fit test) 1.0 0.0))

  (define (choose n k)
    (for/fold ([acc 1]) ([i (in-range k)])
      (/ (* acc (- n i)) (+ i 1))))

  (define (binomial-region n p alpha)
    (define pmf (for/list ([k (in-range (+ n 1))])
                  (* (choose n k) (expt p k) (expt (- 1 p) (- n k)))))
    (define half (/ alpha 2))
    (define lo
      (let loop ([k 0] [below 0])
        (define through (+ below (list-ref pmf k)))
        (if (> through half) k (loop (+ k 1) through))))
    (define hi
      (let loop ([k n] [above 0])
        (define through (+ above (list-ref pmf k)))
        (if (> through half) k (loop (- k 1) through))))
    (values lo hi))

  (define (noise n seed)
    (manual-seed! seed)
    (tensors->probe-dataset (randn n 32) (tensor (seeded-ints n 10 seed))))

  (slow-test "P10: shuffled labels score at chance"
    (define-values (lo hi) (binomial-region 2000 1/10 1/1000))
    (define fit (noise 2000 11))
    (define select (noise 2000 12))
    (define test (noise 2000 13))
    (define probe (linear-probe fit select))
    (for ([accuracy (in-list (list (probe-accuracy probe test)
                                   (knn-accuracy fit test)
                                   (knn-accuracy fit)))]
          [name (in-list '(linear knn knn-leave-one-out))])
      (define hits (~> (* accuracy 2000) round inexact->exact))
      (check-true (<= lo hits hi) (format "~a: ~a hits outside [~a, ~a]" name hits lo hi))))

  (define (column-means xss)
    (for/list ([xs (in-list (apply map list xss))])
      (/ (apply + xs) (length xs))))

  (define (column-stds xss)
    (define n (length xss))
    (for/list ([xs (in-list (apply map list xss))] [mu (in-list (column-means xss))])
      (max 1e-6 (sqrt (/ (for/sum ([x (in-list xs)]) (* (- x mu) (- x mu))) (- n 1))))))

  (define (standardised xss mus sds)
    (for/list ([xs (in-list xss)])
      (map (lambda (x mu sd) (/ (- x mu) sd)) xs mus sds)))

  (define (normalised xss)
    (for/list ([xs (in-list xss)])
      (define norm (max 1e-12 (sqrt (for/sum ([x (in-list xs)]) (* x x)))))
      (map (lambda (x) (/ x norm)) xs)))

  (define (dot xs ys)
    (for/sum ([x (in-list xs)] [y (in-list ys)]) (* x y)))

  (define (similarities fit-rows query-rows standardize?)
    (define mus (column-means fit-rows))
    (define sds (column-stds fit-rows))
    (define (prepared xss)
      (normalised (if standardize? (standardised xss mus sds) xss)))
    (define f (prepared fit-rows))
    (for/list ([q (in-list (prepared query-rows))])
      (for/list ([r (in-list f)]) (dot q r))))

  (define (neighbours sims self k)
    (define ranked
      (sort (for/list ([s (in-list sims)] [j (in-naturals)] #:unless (eqv? j self)) (cons s j))
            > #:key car))
    (take ranked k))

  (define (knn-oracle fit query k temperature standardize?)
    (define fit-rows (rows (probe-dataset-features fit)))
    (define fit-labels (flat (probe-dataset-labels fit)))
    (define query-rows (if query (rows (probe-dataset-features query)) fit-rows))
    (define query-labels (if query (flat (probe-dataset-labels query)) fit-labels))
    (define classes (+ 1 (apply max fit-labels)))
    (define hits
      (for/sum ([sims (in-list (similarities fit-rows query-rows standardize?))]
                [y (in-list query-labels)]
                [i (in-naturals)])
        (define votes (make-vector classes 0.0))
        (for ([entry (in-list (neighbours sims (and (not query) i) k))])
          (define c (list-ref fit-labels (cdr entry)))
          (vector-set! votes c (+ (vector-ref votes c) (~> (car entry) (/ temperature) exp))))
        (define predicted (argmax (lambda (c) (vector-ref votes c)) (range classes)))
        (if (= predicted y) 1 0)))
    (/ hits (length query-labels)))

  (define (overlapping n seed #:width [width 6] #:classes [classes 3])
    (define labels (seeded-ints n classes seed))
    (manual-seed! seed)
    (define centres (* 1.2 (randn classes width #:dtype 'float64)))
    (tensors->probe-dataset (+ (index-select centres 0 (tensor labels))
                               (randn n width #:dtype 'float64))
                            (tensor labels)))

  (bounded-test "P12: knn-accuracy is the plain weighted vote"
    (define fit (overlapping 60 21 #:width 4))
    (define query (overlapping 30 22 #:width 4))
    (for* ([k (in-list '(1 5 20))]
           [standardize? (in-list '(#t #f))]
           [q (in-list (list #f query))])
      (check-= (knn-accuracy fit q #:k k #:temperature 0.07 #:standardize? standardize?)
               (knn-oracle fit q k 0.07 standardize?)
               1e-6
               (format "k ~a, standardize? ~a, query? ~a" k standardize? (and q #t)))))

  (define (logits probe ds)
    (~> (- (probe-dataset-features ds) (linear-probe-mean probe))
        (/ (linear-probe-std probe))
        (@ (linear-probe-weight probe))
        (+ (linear-probe-bias probe))))

  (define (smallest-margin probe ds)
    (define-values (top _) (topk (logits probe ds) 2 #:dim 1))
    (for/fold ([m +inf.0]) ([pair (in-list (rows top))])
      (min m (- (car pair) (cadr pair)))))

  (define (smallest-gap fit query)
    (define fit-rows (rows (probe-dataset-features fit)))
    (define query-rows (if query (rows (probe-dataset-features query)) fit-rows))
    (for/fold ([m +inf.0]) ([sims (in-list (similarities fit-rows query-rows #t))] [i (in-naturals)])
      (define ranked (neighbours sims (and (not query) i) 21))
      (min m (- (car (list-ref ranked 19)) (car (list-ref ranked 20))))))

  (define (permuted ds perm)
    (tensors->probe-dataset (index-select (probe-dataset-features ds) 1 (tensor perm))
                            (probe-dataset-labels ds)))

  (slow-test "P11: permuting feature columns changes no figure"
    (define perm '(3 0 5 1 4 2))
    (define fit (overlapping 300 37))
    (define select (overlapping 90 32))
    (define test (overlapping 150 33))
    (define probe (linear-probe fit select))
    (for ([ds (in-list (list fit select test))])
      (check-true (< 1e-3 (smallest-margin probe ds)) "logit margin precondition"))
    (for ([q (in-list (list #f select test))])
      (check-true (< 1e-5 (smallest-gap fit q)) "similarity gap precondition"))
    (define-values (fit* select* test*) (values (permuted fit perm) (permuted select perm)
                                                (permuted test perm)))
    (define probe* (linear-probe fit* select*))
    (check-equal? (linear-probe-weight-decay probe*) (linear-probe-weight-decay probe))
    (for ([ds (in-list (list fit select test))] [ds* (in-list (list fit* select* test*))])
      (check-equal? (probe-accuracy probe* ds*) (probe-accuracy probe ds)))
    (check-equal? (knn-accuracy fit*) (knn-accuracy fit))
    (check-equal? (knn-accuracy fit* select*) (knn-accuracy fit select))
    (check-equal? (knn-accuracy fit* test*) (knn-accuracy fit test))
    (define w (rows (linear-probe-weight probe)))
    (for ([row* (in-list (rows (linear-probe-weight probe*)))] [p (in-list perm)])
      (for ([a (in-list row*)] [b (in-list (list-ref w p))])
        (check-= a b 1e-5))))

  (bounded-test "P13: the objective's gradient matches central differences"
    (manual-seed! 41)
    (define x (randn 8 3 #:dtype 'float64))
    (define y (tensor '(0 1 2 3 0 1 2 3)))
    (check-true
     (gradcheck (lambda (weight bias) (probe-objective x y weight bias 0.3))
                (list (randn 3 4 #:dtype 'float64) (randn 4 #:dtype 'float64)))))

  (define spd-m
    (for/list ([i (in-range 6)])
      (for/list ([v (in-list (drop (seeded-ints 42 7 51) (* i 6)))] [_ (in-range 6)])
        (- v 3))))

  (define spd-a
    (for/list ([i (in-range 6)])
      (for/list ([j (in-range 6)])
        (+ (for/sum ([row (in-list spd-m)]) (* (list-ref row i) (list-ref row j)))
           (if (= i j) 6 0)))))

  (define spd-b '(3 -1 4 -1 5 -9))

  (define (solve a b)
    (define n (length b))
    (define m (for/vector ([row (in-list a)] [v (in-list b)]) (list->vector (append row (list v)))))
    (for ([col (in-range n)])
      (define pivot (vector-ref (vector-ref m col) col))
      (for ([r (in-range n)] #:unless (= r col))
        (define factor (~> (vector-ref m r) (vector-ref col) (/ pivot)))
        (for ([c (in-range (+ n 1))])
          (vector-set! (vector-ref m r) c
                       (- (vector-ref (vector-ref m r) c)
                          (* factor (vector-ref (vector-ref m col) c)))))))
    (for/list ([r (in-range n)])
      (/ (vector-ref (vector-ref m r) n) (vector-ref (vector-ref m r) r))))

  (define (quadratic-closure x evaluations)
    (define a (tensor spd-a #:dtype 'float64))
    (define b (tensor spd-b #:dtype 'float64))
    (lambda ()
      (set-box! evaluations (+ 1 (unbox evaluations)))
      (zero-grad! x)
      (define loss (- (* 0.5 (sum (* x (@ a x)))) (sum (* b x))))
      (backward! loss)
      loss))

  (bounded-test "P14: L-BFGS reaches the minimum of an SPD quadratic"
    (define exact (solve spd-a spd-b))
    (for ([line-search (in-list '(#f strong-wolfe))])
      (define x (requires-grad! (zeros 6 #:dtype 'float64)))
      (lbfgs-minimize! (list x) (quadratic-closure x (box 0))
                       #:max-iter 100 #:tolerance-change 1e-12 #:line-search line-search)
      (for ([got (in-list (flat x))] [want (in-list exact)])
        (check-= got (exact->inexact want) 1e-6 (format "~a" line-search)))))

  (bounded-test "P14: a gradient already below tolerance costs one evaluation and no step"
    (define start (map exact->inexact (solve spd-a spd-b)))
    (define x (requires-grad! (tensor start #:dtype 'float64)))
    (define evaluations (box 0))
    (lbfgs-minimize! (list x) (quadratic-closure x evaluations) #:line-search 'strong-wolfe)
    (check-equal? (unbox evaluations) 1)
    (check-equal? (flat x) (flat (tensor start #:dtype 'float64))))

  (bounded-test "P14: the closure runs with grad on, under a caller's with-no-grad"
    (define x (requires-grad! (zeros 6 #:dtype 'float64)))
    (define modes (box '()))
    (define quadratic (quadratic-closure x (box 0)))
    (define (closure)
      (set-box! modes (cons (grad-enabled?) (unbox modes)))
      (quadratic))
    (with-no-grad
      (lbfgs-minimize! (list x) closure #:max-iter 5 #:line-search 'strong-wolfe)
      (check-false (grad-enabled?)))
    (check-true (andmap values (unbox modes)))
    (check-true (< 1 (length (unbox modes)))))

  (bounded-test "P14: a closure that raises leaves grad mode as the caller had it"
    (define x (requires-grad! (zeros 6 #:dtype 'float64)))
    (define quadratic (quadratic-closure x (box 0)))
    (define calls (box 0))
    (define (closure)
      (set-box! calls (+ 1 (unbox calls)))
      (when (< 2 (unbox calls))
        (raise 'closure-failed))
      (quadratic))
    (with-no-grad
      (check-equal? (with-handlers ([symbol? values])
                      (lbfgs-minimize! (list x) closure #:max-iter 5 #:line-search 'strong-wolfe))
                    'closure-failed)
      (check-false (grad-enabled?)))
    (check-true (grad-enabled?)))

  (bounded-test "P14: the saved parameters do not require grad"
    (define x (requires-grad! (tensor '(1.0 2.0) #:dtype 'float64)))
    (define saved (copy-of x))
    (check-false (requires-grad? saved))
    (check-equal? (flat saved) '(1.0 2.0)))

  (define (rosenbrock-closure x evaluations)
    (lambda ()
      (set-box! evaluations (+ 1 (unbox evaluations)))
      (zero-grad! x)
      (define x0 (select x 0 0))
      (define x1 (select x 0 1))
      (define loss (+ (* (- 1.0 x0) (- 1.0 x0))
                      (* 100.0 (* (- x1 (* x0 x0)) (- x1 (* x0 x0))))))
      (backward! loss)
      loss))

  (bounded-test "P14: L-BFGS never exceeds max-iter"
    (for ([max-iter (in-list '(1 3 7))])
      (define x (requires-grad! (tensor '(-1.5 2.0) #:dtype 'float64)))
      (define evaluations (box 0))
      (lbfgs-minimize! (list x) (rosenbrock-closure x evaluations)
                       #:lr 1e-3 #:max-iter max-iter #:max-eval 10000)
      (check-equal? (unbox evaluations) max-iter
                    (format "one evaluation per iteration at max-iter ~a" max-iter)))))
