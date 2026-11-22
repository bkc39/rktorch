#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/match match match-define)
         (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  * - abs cat copy! dot dtype grad-enabled? item max maybe-grad narrow neg numel
                  reshape shape sub! sum with-no-grad zeros-like)
         (only-in "../../foreign/error.rkt" check-ok)
         (only-in "../../foreign/raw/autograd.rkt" tr-set-grad-enabled/raw)
         (only-in "copy.rkt" copy-of))

(provide lbfgs-minimize!)

;; torch's 0-d step tensors: arithmetic touching one rounds to the parameters' dtype.
(struct scalar (value round))

(define (scalar-of t)
  (scalar (item t) (if (eq? (dtype t) 'float32) flsingle values)))

(define (value-of v)
  (if (scalar? v) (scalar-value v) v))

(define (rounding a b)
  (cond
    [(scalar? a) (scalar-round a)]
    [(scalar? b) (scalar-round b)]
    [else #f]))

(define (operand v round)
  (if (scalar? v) (scalar-value v) (round (exact->inexact v))))

(define ((arithmetic op) a b)
  (define round (rounding a b))
  (if round
      (~> (op (operand a round) (operand b round)) round (scalar round))
      (op a b)))

(define ((comparison op) a b)
  (define round (rounding a b))
  (if round (op (operand a round) (operand b round)) (op a b)))

(define ((unary op) v)
  (if (scalar? v)
      (scalar ((scalar-round v) (op (scalar-value v))) (scalar-round v))
      (op v)))

(define s+ (arithmetic +))
(define s- (arithmetic -))
(define s* (arithmetic *))
(define s< (comparison <))
(define s<= (comparison <=))
(define s> (comparison >))
(define s>= (comparison >=))
(define s-abs (unary abs))
(define s-sqrt (unary sqrt))
(define s-reciprocal (unary (lambda (v) (/ 1.0 v))))

(define s/
  (let ([divide (arithmetic /)])
    (lambda (a b)
      (if (and (scalar? b) (not (scalar? a)))
          (s* (s-reciprocal b) a)
          (divide a b)))))

(define (py-max a b)
  (if (s> b a) b a))

(define (py-min a b)
  (if (s< b a) b a))

(define (abs-max v)
  (~> v abs max scalar-of))

(define (cubic-interpolate x1 f1 g1 x2 f2 g2 [bounds #f])
  (match-define (cons lo hi) (or bounds (if (s<= x1 x2) (cons x1 x2) (cons x2 x1))))
  (define d1 (s- (s+ g1 g2) (s/ (s* 3 (s- f1 f2)) (s- x1 x2))))
  (define d2-square (s- (s* d1 d1) (s* g1 g2)))
  (cond
    [(s>= d2-square 0)
     (define d2 (s-sqrt d2-square))
     (define min-pos
       (if (s<= x1 x2)
           (s- x2 (s* (s- x2 x1) (s/ (s- (s+ g2 d2) d1) (s+ (s- g2 g1) (s* 2 d2)))))
           (s- x1 (s* (s- x1 x2) (s/ (s- (s+ g1 d2) d1) (s+ (s- g1 g2) (s* 2 d2)))))))
     (py-min (py-max min-pos lo) hi)]
    [else (s/ (s+ lo hi) 2.0)]))

(struct point (t f g gtd))

(define (placed low-first? low high)
  (if low-first? (values low high) (values high low)))

(define (zoom-step t0 f0 gtd0 t1 f1 gtd1 insufficient?)
  (define t-cubic (cubic-interpolate t0 f0 gtd0 t1 f1 gtd1))
  (define hi (py-max t0 t1))
  (define lo (py-min t0 t1))
  (define eps (s* 0.1 (s- hi lo)))
  (cond
    [(~> (py-min (s- hi t-cubic) (s- t-cubic lo)) (s< eps) not) (values t-cubic #f)]
    [(or insufficient? (s>= t-cubic hi) (s<= t-cubic lo))
     (values (if (s< (s-abs (s- t-cubic hi)) (s-abs (s- t-cubic lo))) (s- hi eps) (s+ lo eps)) #f)]
    [else (values t-cubic #t)]))

(define (strong-wolfe obj-func x t d f g gtd
                      #:c1 [c1 1e-4]
                      #:c2 [c2 0.9]
                      #:tolerance-change [tolerance-change 1e-9]
                      #:max-ls [max-ls 25])
  (define d-norm (abs-max d))
  (define (probe step)
    (define-values (f-new g-new) (obj-func x step d))
    (point step f-new g-new (scalar-of (dot g-new d))))
  (define (sufficient-decrease-fails? p)
    (s> (point-f p) (s+ f (s* (s* c1 (point-t p)) gtd))))
  (define (curvature-holds? p)
    (s<= (s-abs (point-gtd p)) (s* (- c2) gtd)))
  (define start (point 0 f g gtd))
  (define-values (first0 first1 first-done? bracketed)
    (let bracketing ([prev start] [new (probe t)] [ls-iter 0])
      (cond
        [(= ls-iter max-ls) (values start new #f ls-iter)]
        [(or (sufficient-decrease-fails? new) (and (> ls-iter 1) (>= (point-f new) (point-f prev))))
         (values prev new #f ls-iter)]
        [(curvature-holds? new) (values new new #t ls-iter)]
        [(s>= (point-gtd new) 0) (values prev new #f ls-iter)]
        [else
         (define t-new (point-t new))
         (define bounds (cons (s+ t-new (s* 0.01 (s- t-new (point-t prev)))) (s* t-new 10)))
         (define t-next
           (cubic-interpolate (point-t prev) (point-f prev) (point-gtd prev)
                              t-new (point-f new) (point-gtd new) bounds))
         (bracketing new (probe t-next) (+ ls-iter 1))])))
  (let zoom ([p0 first0] [p1 first1] [low-first? (<= (point-f first0) (point-f first1))]
             [done? first-done?] [ls-iter bracketed] [insufficient? #f])
    (match-define (point t0 f0 _ gtd0) p0)
    (match-define (point t1 f1 _ gtd1) p1)
    (define-values (low high) (placed low-first? p0 p1))
    (define finished?
      (or done? (>= ls-iter max-ls) (~> (s- t1 t0) s-abs (s* d-norm) (s< tolerance-change))))
    (cond
      [finished?
       (match-define (point t-low f-low g-low _) low)
       (values f-low g-low t-low (+ ls-iter 1))]
      [else
       (define-values (t-next insufficient-next?) (zoom-step t0 f0 gtd0 t1 f1 gtd1 insufficient?))
       (define p (probe t-next))
       (define rejected? (or (sufficient-decrease-fails? p) (>= (point-f p) (point-f low))))
       (define done-next? (and (not rejected?) (curvature-holds? p)))
       (define old-high-dropped?
         (and (not rejected?)
              (not done-next?)
              (~> (point-gtd p) (s* (s- (point-t high) (point-t low))) (s>= 0))))
       (define-values (q0 q1)
         (if rejected?
             (placed low-first? low p)
             (placed low-first? p (if old-high-dropped? low high))))
       (zoom q0 q1 (if rejected? (<= (point-f q0) (point-f q1)) low-first?) done-next?
             (+ ls-iter 1) insufficient-next?)])))

(define (call-with-grad thunk)
  (define was? (grad-enabled?))
  (dynamic-wind (lambda () (check-ok (tr-set-grad-enabled/raw #t) 'lbfgs-minimize!))
                thunk
                (lambda () (check-ok (tr-set-grad-enabled/raw was?) 'lbfgs-minimize!))))

(define (flat-gradient params)
  (cat (for/list ([p (in-list params)])
         (reshape (or (maybe-grad p) (zeros-like p)) -1))))

(define (add-scaled! params step update)
  (for/fold ([offset 0]) ([p (in-list params)])
    (define n (numel p))
    (sub! p (apply reshape (narrow update 0 offset n) (shape p)) (- (value-of step)))
    (+ offset n)))

(define (restore! params saved)
  (for ([p (in-list params)] [s (in-list saved)])
    (copy! p s)))

(struct curvature (y s rho))

(define (in-curvatures memory)
  (in-parallel (map curvature-y memory) (map curvature-s memory) (map curvature-rho memory)))

(define (direction flat-grad memory h-diag)
  (define q (neg flat-grad))
  (define alphas
    (for/list ([(y s rho) (in-curvatures (reverse memory))])
      (define al (~> (dot s q) scalar-of (s* rho)))
      (sub! q y (value-of al))
      al))
  (define r (* q (value-of h-diag)))
  (for ([(y s rho) (in-curvatures memory)] [al (in-list (reverse alphas))])
    (define be (~> (dot y r) scalar-of (s* rho)))
    (sub! r s (value-of (s- be al))))
  r)

(define (remember memory entry history-size)
  (define kept
    (match memory
      [(cons _ rest) #:when (= (length memory) history-size) rest]
      [_ memory]))
  (append kept (list entry)))

(define (lbfgs-minimize! params closure ;; noqa
                         #:lr [lr 1.0]
                         #:max-iter [max-iter 20]
                         #:max-eval [max-eval #f]
                         #:tolerance-grad [tolerance-grad 1e-7]
                         #:tolerance-change [tolerance-change 1e-9]
                         #:history-size [history-size 100]
                         #:line-search [line-search #f])
  (define max-evals (or max-eval (quotient (* max-iter 5) 4)))
  (define (evaluate)
    (call-with-grad (lambda () (item (closure)))))
  (define (directional-evaluate saved step d)
    (add-scaled! params step d)
    (define loss (evaluate))
    (define flat-grad (flat-gradient params))
    (restore! params saved)
    (values loss flat-grad))
  (define (optimal? flat-grad)
    (s<= (abs-max flat-grad) tolerance-grad))
  (with-no-grad
    (define first-loss (evaluate))
    (define first-grad (flat-gradient params))
    (cond
      [(optimal? first-grad) (values 0 first-loss)]
      [else
       (let loop ([n-iter 1] [loss first-loss] [flat-grad first-grad] [current-evals 1]
                  [d #f] [t #f] [memory '()] [h-diag 1] [prev-flat-grad #f])
         (define y (and prev-flat-grad (- flat-grad prev-flat-grad)))
         (define s (and prev-flat-grad (* d (value-of t))))
         (define ys (and prev-flat-grad (scalar-of (dot y s))))
         (define-values (memory* h-diag*)
           (cond
             [(not prev-flat-grad) (values '() 1)]
             [(s> ys 1e-10)
              (values (remember memory (curvature y s (s/ 1.0 ys)) history-size)
                      (s/ ys (scalar-of (dot y y))))]
             [else (values memory h-diag)]))
         (define d* (if prev-flat-grad (direction flat-grad memory* h-diag*) (neg flat-grad)))
         (define t*
           (if prev-flat-grad
               lr
               (s* (py-min 1.0 (s/ 1.0 (~> flat-grad abs sum scalar-of))) lr)))
         (define gtd (scalar-of (dot flat-grad d*)))
         (define uphill? (s> gtd (- tolerance-change)))
         (define last-iter? (= n-iter max-iter))
         (define-values (loss* flat-grad* step opt? evals)
           (cond
             [uphill? (values loss flat-grad t* #t 0)]
             [line-search
              (define-values (f-new g-new t-new ls-evals)
                (strong-wolfe directional-evaluate (map copy-of params) t* d* loss flat-grad gtd
                              #:max-ls (- max-evals current-evals)))
              (add-scaled! params t-new d*)
              (values f-new g-new t-new (optimal? g-new) ls-evals)]
             [last-iter?
              (add-scaled! params t* d*)
              (values loss flat-grad t* #f 0)]
             [else
              (add-scaled! params t* d*)
              (define loss-new (evaluate))
              (define g-new (flat-gradient params))
              (values loss-new g-new t* (optimal? g-new) 1)]))
         (define evals* (+ current-evals evals))
         (define stop?
           (or uphill?
               last-iter?
               (>= evals* max-evals)
               opt?
               (~> (* d* (value-of step)) abs-max (s<= tolerance-change))
               (~> (- loss* loss) abs (< tolerance-change))))
         (if stop?
             (values n-iter loss*)
             (loop (+ n-iter 1) loss* flat-grad* evals* d* step memory* h-diag* flat-grad)))])))
