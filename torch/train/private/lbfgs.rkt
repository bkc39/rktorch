#lang racket/base

(require (only-in racket/flonum flsingle)
         (only-in racket/match match-define)
         (only-in racket/vector vector-copy)
         (only-in threading ~>)
         (only-in "../../foreign.rkt"
                  * - abs cat copy! dot dtype item max maybe-grad narrow neg numel reshape shape
                  sub! sum with-no-grad zeros-like))

(provide lbfgs-minimize!)

;; A 0-d tensor of the parameters' dtype, as torch's step holds gtd, ys, ro,
;; H_diag and the step length: arithmetic touching one rounds to that dtype,
;; while losses and hyperparameters stay Python floats.
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
(define s/ (arithmetic /))
(define s< (comparison <))
(define s<= (comparison <=))
(define s> (comparison >))
(define s>= (comparison >=))
(define s-abs (unary abs))
(define s-sqrt (unary sqrt))

(define (py-max a b)
  (if (s> b a) b a))

(define (py-min a b)
  (if (s< b a) b a))

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

(define (vector-with v i x)
  (define copy (vector-copy v))
  (vector-set! copy i x)
  copy)

(define (low-high bracket)
  (if (<= (point-f (vector-ref bracket 0))
          (point-f (vector-ref bracket (- (vector-length bracket) 1))))
      (values 0 1)
      (values 1 0)))

(define (strong-wolfe obj-func x t d f g gtd
                      #:c1 [c1 1e-4]
                      #:c2 [c2 0.9]
                      #:tolerance-change [tolerance-change 1e-9]
                      #:max-ls [max-ls 25])
  (define d-norm (~> d abs max scalar-of))
  (define (probe step)
    (define-values (f-new g-new) (obj-func x step d))
    (point step f-new g-new (scalar-of (dot g-new d))))
  (define (sufficient-decrease-fails? p)
    (s> (point-f p) (s+ f (s* (s* c1 (point-t p)) gtd))))
  (define (curvature-holds? p)
    (s<= (s-abs (point-gtd p)) (s* (- c2) gtd)))
  (define-values (bracket done? bracketed)
    (let bracketing ([prev (point 0 f g gtd)] [new (probe t)] [ls-iter 0])
      (cond
        [(= ls-iter max-ls) (values (vector (point 0 f g gtd) new) #f ls-iter)]
        [(or (sufficient-decrease-fails? new) (and (> ls-iter 1) (>= (point-f new) (point-f prev))))
         (values (vector prev new) #f ls-iter)]
        [(curvature-holds? new) (values (vector new) #t ls-iter)]
        [(s>= (point-gtd new) 0) (values (vector prev new) #f ls-iter)]
        [else
         (define t-new (point-t new))
         (define bounds (cons (s+ t-new (s* 0.01 (s- t-new (point-t prev)))) (s* t-new 10)))
         (define t-next
           (cubic-interpolate (point-t prev) (point-f prev) (point-gtd prev)
                              t-new (point-f new) (point-gtd new) bounds))
         (bracketing new (probe t-next) (+ ls-iter 1))])))
  (define-values (low0 high0) (low-high bracket))
  (let zoom ([bracket bracket] [done? done?] [ls-iter bracketed] [insufficient? #f]
             [low low0] [high high0])
    (define (finish)
      (match-define (point t-low f-low g-low _) (vector-ref bracket low))
      (values f-low g-low t-low (+ ls-iter 1)))
    (cond
      [(or done? (>= ls-iter max-ls)) (finish)]
      [else
       (match-define (vector (point t0 f0 _ gtd0) (point t1 f1 _ gtd1)) bracket)
       (cond
         [(~> (s- t1 t0) s-abs (s* d-norm) (s< tolerance-change)) (finish)]
         [else
          (define t-cubic (cubic-interpolate t0 f0 gtd0 t1 f1 gtd1))
          (define hi (py-max t0 t1))
          (define lo (py-min t0 t1))
          (define eps (s* 0.1 (s- hi lo)))
          (define-values (t-next insufficient-next?)
            (cond
              [(~> (py-min (s- hi t-cubic) (s- t-cubic lo)) (s< eps) not) (values t-cubic #f)]
              [(or insufficient? (s>= t-cubic hi) (s<= t-cubic lo))
               (values (if (s< (s-abs (s- t-cubic hi)) (s-abs (s- t-cubic lo))) (s- hi eps) (s+ lo eps))
                       #f)]
              [else (values t-cubic #t)]))
          (define p (probe t-next))
          (cond
            [(or (sufficient-decrease-fails? p)
                 (>= (point-f p) (point-f (vector-ref bracket low))))
             (define bracket* (vector-with bracket high p))
             (define-values (low* high*) (low-high bracket*))
             (zoom bracket* #f (+ ls-iter 1) insufficient-next? low* high*)]
            [else
             (define done-next? (curvature-holds? p))
             (define span (s- (point-t (vector-ref bracket high)) (point-t (vector-ref bracket low))))
             (define kept
               (if (and (not done-next?) (~> (point-gtd p) (s* span) (s>= 0)))
                   (vector-with bracket high (vector-ref bracket low))
                   bracket))
             (zoom (vector-with kept low p) done-next? (+ ls-iter 1) insufficient-next? low high)])])])))

(define (flat-gradient params)
  (cat (for/list ([p (in-list params)])
         (reshape (or (maybe-grad p) (zeros-like p)) -1))))

(define (add-scaled! params step update)
  (with-no-grad
    (for/fold ([offset 0]) ([p (in-list params)])
      (define n (numel p))
      (sub! p (apply reshape (narrow update 0 offset n) (shape p)) (- (value-of step)))
      (+ offset n))))

(define (copy-of p)
  (define copy (zeros-like p))
  (copy! copy p)
  copy)

(define (restore! params saved)
  (with-no-grad
    (for ([p (in-list params)] [s (in-list saved)])
      (copy! p s))))

(define (abs-max v)
  (~> v abs max scalar-of))

(define (direction flat-grad memory h-diag)
  (define q (neg flat-grad))
  (define alphas
    (for/list ([m (in-list (reverse memory))])
      (match-define (list y s rho) m)
      (define al (~> (dot s q) scalar-of (s* rho)))
      (sub! q y (value-of al))
      al))
  (define r (* q (value-of h-diag)))
  (for ([m (in-list memory)] [al (in-list (reverse alphas))])
    (match-define (list y s rho) m)
    (define be (~> (dot y r) scalar-of (s* rho)))
    (sub! r s (value-of (s- be al))))
  r)

(define (remember memory entry history-size)
  (append (if (= (length memory) history-size) (cdr memory) memory) (list entry)))

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
    (item (closure)))
  (define (directional-evaluate saved step d)
    (add-scaled! params step d)
    (define loss (evaluate))
    (define flat-grad (flat-gradient params))
    (restore! params saved)
    (values loss flat-grad))
  (define (optimal? flat-grad)
    (s<= (abs-max flat-grad) tolerance-grad))
  (define first-loss (evaluate))
  (define first-grad (flat-gradient params))
  (cond
    [(optimal? first-grad) (values 0 first-loss)]
    [else
     (let loop ([n-iter 1] [loss first-loss] [flat-grad first-grad] [opt? #f] [current-evals 1]
                [d #f] [t #f] [memory '()] [h-diag 1] [prev-flat-grad #f])
       (define-values (memory* h-diag*)
         (cond
           [(= n-iter 1) (values '() 1)]
           [else
            (define y (- flat-grad prev-flat-grad))
            (define s (* d (value-of t)))
            (define ys (scalar-of (dot y s)))
            (if (s> ys 1e-10)
                (values (remember memory (list y s (s/ 1.0 ys)) history-size)
                        (s/ ys (scalar-of (dot y y))))
                (values memory h-diag))]))
       (define d* (if (= n-iter 1) (neg flat-grad) (direction flat-grad memory* h-diag*)))
       (define prev-loss loss)
       (define t*
         (if (= n-iter 1)
             (s* (py-min 1.0 (s/ 1.0 (~> flat-grad abs sum scalar-of))) lr)
             lr))
       (define gtd (scalar-of (dot flat-grad d*)))
       (cond
         [(s> gtd (- tolerance-change)) (values n-iter loss)]
         [else
          (define-values (loss* flat-grad* step opt*? evals)
            (cond
              [line-search
               (define saved (map copy-of params))
               (define-values (f-new g-new t-new ls-evals)
                 (strong-wolfe directional-evaluate saved t* d* loss flat-grad gtd
                               #:max-ls (- max-evals current-evals)))
               (add-scaled! params t-new d*)
               (values f-new g-new t-new (optimal? g-new) ls-evals)]
              [else
               (add-scaled! params t* d*)
               (cond
                 [(= n-iter max-iter) (values loss flat-grad t* opt? 0)]
                 [else
                  (define loss-new (evaluate))
                  (define g-new (flat-gradient params))
                  (values loss-new g-new t* (optimal? g-new) 1)])]))
          (define evals* (+ current-evals evals))
          (if (or (= n-iter max-iter)
                  (>= evals* max-evals)
                  opt*?
                  (~> (* d* (value-of step)) abs-max (s<= tolerance-change))
                  (~> (- loss* prev-loss) abs (< tolerance-change)))
              (values n-iter loss*)
              (loop (+ n-iter 1) loss* flat-grad* opt*? evals* d* step memory* h-diag*
                    flat-grad))]))]))
