#lang racket/base

(module+ test
  (require (only-in racket/list append*)
           (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../main.rkt"
                    * backward! grad item manual-seed! randn reshape sum tensor tensor->list
                    zeros)
           (only-in "../nn.rkt" Parameter clip-grad-norm! clip-grad-value!)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/op-calls.rkt" crossings-of op-calls-of)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "clip-grad-test"))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (with-gradients specs)
    (for/list ([spec (in-list specs)])
      (match-define (list values-in dt) spec)
      (define p (~> (length values-in) (zeros #:dtype dt) Parameter))
      (~> (* p (tensor values-in #:dtype dt)) sum backward!)
      p))

  (define (mixed-setup)
    (define used
      (with-gradients '(((0.5 -1.25 2.0) float32)
                        ((3.0 -0.75) float64)
                        ((-2.5 0.125 1.5 -0.25) float32))))
    (append used (list (Parameter (zeros 2)))))

  (define (gradients ps)
    (for/list ([p (in-list ps)] #:when (grad p))
      (flat (grad p))))

  (define norm-types '(2.0 1.0 0.5 +inf.0))

  (define (clipped norm-type foreach?)
    (define ps (mixed-setup))
    (define total
      (clip-grad-norm! ps 1.0 #:norm-type norm-type #:foreach? foreach?))
    (values (item total) (gradients ps)))

  (for ([norm-type (in-list norm-types)])
    (bounded-test (format "G0-L1: foreach and loop agree bit for bit at norm type ~a" norm-type)
      (define-values (fe-norm fe-grads) (clipped norm-type #t))
      (define-values (lp-norm lp-grads) (clipped norm-type #f))
      (check-= fe-norm lp-norm 0.0)
      (check-equal? fe-grads lp-grads)))

  (bounded-test "G0-L1: an empty parameter list answers a zero norm in both modes"
    (for ([foreach? (in-list '(#t #f))])
      (check-equal? (item (clip-grad-norm! '() 1.0 #:norm-type 1.0 #:foreach? foreach?))
                    0.0)))

  (bounded-test "G0-L2: a bound above the norm leaves every gradient bit-unchanged"
    (for ([norm-type (in-list norm-types)])
      (define ps (mixed-setup))
      (define before (gradients ps))
      (clip-grad-norm! ps 1e6 #:norm-type norm-type)
      (check-equal? (gradients ps) before (format "norm type ~a" norm-type))))

  (bounded-test "G0-L3: a bound of zero zeroes every finite gradient element"
    (for ([norm-type (in-list norm-types)])
      (define ps (mixed-setup))
      (clip-grad-norm! ps 0 #:norm-type norm-type)
      (check-true (andmap zero? (append* (gradients ps))) (format "norm type ~a" norm-type))))

  (bounded-test "G0-L4: the infinity norm is the largest gradient magnitude"
    (define ps (mixed-setup))
    (define largest (apply max (map abs (append* (gradients ps)))))
    (check-= (item (clip-grad-norm! ps 1e6 #:norm-type +inf.0)) largest 0.0))

  (for ([bad (in-list '(+nan.0 +inf.0 -inf.0))])
    (bounded-test (format "G0-L5: a ~a gradient raises before any gradient changes" bad)
      (define ps (with-gradients `(((1.0 ,bad 2.0) float32) ((0.5 0.25) float32))))
      (define before (gradients ps))
      (check-exn #rx"non-finite"
                 (lambda () (clip-grad-norm! ps 1.0 #:error-if-nonfinite? #t)))
      (check-equal? (gradients ps) before)
      (check-not-exn (lambda () (clip-grad-norm! ps 1.0)))))

  (define (clamped xs v)
    (for/list ([x (in-list xs)]) (max (- v) (min v x))))

  (bounded-test "G0-L6: clip-grad-value! is the elementwise clamp, foreach and loop alike"
    (define (run foreach?)
      (define ps (mixed-setup))
      (define before (gradients ps))
      (clip-grad-value! ps 0.25 #:foreach? foreach?)
      (check-equal? (gradients ps) (map (lambda (xs) (clamped xs 0.25)) before))
      (gradients ps))
    (check-equal? (run #t) (run #f)))

  (define (steady-params n dtypes)
    (manual-seed! 0)
    (for*/list ([dt (in-list dtypes)] [_ (in-range n)])
      (define p (Parameter (randn 3 #:dtype dt)))
      (~> (* p p) sum backward!)
      p))

  (define (steady-crossings clip! ps)
    (clip! ps)
    (crossings-of (lambda () (clip! ps))))

  (define (norm-clip! ps) (clip-grad-norm! ps 1.0))
  (define (value-clip! ps) (clip-grad-value! ps 0.25))

  (bounded-test "G0-L8: one bucket's steady clip-grad-norm! makes 10 crossings at any count"
    (for ([n (in-list '(1 8 64))])
      (define ps (steady-params n '(float32)))
      (check-equal? (steady-crossings norm-clip! ps) 10 (format "~a parameters" n))
      (check-equal? (op-calls-of (lambda () (norm-clip! ps))) 9 (format "~a parameters" n))))

  (bounded-test "G0-L9: each extra bucket adds 3 crossings, and the first extra a cat"
    (define one (steady-crossings norm-clip! (steady-params 4 '(float32))))
    (define two (steady-crossings norm-clip! (steady-params 4 '(float32 float64))))
    (define three (steady-crossings norm-clip! (steady-params 4 '(float32 float64 bfloat16))))
    (check-equal? (- two one) 4)
    (check-equal? (- three two) 3))

  (bounded-test "G0-L10: one bucket's clip-grad-value! makes 4 crossings at any count"
    (for ([n (in-list '(1 8 64))])
      (check-equal? (steady-crossings value-clip! (steady-params n '(float32))) 4
                    (format "~a parameters" n))))

  (define bad-calls
    `(("a non-list" ,(lambda (ps) (clip-grad-norm! (car ps) 1.0)))
      ("a negative max-norm" ,(lambda (ps) (clip-grad-norm! ps -1.0)))
      ("a non-real max-norm" ,(lambda (ps) (clip-grad-norm! ps 1+2i)))
      ("a NaN norm-type" ,(lambda (ps) (clip-grad-norm! ps 1.0 #:norm-type +nan.0)))
      ("a negative clip-value" ,(lambda (ps) (clip-grad-value! ps -0.5)))))

  (for ([case (in-list bad-calls)])
    (match-define (list label call) case)
    (bounded-test (format "G0-L11: ~a blames the caller" label)
      (define ps (mixed-setup))
      (check-exn blames-this-test? (lambda () (call ps))))))
