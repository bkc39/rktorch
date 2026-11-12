#lang racket/base

(module+ test
  (require (only-in racket/flonum flsingle)
           (only-in racket/list append* first)
           (only-in racket/match match-define)
           rackunit
           (only-in threading ~>)
           (only-in "../generated.rkt" linalg-vector-norm)
           (only-in "../main.rkt"
                    * backward! cuda-available? device grad has-grad? item manual-seed! randn
                    reshape sum tensor tensor->list zeros)
           (only-in "../nn.rkt" Linear Parameter clip-grad-norm! clip-grad-value! parameters)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/op-calls.rkt" crossings-of op-calls-of)
           (only-in "private/within.rkt" bounded-test))

  (define blames-this-test? (blames-caller? "clip-grad-test"))

  (define (flat t)
    (~> t (reshape -1) tensor->list))

  (define (with-gradients specs #:device [dev 'cpu])
    (for/list ([spec (in-list specs)])
      (match-define (list values-in dt) spec)
      (define p (~> (length values-in) (zeros #:dtype dt #:device dev) Parameter))
      (~> (* p (tensor values-in #:dtype dt #:device dev)) sum backward!)
      p))

  (define (mixed-setup)
    (define used
      (with-gradients '(((0.5 -1.25 2.0) float32)
                        ((3.0 -0.75) float64)
                        ((-2.5 0.125 1.5 -0.25) float32))))
    (append used (list (Parameter (zeros 2)))))

  (define (gradients ps)
    (for/list ([p (in-list ps)] #:when (has-grad? p))
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

  (bounded-test "G0-L5: the flag on a finite norm changes neither the norm nor any gradient"
    (define (run flag?)
      (define ps (mixed-setup))
      (define total (item (clip-grad-norm! ps 1.0 #:error-if-nonfinite? flag?)))
      (list total (gradients ps)))
    (check-equal? (run #t) (run #f)))

  (bounded-test "G0-L6: a clip-value of zero keeps each gradient's sign on its zero"
    (for ([foreach? (in-list '(#t #f))])
      (define ps (with-gradients '(((0.5 -1.25 2.0) float32) ((-3.0 0.75) float64))))
      (clip-grad-value! ps 0 #:foreach? foreach?)
      (for ([got (in-list (append* (gradients ps)))]
            [want (in-list '(0.0 -0.0 0.0 -0.0 0.0))])
        (check-eqv? got want (format "foreach? ~a" foreach?)))))

  (define (scaled-linear-gradients)
    (manual-seed! 0)
    (define first-layer (Linear 3 2))
    (define second-layer (Linear 2 1))
    (~> (randn 64 3) first-layer second-layer sum backward!)
    (append (parameters first-layer) (parameters second-layer)))

  (define (largest-magnitude ps)
    (apply max (map abs (append* (gradients ps)))))

  (bounded-test "G0-L6: a clamped float32 gradient's largest magnitude is the float32 bound"
    (for ([bound (in-list '(0.25 0.1))])
      (define ps (scaled-linear-gradients))
      (check-true (> (largest-magnitude ps) bound))
      (clip-grad-value! ps bound)
      (check-= (largest-magnitude ps) (flsingle bound) 0.0 (format "bound ~a" bound)))
    (check-false (= (flsingle 0.1) 0.1)))

  (when (cuda-available?)
    (define cpu-spec '(((0.5 -1.25 2.0) float32)))
    (define cuda-spec '(((3.0 -0.75) float32)))
    (define (ordered first-dev cpu-part cuda-part)
      (if (eq? first-dev 'cpu) (append cpu-part cuda-part) (append cuda-part cpu-part)))
    (define (mixed first-dev)
      (ordered first-dev (with-gradients cpu-spec) (with-gradients cuda-spec #:device 'cuda)))
    (define (all-cpu first-dev)
      (with-gradients (ordered first-dev cpu-spec cuda-spec)))
    (for ([first-dev (in-list '(cpu cuda))])
      (bounded-test (format "G0-L1: a CPU and a CUDA bucket clip as the all-CPU run, ~a first"
                            first-dev)
        (define ps (mixed first-dev))
        (define total (clip-grad-norm! ps 1.0))
        (check-equal? (device total) (device (grad (first ps))))
        (define reference (all-cpu first-dev))
        (check-= (item total) (item (clip-grad-norm! reference 1.0)) 0.0)
        (check-equal? (gradients ps) (gradients reference)))))

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

  (define (per-result-crossings)
    (define t (randn 3))
    (- (steady-crossings (lambda (t) (linalg-vector-norm t 2.0 #f #f #f)) t) 1))

  (define one-bucket-fixed 29)
  (define extra-bucket-fixed 7)
  (define cat-fixed 5)

  (define (norm-crossings n dtypes)
    (steady-crossings norm-clip! (steady-params n dtypes)))

  (bounded-test "G0-L8: one bucket's steady clip-grad-norm! is 29 crossings plus r per parameter"
    (define r (per-result-crossings))
    (define c1 (norm-crossings 1 '(float32)))
    (check-equal? c1 (+ one-bucket-fixed r))
    (for ([n (in-list '(8 64))])
      (check-equal? (- (norm-crossings n '(float32)) c1) (* (- n 1) r)
                    (format "~a parameters" n))))

  (bounded-test "G0-L9: an extra bucket of k adds 7 crossings plus k r, the first extra a cat"
    (define r (per-result-crossings))
    (define one (norm-crossings 4 '(float32)))
    (for ([k (in-list '(1 4))])
      (define two (steady-crossings norm-clip!
                                    (append (steady-params 4 '(float32))
                                            (steady-params k '(float64)))))
      (check-equal? (- two one) (+ cat-fixed extra-bucket-fixed (* k r)) (format "k = ~a" k)))
    (define two (norm-crossings 4 '(float32 float64)))
    (define three (norm-crossings 4 '(float32 float64 bfloat16)))
    (check-equal? (- three two) (+ extra-bucket-fixed (* 4 r))))

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
