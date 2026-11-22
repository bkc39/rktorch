#lang racket/base

(module+ test
  (require rackunit
           (only-in "../main.rkt" full ones randn zeros)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run decoded-form stored))

  (define dtypes '(float32 float64 float16 bfloat16 int64 uint8 bool))

  (define (pick xs) (list-ref xs (random (length xs))))

  (define (random-tensor)
    (define dt (pick dtypes))
    (pick (list (full 1 #:dtype dt) (zeros 0 3 #:dtype dt) (ones 2 3 #:dtype dt)
                (if (memq dt '(float32 float64)) (randn 3 #:dtype dt) (zeros 4 #:dtype dt)))))

  (define (random-leaf)
    (pick (list #t #f 0 -7 (expt 2 70) (- (expt 2 70)) 1.5 -0.0 0.0 +inf.0 -inf.0 +nan.0 1e-300
                "" "plain" "non-ASCII π ✓" "$ in a value" 'sym '|with space| '$sym '()
                (random-tensor))))

  (define (stable-key)
    (pick (list 'a 'b '$c 0 1 -1 #t #f '())))

  (define (equal-key)
    (pick (list "a" "b" "π" 3 2.5 'sym '(1 2) (cons 'x 1.0) #f)))

  (define (random-node depth)
    (define width (random 6))
    (define (child) (random-value (sub1 depth)))
    (case (random 5)
      [(0) (for/list ([_ (in-range width)]) (child))]
      [(1) (cons (child) (random-leaf))]
      [(2) (for/hasheq ([_ (in-range width)]) (values (stable-key) (child)))]
      [(3) (for/hasheqv ([_ (in-range width)]) (values (stable-key) (child)))]
      [else (for/hash ([_ (in-range width)]) (values (equal-key) (child)))]))

  (define (random-value depth)
    (if (or (zero? depth) (< (random) 0.3)) (random-leaf) (random-node depth)))

  (define (immutable-throughout? v)
    (cond
      [(hash? v) (and (immutable? v)
                      (for/and ([(k x) (in-hash v)])
                        (and (immutable-throughout? k) (immutable-throughout? x))))]
      [(pair? v) (and (immutable-throughout? (car v)) (immutable-throughout? (cdr v)))]
      [else #t]))

  (define values-to-check
    (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
      (random-seed 386)
      (append (list (make-hash (list (cons "mutable" (make-hasheqv (list (cons 1 2.0))))))
                    (list (full 1.5) (zeros 0 #:dtype 'bool) (ones 0 2 #:dtype 'bfloat16)))
              (for/list ([_ (in-range 200)]) (random-value 4)))))

  (test-case "T3-L8: every checkpoint value reads back equal, hashes immutable"
    (with-temporary-directory (dir #:template "codec-~a")
      (define current (box #f))
      (define path (build-path dir "probe.ckpt"))
      (void
       (call-with-live-run
        #:callbacks (list (stored current))
        (lambda (r)
          (for ([v (in-list values-to-check)] [i (in-naturals)])
            (set-box! current v)
            (save-checkpoint! r path)
            (define back (hash-ref (read-checkpoint path) "probe"))
            (check-equal? (decoded-form back) (decoded-form v) (format "value ~a: ~e" i v))
            (check-true (immutable-throughout? back) (format "value ~a is immutable" i)))))))))
