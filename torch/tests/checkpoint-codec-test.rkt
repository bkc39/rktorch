#lang racket/base

(module+ test
  (require (only-in json string->jsexpr)
           (only-in racket/contract/base flat-contract-predicate)
           (only-in racket/match match)
           rackunit
           (only-in "../main.rkt" full ones randn zeros)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt" call-with-live-run stored)
           (only-in "private/comparable.rkt" comparable)
           (only-in "private/safetensors-reader.rkt" safetensors-metadata))

  (define dtypes '(float32 float64 float16 bfloat16 int64 uint8 bool))

  (define (pick xs) (list-ref xs (random (length xs))))

  (define (random-tensor)
    (define dt (pick dtypes))
    (pick (list (full 1 #:dtype dt) (zeros 0 3 #:dtype dt) (ones 2 3 #:dtype dt)
                (if (memq dt '(float32 float64)) (randn 3 #:dtype dt) (zeros 4 #:dtype dt)))))

  (define (random-leaf)
    (pick (list #t #f 0 -7 (expt 2 70) (- (expt 2 70)) 1/10 -7/3 (/ (expt 10 30) 7)
                1.5 -0.0 0.0 +inf.0 -inf.0 +nan.0 1e-300
                "" "plain" "non-ASCII π ✓" "$ in a value" 'sym '|with space| '$sym '()
                (random-tensor))))

  (define (stable-key)
    (pick (list 'a 'b '$c 0 1 -1 #t #f '())))

  (define (eqv-key)
    (pick (list 'a '$c 0 -1 #t '() 1.5 -0.0 0.0 +nan.0 (expt 2 80) -7/3)))

  (define (equal-key)
    (pick (list "a" "b" "π" 3 2.5 'sym '(1 2) (cons 'x 1.0) #f)))

  (define (random-node depth)
    (define width (random 6))
    (define (child) (random-value (sub1 depth)))
    (case (random 5)
      [(0) (for/list ([_ (in-range width)]) (child))]
      [(1) (cons (child) (random-leaf))]
      [(2) (for/hasheq ([_ (in-range width)]) (values (stable-key) (child)))]
      [(3) (for/hasheqv ([_ (in-range width)]) (values (eqv-key) (child)))]
      [else (for/hash ([_ (in-range width)]) (values (equal-key) (child)))]))

  (define (random-value depth)
    (if (or (zero? depth) (< (random) 0.3)) (random-leaf) (random-node depth)))

  (define (immutable-throughout? v)
    (match v
      [(? hash?) (and (immutable? v)
                      (for/and ([(k x) (in-hash v)])
                        (and (immutable-throughout? k) (immutable-throughout? x))))]
      [(cons a d) (and (immutable-throughout? a) (immutable-throughout? d))]
      [_ #t]))

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
            (check-equal? (comparable back #:hash-kinds? #t) (comparable v #:hash-kinds? #t)
                          (format "value ~a: ~e" i v))
            (check-true (immutable-throughout? back) (format "value ~a is immutable" i))))))))

  (test-case "T3-L8: an eq? or eqv? key must keep its identity through a read"
    (define value? (flat-contract-predicate checkpoint-value/c))
    (for ([v (list (hasheq "a" 1) (hasheq 1.5 1) (hasheq (expt 2 80) 1) (hasheq 1/3 1)
                   (hasheq '(1 2) 1) (hasheq (cons 'a 'b) 1)
                   (hasheqv "a" 1) (hasheqv '(1 2) 1) (hash 'k (hasheq "nested" 1)))])
      (check-false (value? v) (format "~e" v)))
    (for ([v (list (hasheq 'a 1 0 2 #t 3 '() 4) (hasheqv 1.5 1 (expt 2 80) 2 1/3 3 -0.0 4)
                   (hash "a" 1 1.5 2 '(1 2) 3))])
      (check-true (value? v) (format "~e" v))))

  (test-case "T3-L8: an exact non-integer is {\"$rational\": \"n/d\"}, an integer a JSON integer"
    (with-temporary-directory (dir #:template "rational-~a")
      (define path (build-path dir "r.ckpt"))
      (void (call-with-live-run #:callbacks (list (stored (box (list 1/10 -7/3 5 0.5))))
                                (lambda (r) (save-checkpoint! r path))))
      (define tree (string->jsexpr (hash-ref (safetensors-metadata path) 'checkpoint)))
      (check-equal? (hash-ref tree 'probe)
                    (list (hasheq '$rational "1/10") (hasheq '$rational "-7/3") 5 0.5))
      (check-equal? (hash-ref (read-checkpoint path) "probe") (list 1/10 -7/3 5 0.5)))))
