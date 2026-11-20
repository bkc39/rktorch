#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list remove-duplicates)
           (only-in racket/match match-define)
           "../main.rkt"
           "../vision/transforms.rkt"
           (only-in (submod "../vision/transforms.rkt" private)
                    batch-rng native-uint8-resize? resize/path resized-crop)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/gradcheck.rkt" gradcheck)
           (only-in "private/op-calls.rkt" op-calls-of)
           (only-in "private/within.rkt" bounded-test))

  (define image-dtypes '(uint8 float16 bfloat16 float32 float64))

  (define (pixels shape #:seed [seed 0])
    (apply reshape
           (~> (rand (apply * shape) #:generator (make-generator seed))
               (* 255.0)
               (to-dtype 'uint8))
           shape))

  (define (image shape dtype #:seed [seed 0])
    (define bytes-image (pixels shape #:seed seed))
    (if (eq? dtype 'uint8)
        bytes-image
        (convert-image-dtype bytes-image dtype)))

  (define (same-bytes? a b)
    (equal? (tensor->bytes a) (tensor->bytes b)))

  (define (contiguous-copy v)
    (bytes->tensor (tensor->bytes v) (dtype v) (shape v)))

  (define windows
    (let ([rng (batch-rng (make-generator 0))])
      (append
       (for/list ([_ (in-range 20)])
         (define h (add1 (random 37 rng)))
         (define w (add1 (random 53 rng)))
         (list (random (- 38 h) rng) (random (- 54 w) rng) h w))
       '((0 0 1 1) (0 52 1 1) (36 0 1 1) (36 52 1 1) (0 0 37 53)))))

  (define (crop x window)
    (define rank (length (shape x)))
    (~> x
        (narrow (- rank 2) (car window) (caddr window))
        (narrow (- rank 1) (cadr window) (cadddr window))))

  (bounded-test "E1: every image dtype keeps its dtype, either filter, image or batch"
    (for* ([d (in-list image-dtypes)]
           [lead (in-list '(() (2)))]
           [antialias? (in-list '(#t #f))])
      (define x (image (append lead '(3 9 12)) d))
      (define where (format "~a ~a antialias ~a" d lead antialias?))
      (check-eq? (dtype (resize x 5 #:antialias? antialias?)) d where)
      (check-eq? (dtype (resize x '(13 4) #:antialias? antialias?)) d where)
      (check-eq? (dtype (random-resized-crop x 8 #:antialias? antialias?)) d where)))

  (bounded-test "E2: uint8 takes float's shapes; random-resized-crop a square of the size"
    (define u (pixels '(3 4 6)))
    (check-equal? (shape (resize u 2)) '(3 2 3))
    (check-equal? (shape (resize u '(5 7))) '(3 5 7))
    (check-equal? (shape (resize (pixels '(2 3 6 4)) 3)) '(2 3 4 3))
    (check-equal? (shape (resize (pixels '(3 299 401)) 256)) '(3 256 343))
    (check-equal? (shape (random-resized-crop u 16)) '(3 16 16))
    (check-equal? (shape (random-resized-crop (pixels '(5 3 30 40)) 8)) '(5 3 8 8)))

  (bounded-test "E3: at its own size the image itself, so a resize is idempotent"
    (for ([d (in-list image-dtypes)])
      (define x (image '(3 4 6) d))
      (check-eq? (resize x '(4 6)) x (format "~a (4 6)" d))
      (check-eq? (resize x 4 #:antialias? #f) x (format "~a short side" d))
      (for ([s (in-list '(3 (7 5) 10))])
        (define y (resize x s))
        (check-eq? (resize y s) y (format "~a twice to ~a" d s)))))

  (define (window-copy x window)
    (match-define (list c rows cols) (shape x))
    (match-define (list top left h w) window)
    (define size (quotient (bytes-length (tensor->bytes x)) (* c rows cols)))
    (define all (tensor->bytes x))
    (define kept
      (for*/list ([k (in-range c)] [row (in-range top (+ top h))])
        (define start (* size (~> (* k rows) (+ row) (* cols) (+ left))))
        (subbytes all start (+ start (* size w)))))
    (bytes->tensor (apply bytes-append kept) (dtype x) (list c h w)))

  (bounded-test "E4: resized-crop resizes the window's pixels"
    (for* ([d (in-list '(uint8 float32))]
           [antialias? (in-list '(#t #f))]
           [window (in-list windows)])
      (define x (image '(3 37 53) d #:seed 4))
      (check-true (same-bytes? (resized-crop x window '(11 17) #:antialias? antialias?)
                               (resize (window-copy x window) '(11 17) #:antialias? antialias?))
                  (format "~a ~a antialias ~a" d window antialias?))))

  (bounded-test "E5: a crop view resizes as its contiguous copy does"
    (for* ([d (in-list '(uint8 float32))]
           [antialias? (in-list '(#t #f))]
           [window (in-list (list-tail windows 15))])
      (define v (crop (image '(3 37 53) d #:seed 5) window))
      (check-true (same-bytes? (resize v '(9 14) #:antialias? antialias?)
                               (resize (contiguous-copy v) '(9 14) #:antialias? antialias?))
                  (format "~a ~a antialias ~a" d window antialias?))))

  (bounded-test "E6: a batch resizes as each of its images"
    (for* ([d (in-list '(uint8 float32))]
           [antialias? (in-list '(#t #f))])
      (define b (image '(4 3 21 30) d #:seed 6))
      (define out (resize b 12 #:antialias? antialias?))
      (for ([i (in-range 4)])
        (check-true (same-bytes? (select out 0 i)
                                 (resize (select b 0 i) 12 #:antialias? antialias?))
                    (format "~a image ~a antialias ~a" d i antialias?)))))

  (define paths (remove-duplicates (list 'float (if native-uint8-resize? 'native 'float))))

  (bounded-test "E7: a constant image stays that constant"
    (for* ([size (in-list '((20 30) (100 141) (7 3)))]
           [antialias? (in-list '(#t #f))])
      (for ([path (in-list paths)])
        (define u (full 173 3 37 53 #:dtype 'uint8))
        (check-equal? (remove-duplicates (tensor->list (resize/path u size antialias? path)))
                      '(173)
                      (format "uint8 ~a ~a antialias ~a" path size antialias?)))
      (define f (full 0.5 3 37 53))
      (check-true (< (item (max (abs (- (resize f size #:antialias? antialias?) 0.5)))) 3e-7)
                  (format "float32 ~a antialias ~a" size antialias?))))

  (bounded-test "E8: the float path rounds half to even; the native x86 kernel in fixed point"
    (define u (reshape (tensor #"\12\13\24\27") 1 1 4))
    (check-equal? (tensor->list (resize/path u '(1 2) #f 'float)) '(10 22))
    (when (and native-uint8-resize? (eq? (system-type 'arch) 'x86_64))
      (check-equal? (tensor->list (resize u '(1 2) #:antialias? #f)) '(11 22))))

  (bounded-test "E10: dtype commutes with the draws, within a count"
    (define u (pixels '(3 61 83) #:seed 7))
    (for ([k (in-range 5)])
      (define a (convert-image-dtype (random-resized-crop u 32 #:generator (make-generator k))))
      (define b (random-resized-crop (convert-image-dtype u) 32 #:generator (make-generator k)))
      (check-true (<= (item (max (abs (- a b)))) (+ 1/255 1e-6)) (format "seed ~a" k))))

  (define blamed? (blames-caller? "resize-test.rkt"))

  (bounded-test "E11: what resize and random-resized-crop refuse, blaming the caller"
    (define x (rand 3 8 8))
    (for ([bad (list (to-dtype x 'int64) (ne x 0.5) (rand 8 8) (rand 1 1 3 8 8)
                     (zeros 3 0 8))])
      (check-exn blamed? (lambda () (resize bad 4)) (format "~a ~a" (dtype bad) (shape bad))))
    (check-exn blamed? (lambda () (resize x 0)))
    (check-exn blamed? (lambda () (resize x '(1 2 3))))
    (for ([bad (list (to-dtype x 'int64) (ne x 0.5))])
      (check-exn blamed? (lambda () (random-resized-crop bad 4)) (format "~a" (dtype bad))))
    (define u (to-dtype x 'uint8))
    (check-exn blamed? (lambda () (normalize u '(0 0 0) '(1 1 1))))
    (check-exn blamed? (lambda () (imagenet-normalize u))))

  (define (calls thunk)
    (thunk)
    (op-calls-of thunk))

  (define probe (rand 2 3 8 8))
  (define per-op (calls (lambda () (flip probe 3))))
  (define per-read (calls (lambda () (dtype probe))))
  (define (budget ops reads)
    (+ (* ops per-op) (* reads per-read)))

  (bounded-test "E12: native calls per resize, the same at any size or batch"
    (define (resize-calls lead side d [path #f])
      (define x (image (append lead (list 3 side side)) d))
      (calls (lambda () (resize/path x '(7 9) #t path))))
    (define (contracted-calls x)
      (calls (lambda () (resize x '(7 9)))))
    (define (native? d) (or (eq? d 'float32) native-uint8-resize?))
    (for ([d (in-list '(uint8 float32))])
      (define device-read (if (eq? d 'uint8) 1 0))
      (define four (resize-calls '(2) 16 d))
      (check-equal? (resize-calls '(2) 512 d) four (format "~a 16 vs 512" d))
      (check-equal? (resize-calls '(64) 16 d) four (format "~a N 2 vs 64" d))
      (check-true (<= four (if (native? d)
                               (budget 1 (+ 1 device-read))
                               (budget 4 (+ 3 device-read))))
                  (format "~a [N C H W]: ~a" d four))
      (check-equal? (contracted-calls (image '(2 3 16 16) d)) (+ four per-read)
                    (format "~a: the contract reads the dtype once" d))
      (define three (resize-calls '() 16 d))
      (check-equal? (resize-calls '() 512 d) three)
      (check-true (<= three (if (native? d)
                                (budget 3 (+ 1 device-read))
                                (budget 6 (+ 3 device-read))))
                  (format "~a [C H W]: ~a" d three)))
    (check-true (<= (resize-calls '() 16 'uint8 'float) (budget 6 3)))
    (define u (pixels '(3 64 64)))
    (define g (make-generator 1))
    (define rrc (calls (lambda () (random-resized-crop u 24 #:generator g))))
    (check-true (<= rrc (if native-uint8-resize? (budget 5 5) (budget 8 6)))
                (format "random-resized-crop: ~a" rrc)))

  (bounded-test "E15: a [C H W] or [N C H W] uint8 image in NCHW order resizes to 1x1 as its channels-last copy does"
    (define (channels-last-copy x)
      (define-values (to-hwc from-hwc)
        (if (~> (shape x) length (= 3))
            (values '(1 2 0) '(2 0 1))
            (values '(0 2 3 1) '(0 3 1 2))))
      (define hwc (apply permute x to-hwc))
      (apply permute (bytes->tensor (tensor->bytes hwc) 'uint8 (shape hwc)) from-hwc))
    (for* ([lead (in-list '(() (1) (2)))]
           [c (in-list '(1 3 4))]
           [antialias? (in-list '(#t #f))])
      (define x (pixels (append lead (list c 8 8)) #:seed 9))
      (check-true (same-bytes? (resize x '(1 1) #:antialias? antialias?)
                               (resize (channels-last-copy x) '(1 1) #:antialias? antialias?))
                  (format "~a ~a antialias ~a" lead c antialias?))))

  (define accelerators
    (append (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (device 'mps)) '())))

  (bounded-test "E13: on an accelerator uint8 stays uint8 there, within a count of the CPU"
    (define u (pixels '(3 37 53) #:seed 8))
    (for ([dev (in-list accelerators)])
      (define there (resize (to u dev) '(20 30)))
      (check-equal? (device there) dev)
      (check-eq? (dtype there) 'uint8)
      (define gap (~> (to there (device 'cpu)) (to-dtype 'float32)
                      (- (to-dtype (resize u '(20 30)) 'float32)) abs max item))
      (check-true (<= gap 1.0) (format "~a: ~a" dev gap))))

  (bounded-test "E14: a float resize is differentiable"
    (for* ([size (in-list '((3 4) (9 11)))]
           [antialias? (in-list '(#t #f))])
      (check-true (gradcheck (lambda (x) (resize x size #:antialias? antialias?))
                             (list (randn 2 5 7 #:dtype 'float64)))
                  (format "~a antialias ~a" size antialias?)))))
