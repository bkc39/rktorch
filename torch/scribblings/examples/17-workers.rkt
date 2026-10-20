#lang scribble/lp2

@(require torch/scribblings/labels)

@title[#:tag "ex-workers"]{Loading batches in parallel}

The convnet of @secref["ex-mnist"] again, now trained from images stored as
plain-PGM text and decoded, rotated and shifted in Racket as each is loaded,
about 1.5 ms per image. On an RTX 3090 Ti one epoch takes 86 s with no
workers and 29 s with @racket[#:workers 4], and the network reaches
98.7% test accuracy after the two.

The model, the device and @tt{accuracy} come from the MNIST chapter.

@chunk[<r17-require>
(require torch torch/nn torch/data/loader
         (only-in racket/list first second)
         (only-in racket/math exact-floor pi)
         (only-in torch/vision/mnist load-mnist load-mnist-fixture))
(begin
  (require (only-in torch/scribblings/examples/05-mnist
                    accuracy convnet pick-device)))]

@chunk[<r17-provide>
(provide (struct-out epoch-result)
         collate-digits encoded-digits first-batch first-labels jitter
         pgm-decode pgm-encode run-example train-workers)]

@bold{The encoding.} A PGM file in its plain form is a header, @tt{P2},
the width, the height and the largest value, then one decimal number per
pixel. The training images are encoded once, up front; the test images stay
a tensor.

@chunk[<r17-encode>
(define (pgm-encode pixels)
  (define out (open-output-bytes))
  (write-bytes #"P2\n28 28\n255\n" out)
  (for ([v (in-bytes pixels)] [k (in-naturals 1)])
    (write-string (number->string v) out)
    (write-char (if (zero? (modulo k 28)) #\newline #\space) out))
  (get-output-bytes out))

(define (images->pgms images)
  (define raw (~> images (* 255) (+ 0.5) (to-dtype 'uint8) tensor->bytes))
  (for/vector ([start (in-range 0 (bytes-length raw) 784)])
    (pgm-encode (subbytes raw start (+ start 784)))))]

@bold{Decoding and jitter, in Racket.} The decoder reads the numbers after
@tt{P2}; the first three are the header. @racket[jitter] rotates the digit by
an angle in radians and shifts it by @racket[dx] and @racket[dy] pixels,
sampling the source bilinearly, and writes float32 pixels;
@racket[random-jitter] draws up to 15 degrees and two pixels from
@racket[current-pseudo-random-generator], which each worker sets to its own,
seeded from the loader's generator.

@chunk[<r17-decode>
(define (pgm-decode bs)
  (define pixels (make-bytes 784))
  (for/fold ([count -3] [value #f] #:result pixels)
            ([b (in-bytes bs 2)])
    (cond
      [(<= 48 b 57) (values count (+ (* 10 (or value 0)) (- b 48)))]
      [(not value) (values count #f)]
      [(negative? count) (values (add1 count) #f)]
      [else
       (bytes-set! pixels count value)
       (values (add1 count) #f)])))

(define (pixel-at pixels x y)
  (if (and (< -1 x 28) (< -1 y 28))
      (bytes-ref pixels (+ x (* 28 y)))
      0))

(define (bilinear pixels x y)
  (define x0 (exact-floor x))
  (define y0 (exact-floor y))
  (define fx (- x x0))
  (define fy (- y y0))
  (+ (* (- 1 fx) (- 1 fy) (pixel-at pixels x0 y0))
     (* fx (- 1 fy) (pixel-at pixels (add1 x0) y0))
     (* (- 1 fx) fy (pixel-at pixels x0 (add1 y0)))
     (* fx fy (pixel-at pixels (add1 x0) (add1 y0)))))

(define (jitter pixels angle dx dy)
  (define c (cos angle))
  (define s (sin angle))
  (define out (make-bytes (* 4 784)))
  (for* ([y (in-range 28)] [x (in-range 28)])
    (define u (- x 13.5 dx))
    (define v (- y 13.5 dy))
    (real->floating-point-bytes
     (/ (bilinear pixels (+ 13.5 (* c u) (* s v)) (+ 13.5 (* c v) (- (* s u))))
        255.0)
     4 (system-big-endian?) out (~> y (* 28) (+ x) (* 4))))
  out)

(define (random-jitter pixels)
  (jitter pixels
          (* (- (random) 0.5) (/ pi 6))
          (* 4 (- (random) 0.5))
          (* 4 (- (random) 0.5))))]

@bold{The dataset.} Item @racket[i] is the decoded, jittered image as 3136
bytes of float32, and its label. The collate builds one tensor per field for
the whole batch, so a worker makes two native calls per batch and does
everything else in Racket, since torch calls inside a worker do not yet run
in parallel (see @racket[dataloader]). It names the CPU because the step
below runs inside @racket[with-default-device].

@chunk[<r17-dataset>
(define (collate-digits items)
  (values (bytes->tensor (apply bytes-append (map first items)) 'float32
                         (list (length items) 1 28 28)
                         #:device 'cpu)
          (tensor (map second items) #:dtype 'int64 #:device 'cpu)))

(define-dataset encoded-digits (pgms labels)
  #:init (images label-tensor)
  (set! pgms (images->pgms images))
  (set! labels (list->vector (tensor->list label-tensor)))
  #:length (vector-length labels)
  #:ref (i)
  (values (~> (vector-ref pgms i) pgm-decode random-jitter)
          (vector-ref labels i)))]

@bold{One epoch.} Workers build CPU tensors, and @racket[dataloader] refuses
@racket[#:workers] for a dataset on an accelerator, so each batch moves to the
device in the step. The loss is read every step, which also keeps the wall
time honest on a GPU.

@chunk[<r17-epoch>
(define (train-epoch! net opt loader device)
  (define start (current-inexact-milliseconds))
  (define total
    (for/sum ([(xb yb) (in-dataloader loader)])
      (zero-grads! opt)
      (define loss (~> xb (to device) net (cross-entropy (to yb device))))
      (backward! loss)
      (step! opt)
      (item loss)))
  (values (/ total (length loader))
          (/ (- (current-inexact-milliseconds) start) 1000.0)))]

@bold{Two epochs, two worker counts.} One network trains an epoch with no
workers, then one with @racket[workers]; each @racket[epoch-result] holds the
worker count, the mean training loss, the seconds the epoch took and the
test accuracy after it. Both loaders draw from one generator, so the second
epoch's order follows the first's. Three forward and backward passes on test
images, with no optimizer step, take the device's one-time costs out of the
first timed epoch.

@chunk[<r17-compare>
(struct epoch-result (workers loss seconds accuracy) #:transparent)

(define (warm-up! net xs ys batch)
  (for ([_ (in-range 3)])
    (~> (narrow xs 0 0 batch) net (cross-entropy (narrow ys 0 0 batch))
        backward!)))

(define (compare-workers ds test-x test-y
                         #:workers workers #:batch batch #:device device)
  (define generator (make-generator 0))
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed 0)
    (with-default-device device
      (manual-seed! 0)
      (define net (convnet))
      (define opt (adam (parameters net) #:lr 0.001))
      (define xs (to test-x device))
      (define ys (to test-y device))
      (warm-up! net xs ys batch)
      (for/list ([n (in-list (list 0 workers))])
        (define loader
          (dataloader ds #:batch-size batch #:shuffle? #t #:collate collate-digits
                      #:generator generator #:workers n))
        (define-values (loss seconds) (train-epoch! net opt loader device))
        (epoch-result n loss seconds (accuracy net xs ys))))))]

@racket[run-example] is the offline shadow the test drives, on the committed
256-image fixture in batches of eight; @racket[train-workers] is the headline
on the full set.

@chunk[<r17-run>
(define (run-example #:workers [workers 2] #:device [device (pick-device)])
  (define-values (xs ys) (load-mnist-fixture))
  (compare-workers (encoded-digits xs ys) xs ys
                   #:workers workers #:batch 8 #:device device))

(define (train-workers #:workers [workers 4] #:batch [batch 64]
                       #:device [device (pick-device)])
  (define-values (train-x train-y) (load-mnist 'train))
  (define-values (test-x test-y) (load-mnist 'test))
  (compare-workers (encoded-digits train-x train-y) test-x test-y
                   #:workers workers #:batch batch #:device device))]

@racket[first-batch] takes one batch and leaves the loop, with nothing to
close; the sampler runs on the consuming thread, so @racket[first-labels] is
the same for any worker count.

@chunk[<r17-first-batch>
(define (first-batch ds #:workers [workers 0] #:collate [collate collate-digits])
  (for/first ([(xb yb) (in-dataloader
                        (dataloader ds #:batch-size 8 #:shuffle? #t
                                    #:collate collate
                                    #:generator (make-generator 0)
                                    #:workers workers))])
    (list xb yb)))

(define (first-labels ds workers)
  (~> (first-batch ds #:workers workers) second tensor->list))]

@chunk[<*>
  <r17-require>
  <r17-provide>
  <r17-encode>
  <r17-decode>
  <r17-dataset>
  <r17-epoch>
  <r17-compare>
  <r17-run>
  <r17-first-batch>]
