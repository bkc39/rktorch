#lang scribble/lp2

@(require torch/scribblings/labels)

@title[#:tag "ex-workers"]{Loading batches in parallel}

The convnet of @secref["ex-mnist"] again, now trained from images stored the
way files arrive: each one encoded as plain-PGM text, decoded and randomly
rotated and shifted in Racket as it is loaded. That costs about 1.5 ms per
image, so a batch of 64 takes about 90 ms to load against a few milliseconds
to train on the GPU: with @racket[#:workers] at 0, loading is the epoch. With
@racket[#:workers 4], four threads decode the next batches while the step runs.
On an RTX 3090 Ti one epoch takes 87 s with no workers and 30 s with four, and
the network reaches 98.6% test accuracy after the two.

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
(provide collate-digits encoded-digits first-labels run-example train-workers)]

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
@tt{P2}; the first three are the header. The jitter rotates the digit by up
to 15 degrees and shifts it by up to two pixels, sampling the source
bilinearly, and writes float32 pixels. Its draws come from
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

(define (jitter pixels)
  (define angle (* (- (random) 0.5) (/ pi 6)))
  (define c (cos angle))
  (define s (sin angle))
  (define dx (* 4 (- (random) 0.5)))
  (define dy (* 4 (- (random) 0.5)))
  (define out (make-bytes (* 4 784)))
  (for* ([y (in-range 28)] [x (in-range 28)])
    (define u (- x 13.5 dx))
    (define v (- y 13.5 dy))
    (real->floating-point-bytes
     (/ (bilinear pixels (+ 13.5 (* c u) (* s v)) (+ 13.5 (* c v) (- (* s u))))
        255.0)
     4 (system-big-endian?) out (~> y (* 28) (+ x) (* 4))))
  out)]

@bold{The dataset.} Item @racket[i] is the decoded, jittered image as 3136
bytes of float32, and its label. The collate builds one tensor per field for
the whole batch, so a worker makes two native calls per batch and does
everything else in Racket. It names the CPU because the step below runs
inside @racket[with-default-device].

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
  (values (~> (vector-ref pgms i) pgm-decode jitter)
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
      (define loss (cross-entropy (net (to xb device)) (to yb device)))
      (backward! loss)
      (step! opt)
      (item loss)))
  (values (/ total (length loader))
          (/ (- (current-inexact-milliseconds) start) 1000.0)))]

@bold{Two epochs, two worker counts.} One network trains an epoch with no
workers, then one with @racket[workers]; each row is the worker count, the
mean training loss, the seconds the epoch took and the test accuracy after it.
Both loaders draw from one generator, so the second epoch's order follows
the first's.

@chunk[<r17-compare>
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
      (for/list ([n (in-list (list 0 workers))])
        (define loader
          (dataloader ds #:batch-size batch #:shuffle? #t #:collate collate-digits
                      #:generator generator #:workers n))
        (define-values (loss seconds) (train-epoch! net opt loader device))
        (list n loss seconds (accuracy net xs ys))))))]

@racket[run-example] is the offline shadow the test drives, on the committed
256-image fixture; @racket[train-workers] is the headline on the full set.

@chunk[<r17-run>
(define (run-example #:workers [workers 2] #:device [device (pick-device)])
  (define-values (xs ys) (load-mnist-fixture))
  (compare-workers (encoded-digits xs ys) xs ys
                   #:workers workers #:batch 64 #:device device))

(define (train-workers #:workers [workers 4] #:batch [batch 64]
                       #:device [device (pick-device)])
  (define-values (train-x train-y) (load-mnist 'train))
  (define-values (test-x test-y) (load-mnist 'test))
  (compare-workers (encoded-digits train-x train-y) test-x test-y
                   #:workers workers #:batch batch #:device device))]

@bold{Order, breaks and errors.} The sampler runs on the consuming thread, so
a seed gives the same batch order for any worker count; only the jitter
differs. @racket[first-labels] takes one batch and leaves the loop: stopping
early needs nothing closed, since the workers stop once the traversal is
collected. @racket[(equal? (first-labels ds 0) (first-labels ds 4))] is
@racket[#t]. A batch whose construction raises raises in the loop at that
batch's position, as it would with no workers.

@chunk[<r17-first-labels>
(define (first-labels ds workers)
  (for/first ([(_xb yb) (in-dataloader
                         (dataloader ds #:batch-size 8 #:shuffle? #t
                                     #:collate collate-digits
                                     #:generator (make-generator 0)
                                     #:workers workers))])
    (tensor->list yb)))]

@bold{What does not parallelize yet.} Torch operations called from a worker
run one at a time, through the main thread, until
@hyperlink["https://github.com/bkc39/rktorch/issues/296"]{#296} lets them
leave atomic mode. A transform built from them gains nothing from workers:
the same epoch with @racket[random-crop] applied per image took 14.1 s with
no workers and 25.0 s with four. Keep a worker's per-item work in Racket and
its native calls to one per batch; the @racket[dataloader] entry has the
rest.

@chunk[<*>
  <r17-require>
  <r17-provide>
  <r17-encode>
  <r17-decode>
  <r17-dataset>
  <r17-epoch>
  <r17-compare>
  <r17-run>
  <r17-first-labels>]
