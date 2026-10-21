#lang racket/base

(require (only-in racket/match match match-define)
         (only-in racket/math nan? pi)
         torch
         torch/data/loader
         torch/scribblings/examples/17-workers
         (only-in torch/scribblings/examples/05-mnist pick-device)
         (only-in torch/vision/mnist load-mnist-fixture))

(module+ main
  (require (only-in "private/env.rkt" env-number))
  (define workers (env-number "WORKERS" 4))
  (printf "device: ~a\n" (pick-device))
  (define-values (xs ys) (load-mnist-fixture))
  (define ds (encoded-digits xs ys))
  (printf "same first batch for workers 0 and ~a: ~a\n" workers
          (equal? (first-labels ds 0) (first-labels ds workers)))
  (define start (current-inexact-milliseconds))
  (for ([i (in-range (length ds))])
    (dataset-ref ds i))
  (printf "decode + jitter: ~a ms per image\n"
          (real->decimal-string (/ (- (current-inexact-milliseconds) start) (length ds)) 3))
  (for ([row (in-list (train-workers #:workers workers))])
    (match-define (epoch-result n loss seconds acc) row)
    (printf "workers ~a: ~a s, mean loss ~a, test acc ~a\n"
            n (real->decimal-string seconds 2) (real->decimal-string loss 4)
            (real->decimal-string acc 4))))

(define (pixels->floats pixels)
  (apply bytes-append
         (for/list ([p (in-bytes pixels)])
           (real->floating-point-bytes (/ p 255.0) 4 (system-big-endian?)))))

(define (floats->list floats)
  (for/list ([start (in-range 0 (bytes-length floats) 4)])
    (floating-point-bytes->real floats (system-big-endian?) start (+ start 4))))

(define (shift pixels dx dy)
  (apply bytes
         (for*/list ([y (in-range 28)] [x (in-range 28)])
           (define sx (- x dx))
           (define sy (- y dy))
           (if (and (< -1 sx 28) (< -1 sy 28))
               (bytes-ref pixels (+ sx (* 28 sy)))
               0))))

(define (majority-share labels)
  (define counts
    (for/fold ([counts (hash)]) ([y (in-list labels)])
      (hash-update counts y add1 0)))
  (/ (apply max (hash-values counts)) (length labels)))

(module+ test
  (require rackunit)
  (define-values (xs ys) (load-mnist-fixture))
  (define images
    (let ([raw (~> xs (* 255) (+ 0.5) (to-dtype 'uint8) tensor->bytes)])
      (for/list ([start (in-range 0 (bytes-length raw) 784)])
        (subbytes raw start (+ start 784)))))
  (define every-value
    (apply bytes (for/list ([k (in-range 784)]) (modulo (* 7 k) 256))))

  (for ([pixels (in-list (cons every-value images))])
    (check-equal? (pgm-decode (pgm-encode pixels)) pixels))
  (check-equal? (pgm-decode (regexp-replace #px#"\\s+$" (pgm-encode every-value) #""))
                every-value
                "a PGM with no trailing whitespace lost its last pixel")
  (for ([pixels (in-list (list every-value (car images)))])
    (check-equal? (jitter pixels 0 0 0) (pixels->floats pixels)
                  "angle 0, shift 0 did not return the decoded image")
    (check-equal? (jitter pixels 0 1 2) (pixels->floats (shift pixels 1 2))
                  "a shift of (1, 2) did not move every pixel one right, two down")
    (check-within (floats->list (jitter pixels pi 0 0))
                  (~> pixels bytes->list reverse list->bytes pixels->floats floats->list)
                  1e-6
                  "a half turn did not move every pixel to its mirror through the centre")
    (for ([dx+dy (in-list '((0.5 0) (0 0.5)))]
          [neighbour (in-list (list (shift pixels 1 0) (shift pixels 0 1)))])
      (check-within (floats->list (apply jitter pixels 0 dx+dy))
                    (map (lambda (a b) (/ (+ a b) 2))
                         (~> pixels pixels->floats floats->list)
                         (~> neighbour pixels->floats floats->list))
                    1e-6
                    (format "a shift of ~a did not average each pixel with its neighbour"
                            dx+dy))))

  (define rows (run-example #:workers 2 #:device 'cpu))
  (define majority (majority-share (tensor->list ys)))
  (check-equal? (map epoch-result-workers rows) '(0 2))
  (for ([row (in-list rows)])
    (match-define (epoch-result n loss seconds acc) row)
    (check-true (and (rational? loss) (not (nan? loss)))
                (format "workers ~a: non-finite loss ~a" n loss))
    (check-true (positive? seconds))
    (check-true (and (> acc (+ majority 0.1)) (<= acc 1.0))
                (format "workers ~a: accuracy ~a, majority class ~a" n acc majority)))
  (match rows
    [(list (epoch-result _ first-loss _ _) (epoch-result _ second-loss _ _))
     (check-true (< second-loss first-loss)
                 (format "the second epoch's loss did not fall: ~a" rows))])

  (define ds (encoded-digits xs ys))
  (check-equal? (length ds) 256)
  (check-equal? (first-labels ds 0) (first-labels ds 2)
                "the batch order depends on the worker count")
  (define (first-images workers)
    (match-define (list xb _) (first-batch ds #:workers workers))
    (check-equal? (shape xb) '(8 1 28 28))
    (check-equal? (dtype xb) 'float32)
    (tensor->list xb))
  (check-equal? (first-images 2) (first-images 2)
                "a seeded loader did not replay its jitter")
  (match-define (list plain _)
    (first-batch (tensor-dataset xs ys) #:collate default-collate))
  (match-define (list jittered _)
    (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
      (random-seed 0)
      (first-batch ds)))
  (match-define (list order _)
    (~> (length ds)
        (arange #:dtype 'int64)
        (tensor-dataset ys)
        (first-batch #:collate default-collate)))
  (define replayed
    (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
      (random-seed 0)
      (for/list ([i (in-list (tensor->list order))])
        (apply jitter (list-ref images i) (jitter-draws)))))
  (check-equal? (tensor->bytes jittered) (apply bytes-append replayed)
                "the batch is not each image jittered by its own draws, in order")
  (check-not-equal? (tensor->list jittered) (tensor->list plain)
                    "the jitter left every image in place"))

(module+ test
  ;; Parity against examples/python/17_workers.py: the PGM bytes, the jitter
  ;; at the twin's fixed angles and shifts against torch's grid_sample, then
  ;; five full-batch steps on the fixture jittered image by image.
  (require (only-in file/sha1 hex-string->bytes)
           (only-in torch/nn adam cross-entropy parameters step! zero-grads!)
           (only-in torch/scribblings/examples/05-mnist convnet)
           (only-in torch/tests/private/python-env
                    check-training-twin python-result python-torch-available? tol))

  (define (worst-difference ours theirs)
    (for/fold ([worst 0.0]) ([a (in-list ours)] [b (in-list theirs)])
      (max worst (abs (- a b)))))

  (define (check-jitter pixels motion theirs what)
    (define difference
      (worst-difference (floats->list (apply jitter pixels motion)) theirs))
    (check-true (<= difference 1e-6)
                (format "~a jittered by ~a: max |difference| ~a from torch's"
                        what motion difference)))

  (define (check-parity)
    (define j (python-result "python/17_workers.py"))
    (define motions (hash-ref j 'cases))
    (define (motion-for i) (list-ref motions (modulo i (length motions))))
    (check-equal? (apply bytes-append (map pgm-encode (cons every-value images)))
                  (hex-string->bytes (hash-ref j 'pgm))
                  "the PGM bytes differ from the twin's")
    (for ([loose (in-list (hash-ref j 'loose_pgms))]
          [pixels (in-list (cons every-value images))])
      (check-equal? (pgm-decode (hex-string->bytes loose)) pixels
                    "a PGM spaced with tabs and CRLF decoded differently"))
    (for ([motion (in-list motions)]
          [theirs (in-list (hash-ref j 'pattern_jitters))])
      (check-jitter every-value motion theirs "the pattern"))
    (for ([pixels (in-list images)]
          [theirs (in-list (hash-ref j 'image_jitters))]
          [i (in-naturals)])
      (check-jitter pixels (motion-for i) theirs (format "image ~a" i)))
    (define (train-on device)
      (with-default-device device
        (manual-seed! 0)
        (define net (convnet))
        (define opt (adam (parameters net) #:lr 0.001))
        (define-values (xb yb)
          (collate-digits
           (for/list ([pixels (in-list images)]
                      [y (in-list (tensor->list ys))]
                      [i (in-naturals)])
             (list (apply jitter (~> pixels pgm-encode pgm-decode) (motion-for i))
                   y))))
        (define batch (to xb device))
        (define labels (to yb device))
        (define losses
          (for/list ([_ (in-range 5)])
            (zero-grads! opt)
            (define loss (cross-entropy (net batch) labels))
            (backward! loss)
            (step! opt)
            (item loss)))
        (values losses
                (cat (for/list ([p (in-list (parameters net))])
                       (reshape p -1))))))
    (check-training-twin "17-workers" "python/17_workers.py" train-on 'cpu tol))

  (cond
    [(not (python-torch-available?))
     (displayln "[17-workers] parity skipped: python3 `torch` not available")]
    [else (check-parity)]))
