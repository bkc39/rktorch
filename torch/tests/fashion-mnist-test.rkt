#lang racket/base

(module+ test
  (require (only-in file/gzip gzip-through-ports)
           (only-in racket/file file->bytes)
           (only-in racket/list range remove-duplicates)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../data/loader.rkt" dataset-length)
           (only-in "../main.rkt"
                    * + ~> accelerator-if-available cpu-device device device-type dtype
                    ref reshape shape tensor->list to-dtype to-device)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../vision/fashion-mnist.rkt"
                    fashion-mnist-cached? fashion-mnist-classes fashion-mnist-dataset
                    load-fashion-mnist load-fashion-mnist-fixture)
           (only-in (submod "../vision/fashion-mnist.rkt" private)
                    archive-valid? fashion-mnist-cache-dir fashion-mnist-idx
                    fashion-mnist-resources fixture-files)
           (only-in "../vision/private/idx.rkt" idx-head)
           (only-in "../vision/mnist.rkt" read-idx)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/python-env.rkt"
                    python-check python-module-available? python-torch-available? unpack))

  (define-runtime-path fixture-dir "../vision/fixtures/fashion-mnist")

  (define fixture-splits
    '((train "train-512-images-idx3-ubyte" "train-512-labels-idx1-ubyte" 512)
      (test "t10k-128-images-idx3-ubyte" "t10k-128-labels-idx1-ubyte" 128)))

  (define (payload name)
    (define-values (_dims data) (~> (build-path fixture-dir name) file->bytes read-idx))
    data)

  (define (record-count bs)
    (define-values (dims _data) (read-idx bs))
    (car dims))

  (define (pixel-bytes xs)
    (~> xs (* 255.0) (+ 0.5) (to-dtype 'uint8) (reshape -1) tensor->list))

  (test-case "G4-L1: the fixture's splits are float32 images in [0, 1] and int64 labels"
    (for ([entry (in-list fixture-splits)])
      (match-define (list split _ _ n) entry)
      (define-values (xs ys) (load-fashion-mnist-fixture split))
      (check-equal? (shape xs) (list n 1 28 28))
      (check-equal? (dtype xs) 'float32)
      (check-equal? (shape ys) (list n))
      (check-equal? (dtype ys) 'int64)
      (define pixels (tensor->list (reshape xs -1)))
      (check-true (<= 0.0 (apply min pixels) (apply max pixels) 1.0))
      (check-true (andmap (lambda (y) (<= 0 y 9)) (tensor->list ys))))
    (define-values (_xs ys) (load-fashion-mnist-fixture))
    (check-equal? (sort (remove-duplicates (tensor->list ys)) <) (range 10)))

  (test-case "the reference's example: the first test picture is an ankle boot"
    (define-values (xs ys) (load-fashion-mnist-fixture 'test))
    (check-equal? (shape xs) '(128 1 28 28))
    (check-equal? (list-ref fashion-mnist-classes (ref ys 0)) "Ankle boot"))

  (test-case "G4-L2: the images and labels give back the fixture's bytes"
    (for ([entry (in-list fixture-splits)])
      (match-define (list split images labels _) entry)
      (define-values (xs ys) (load-fashion-mnist-fixture split))
      (check-equal? (pixel-bytes xs) (bytes->list (payload images)))
      (check-equal? (tensor->list ys) (bytes->list (payload labels)))))

  (define accel (accelerator-if-available))
  (unless (eq? (device-type accel) 'cpu)
    (test-case "G4-L2: #:device places both on the accelerator, with the same values"
      (define-values (xs ys) (load-fashion-mnist-fixture 'test))
      (define-values (dxs dys) (load-fashion-mnist-fixture 'test #:device accel))
      (check-equal? (device dxs) accel)
      (check-equal? (device dys) accel)
      (check-equal? (tensor->list (to-device dxs (cpu-device))) (tensor->list xs))
      (check-equal? (tensor->list (to-device dys (cpu-device))) (tensor->list ys))))

  (define images (file->bytes (build-path fixture-dir "train-512-images-idx3-ubyte")))

  (test-case "G4-L3: idx-head keeps a prefix of the records"
    (check-equal? (idx-head images (record-count images)) images)
    (for* ([m (in-list '(0 1 7 512))] [n (in-list '(0 3 7 300))] #:when (<= n 512))
      (check-equal? (idx-head (idx-head images m) (min m n)) (idx-head images (min m n)))
      (check-equal? (record-count (idx-head images n)) n))
    (check-equal? (bytes-length (idx-head images 3)) (+ 16 (* 3 784)))
    (check-exn (blames-caller? "fashion-mnist-test.rkt") (lambda () (idx-head images 513))))

  (test-case "G4-L3: a buffer that is not IDX, or is cut short, blames the caller"
    (define blamed? (blames-caller? "fashion-mnist-test.rkt"))
    (for ([bs (in-list (list #"" #"not an IDX buffer" (subbytes images 0 6)
                             (bytes-append (subbytes images 0 3) #"\0")))])
      (check-exn blamed? (lambda () (idx-head bs 0)) (format "~s" bs)))
    (check-exn blamed? (lambda () (idx-head (subbytes images 0 (+ 16 (* 3 784) 1)) 4)))
    (check-equal? (idx-head (subbytes images 0 (+ 16 (* 3 784))) 3) (idx-head images 3)))

  (define (gzipped bs)
    (define out (open-output-bytes))
    (gzip-through-ports (open-input-bytes bs) out #f 0)
    (get-output-bytes out))

  (define (valid-bytes? bs)
    (with-temporary-directory (dir)
      (define path (build-path dir "archive.gz"))
      (call-with-output-file path (lambda (port) (write-bytes bs port)))
      (archive-valid? path)))

  (test-case "G4-L4: archive-valid? turns away a gzip that is not torchvision's"
    (check-false (valid-bytes? (gzipped (idx-head images 3))))
    (check-false (valid-bytes? (gzipped images)))
    (check-false (valid-bytes? #"not a gzip")))

  (define (with-cache files proc)
    (with-temporary-directory (dir)
      (for ([file (in-list files)])
        (match-define (cons name bs) file)
        (call-with-output-file (build-path dir name) (lambda (port) (write-bytes bs port))))
      (parameterize ([current-environment-variables
                      (environment-variables-copy (current-environment-variables))])
        (putenv "RKTORCH_FASHION_MNIST_DIR" (path->string dir))
        (proc))))

  (define labels-60000
    (bytes-append #"\0\0\10\1" (integer->integer-bytes 60000 4 #f #t) (make-bytes 60000 0)))

  (test-case "a cached file that is not torchvision's archive is an error naming it"
    (with-cache (list (cons "train-images-idx3-ubyte.gz" #"not a gzip")
                      (cons "train-labels-idx1-ubyte.gz" (gzipped labels-60000)))
      (lambda ()
        (check-exn #rx"train-images-idx3-ubyte[.]gz is not torchvision's archive"
                   (lambda () (load-fashion-mnist)))
        (check-exn #rx"file: .*train-labels-idx1-ubyte[.]gz"
                   (lambda () (fashion-mnist-idx "train-labels-idx1-ubyte.gz"))))))

  (define cache (fashion-mnist-cache-dir))

  (cond
    [(not (fashion-mnist-cached?))
     (displayln "[fashion-mnist-test] full splits skipped: Fashion-MNIST is not cached")]
    [else
     (test-case "G4-L3: the full splits, and the fixture cut from them"
       (define-values (xs ys) (load-fashion-mnist))
       (check-equal? (shape xs) '(60000 1 28 28))
       (check-equal? (shape ys) '(60000))
       (define-values (txs _tys) (load-fashion-mnist 'test))
       (check-equal? (shape txs) '(10000 1 28 28))
       (for ([entry (in-list fixture-files)])
         (match-define (list archive name n) entry)
         (define fixture (file->bytes (build-path fixture-dir name)))
         (check-equal? (record-count fixture) n name)
         (check-equal? (idx-head (fashion-mnist-idx archive) n) fixture name)))
     (test-case "G4-L4: each cached archive is valid, and a flipped byte is not"
       (for ([archive (in-list (map car fashion-mnist-resources))])
         (define path (build-path cache archive))
         (check-true (archive-valid? path) archive)
         (define bs (bytes-copy (file->bytes path)))
         (define at (quotient (bytes-length bs) 2))
         (bytes-set! bs at (bitwise-xor 1 (bytes-ref bs at)))
         (check-false (valid-bytes? bs) archive)))
     (test-case "fashion-mnist-dataset holds the split"
       (check-equal? (dataset-length (fashion-mnist-dataset 'test)) 10000))])

  (cond
    [(not (and (python-torch-available?) (python-module-available? "torchvision")))
     (displayln "[fashion-mnist-test] G4-L5 skipped: python3 torchvision not available")]
    [else
     (test-case "G4-L5: torchvision's FashionMNIST reads the fixture to the same tensors"
       (define j (python-check "fashion_mnist_fixture.py"))
       (check-equal? (hash-ref j 'classes) fashion-mnist-classes)
       (for ([split (in-list '(train test))])
         (define-values (xs ys) (load-fashion-mnist-fixture split))
         (define theirs (hash-ref j split))
         (check-equal? (tensor->list (unpack (hash-ref theirs 'images) 'float32))
                       (tensor->list xs))
         (check-equal? (tensor->list (unpack (hash-ref theirs 'labels) 'int64))
                       (tensor->list ys))))])

  (test-case "G4-L6: a bad split or device blames the caller"
    (define blamed? (blames-caller? "fashion-mnist-test.rkt"))
    (check-exn blamed? (lambda () (load-fashion-mnist 'val)))
    (check-exn blamed? (lambda () (fashion-mnist-dataset "train")))
    (check-exn blamed? (lambda () (load-fashion-mnist-fixture #:device 3)))))
