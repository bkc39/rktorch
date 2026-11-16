#lang racket/base

(require (only-in file/gunzip gunzip-through-ports)
         (only-in file/md5 md5)
         (only-in racket/contract/base -> ->* listof or/c)
         (only-in racket/file file->bytes)
         (only-in racket/match match)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../data/loader.rkt" dataset? tensor-dataset)
         (only-in "../main.rkt" ~> default-device device/c tensor? with-default-device)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/download.rkt" download-cached)
         (only-in "../private/util.rkt" cache-dir)
         (only-in "mnist.rkt" idx->images idx->labels))

(module+ private
  (provide archive-valid?
           fashion-mnist-idx
           fashion-mnist-resources))

(define/contract-out fashion-mnist-classes (listof string?) ;; noqa
  '("T-shirt/top" "Trouser" "Pullover" "Dress" "Coat"
    "Sandal" "Shirt" "Sneaker" "Bag" "Ankle boot"))

(define fashion-mnist-mirror "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")

(define fashion-mnist-resources
  '(("train-images-idx3-ubyte.gz" . "8d4fb7e6c68d591d4c3dfef9ec88bf0d")
    ("train-labels-idx1-ubyte.gz" . "25c81989df183df01b3e8a0aad5dffbe")
    ("t10k-images-idx3-ubyte.gz" . "bef4ecab320f06d8554ea6380940ec79")
    ("t10k-labels-idx1-ubyte.gz" . "bb300cfdad3c16e7a12a480ee83cd310")))

(define (fashion-mnist-cache-dir)
  (cache-dir "RKTORCH_FASHION_MNIST_DIR" "fashion-mnist"))

(define (gunzipped bs)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (define out (open-output-bytes))
    (gunzip-through-ports (open-input-bytes bs) out)
    (get-output-bytes out)))

(define (whole-split? idx)
  (and (>= (bytes-length idx) 8)
       (memv (integer-bytes->integer idx #f #t 4 8) '(60000 10000))
       #t))

(define (archive-valid? path [name #f])
  (define bs (file->bytes path))
  (define digest (bytes->string/latin-1 (md5 bs)))
  (define expected
    (if name
        (list (cdr (assoc name fashion-mnist-resources)))
        (map cdr fashion-mnist-resources)))
  (and (member digest expected)
       (match (gunzipped bs)
         [#f #f]
         [idx (whole-split? idx)])))

(define (fashion-mnist-idx name)
  (define archive
    (download-cached 'load-fashion-mnist (fashion-mnist-cache-dir) name
                     (string-append fashion-mnist-mirror name)
                     #:valid? (lambda (tmp) (archive-valid? tmp name))))
  (gunzipped (file->bytes archive)))

(define/contract-out (fashion-mnist-cached?) ;; noqa
  (-> boolean?)
  (for/and ([entry (in-list fashion-mnist-resources)])
    (file-exists? (build-path (fashion-mnist-cache-dir) (car entry)))))

(define (prefix-of split)
  (match split
    ['train "train"]
    ['test "t10k"]))

(define (idx->tensors images labels device)
  (with-default-device (or device (default-device))
    (values (idx->images images) (idx->labels labels))))

(define/contract-out (load-fashion-mnist [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) (values tensor? tensor?))
  (define prefix (prefix-of split))
  (idx->tensors (fashion-mnist-idx (string-append prefix "-images-idx3-ubyte.gz"))
                (fashion-mnist-idx (string-append prefix "-labels-idx1-ubyte.gz"))
                device))

(define/contract-out (fashion-mnist-dataset [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) dataset?)
  (define-values (images labels) (load-fashion-mnist split #:device device))
  (tensor-dataset images labels))

(define-runtime-path fixture-dir "fixtures/fashion-mnist")

(define/contract-out (load-fashion-mnist-fixture [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) (values tensor? tensor?))
  (define-values (prefix n)
    (match split
      ['train (values "train" 512)]
      ['test (values "t10k" 128)]))
  (define (fixture kind)
    (~> (format "~a-~a-~a-ubyte" prefix n kind) (build-path fixture-dir _) file->bytes))
  (idx->tensors (fixture "images-idx3") (fixture "labels-idx1") device))
