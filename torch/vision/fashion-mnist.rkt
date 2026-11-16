#lang racket/base

(require (only-in file/gunzip gunzip-through-ports)
         (only-in file/md5 md5)
         (only-in racket/contract/base -> ->* listof or/c)
         (only-in racket/file file->bytes)
         (only-in racket/match match-define)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../data/loader.rkt" dataset? tensor-dataset)
         (only-in "../main.rkt"
                  cpu-device default-device device/c tensor? to-device with-default-device)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/download.rkt" download-cached)
         (only-in "../private/util.rkt" cache-dir)
         (only-in "mnist.rkt" idx->images idx->labels))

(module+ private
  (provide archive-valid?
           fashion-mnist-cache-dir
           fashion-mnist-idx
           fashion-mnist-resources
           fixture-files))

(define/contract-out fashion-mnist-classes (listof string?) ;; noqa
  '("T-shirt/top" "Trouser" "Pullover" "Dress" "Coat"
    "Sandal" "Shirt" "Sneaker" "Bag" "Ankle boot"))

(define fashion-mnist-mirror "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/")

(define fashion-mnist-resources
  '(("train-images-idx3-ubyte.gz" . "8d4fb7e6c68d591d4c3dfef9ec88bf0d")
    ("train-labels-idx1-ubyte.gz" . "25c81989df183df01b3e8a0aad5dffbe")
    ("t10k-images-idx3-ubyte.gz" . "bef4ecab320f06d8554ea6380940ec79")
    ("t10k-labels-idx1-ubyte.gz" . "bb300cfdad3c16e7a12a480ee83cd310")))

(define splits
  '((train "train" 512)
    (test "t10k" 128)))

(define kinds '("images-idx3" "labels-idx1"))

(define (archive-name prefix kind)
  (format "~a-~a-ubyte.gz" prefix kind))

(define (fixture-name prefix n kind)
  (format "~a-~a-~a-ubyte" prefix n kind))

(define fixture-files
  (for*/list ([split (in-list splits)] [kind (in-list kinds)])
    (match-define (list _ prefix n) split)
    (list (archive-name prefix kind) (fixture-name prefix n kind) n)))

(define (fashion-mnist-cache-dir)
  (cache-dir "RKTORCH_FASHION_MNIST_DIR" "fashion-mnist"))

(define (gunzipped bs)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (define out (open-output-bytes))
    (gunzip-through-ports (open-input-bytes bs) out)
    (get-output-bytes out)))

(define (whole-split? idx)
  (and (>= (bytes-length idx) 8)
       (memv (integer-bytes->integer idx #f #t 4 8) '(60000 10000))))

(define (archive-idx bs name)
  (define expected
    (if name
        (list (cdr (assoc name fashion-mnist-resources)))
        (map cdr fashion-mnist-resources)))
  (define idx
    (and (member (bytes->string/latin-1 (md5 bs)) expected)
         (gunzipped bs)))
  (and idx (whole-split? idx) idx))

(define (archive-valid? path [name #f])
  (bytes? (archive-idx (file->bytes path) name)))

(define (fashion-mnist-idx name)
  (define archive
    (download-cached 'load-fashion-mnist (fashion-mnist-cache-dir) name
                     (string-append fashion-mnist-mirror name)
                     #:valid? (lambda (tmp) (archive-valid? tmp name))))
  (or (archive-idx (file->bytes archive) name)
      (error 'load-fashion-mnist
             "the cached ~a is not torchvision's archive; delete it to fetch it again\n  file: ~a"
             name archive)))

(define/contract-out (fashion-mnist-cached?) ;; noqa
  (-> boolean?)
  (for/and ([entry (in-list fashion-mnist-resources)])
    (file-exists? (build-path (fashion-mnist-cache-dir) (car entry)))))

(define (idx->tensors read device)
  (define-values (xs ys)
    (with-default-device (cpu-device)
      (values (idx->images (read "images-idx3")) (idx->labels (read "labels-idx1")))))
  (define target (or device (default-device)))
  (values (to-device xs target) (to-device ys target)))

(define split/c (or/c 'train 'test))

(define split-tensors/c
  (->* () (split/c #:device (or/c #f device/c)) (values tensor? tensor?)))

(define (split-row split)
  (cdr (assq split splits)))

(define/contract-out (load-fashion-mnist [split 'train] #:device [device #f]) ;; noqa
  split-tensors/c
  (match-define (list prefix _) (split-row split))
  (idx->tensors (lambda (kind) (fashion-mnist-idx (archive-name prefix kind))) device))

(define/contract-out (fashion-mnist-dataset [split 'train] #:device [device #f]) ;; noqa
  (->* () (split/c #:device (or/c #f device/c)) dataset?)
  (define-values (images labels) (load-fashion-mnist split #:device device))
  (tensor-dataset images labels))

(define-runtime-path fixture-dir "fixtures/fashion-mnist")

(define/contract-out (load-fashion-mnist-fixture [split 'train] #:device [device #f]) ;; noqa
  split-tensors/c
  (match-define (list prefix n) (split-row split))
  (idx->tensors (lambda (kind) (file->bytes (build-path fixture-dir (fixture-name prefix n kind))))
                device))
