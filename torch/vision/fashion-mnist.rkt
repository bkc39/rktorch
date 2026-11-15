#lang racket/base

(require (only-in file/gunzip gunzip-through-ports)
         (only-in file/md5 md5)
         (only-in racket/contract/base -> ->* listof or/c)
         (only-in racket/file file->bytes)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "../data/loader.rkt" dataset? tensor-dataset)
         (only-in "../main.rkt" default-device device/c tensor? with-default-device)
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

(define (archive-valid? path [name #f])
  (error 'archive-valid? "unimplemented"))

(define (fashion-mnist-idx name)
  (error 'fashion-mnist-idx "unimplemented"))

(define/contract-out (fashion-mnist-cached?) ;; noqa
  (-> boolean?)
  (error 'fashion-mnist-cached? "unimplemented"))

(define/contract-out (load-fashion-mnist [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) (values tensor? tensor?))
  (error 'load-fashion-mnist "unimplemented"))

(define/contract-out (fashion-mnist-dataset [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) dataset?)
  (error 'fashion-mnist-dataset "unimplemented"))

(define-runtime-path fixture-dir "fixtures/fashion-mnist")

(define/contract-out (load-fashion-mnist-fixture [split 'train] #:device [device #f]) ;; noqa
  (->* () ((or/c 'train 'test) #:device (or/c #f device/c)) (values tensor? tensor?))
  (error 'load-fashion-mnist-fixture "unimplemented"))
