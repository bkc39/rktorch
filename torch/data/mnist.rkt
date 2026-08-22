#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in racket/file file->bytes make-directory*)
         (only-in racket/port copy-port)
         (only-in net/url string->url get-pure-port call/input-url)
         (only-in file/gunzip gunzip-through-ports)
         (only-in "../private/util.rkt" with-temporary-file)
         (only-in "../main.rkt" tensor reshape to-dtype))

(provide read-idx
         idx->images
         idx->labels
         load-mnist-fixture
         download-cached
         load-mnist)

(define (read-idx bs)
  (unless (and (>= (bytes-length bs) 4)
               (zero? (bytes-ref bs 0))
               (zero? (bytes-ref bs 1))
               (= 8 (bytes-ref bs 2)))
    (error 'read-idx "not a uint8 IDX buffer"))
  (define ndim (bytes-ref bs 3))
  (define dims
    (for/list ([i (in-range ndim)])
      (integer-bytes->integer bs #t #t (+ 4 (* i 4)) (+ 8 (* i 4)))))
  (values dims (subbytes bs (+ 4 (* ndim 4)))))

(define (idx->images bs)
  (define-values (dims data) (read-idx bs))
  (define floats
    (for/list ([b (in-bytes data)]) (/ (exact->inexact b) 255.0)))
  (apply reshape (tensor floats)
         (list (car dims) 1 (cadr dims) (caddr dims))))

(define (idx->labels bs)
  (define-values (_dims data) (read-idx bs))
  (to-dtype (tensor (for/list ([b (in-bytes data)]) b)) 'int64))

(define-runtime-path images-fixture "fixtures/mnist-256-images-idx3-ubyte")
(define-runtime-path labels-fixture "fixtures/mnist-256-labels-idx1-ubyte")

(define (load-mnist-fixture)
  (values (idx->images (file->bytes images-fixture))
          (idx->labels (file->bytes labels-fixture))))

;; the mirror torchvision uses (yann.lecun.com is gone)
(define mnist-mirror "https://ossci-datasets.s3.amazonaws.com/mnist/")

(define (mnist-cache-dir)
  (define override (getenv "RKTORCH_MNIST_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "mnist")))

(define (download-cached name)
  (define dest (build-path (mnist-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (mnist-cache-dir))
    ;; temp file + atomic rename: an interrupted fetch must not poison the cache
    (with-temporary-file (tmp #:template "mnist-~a.part"
                              #:directory (mnist-cache-dir))
      (call/input-url (string->url (string-append mnist-mirror name))
                      get-pure-port
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))
                        (rename-file-or-directory tmp dest #t)))))
  (define out (open-output-bytes))
  (gunzip-through-ports (open-input-bytes (file->bytes dest)) out)
  (get-output-bytes out))

(define (load-mnist [split 'train])
  (define prefix (if (eq? split 'test) "t10k" "train"))
  (values
   (idx->images (download-cached (string-append prefix "-images-idx3-ubyte.gz")))
   (idx->labels (download-cached (string-append prefix "-labels-idx1-ubyte.gz")))))
