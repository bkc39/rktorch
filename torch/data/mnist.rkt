#lang racket/base

;; MNIST: a pure-Racket IDX reader (bytes -> float32 image tensors / int64
;; label tensors) plus a download-with-cache helper for the full dataset. A
;; small 256-image fixture is committed under data/fixtures/ so the tests and
;; the convnet smoke train run offline; load-mnist fetches the real dataset
;; (cached) for local training.

;; whole-module on purpose: define-runtime-path expands into phase-1 code
;; that needs bindings (#%datum, ...) only-in would strip (the documented
;; runtime-path exemption to the only-in convention).
(require racket/runtime-path
         (only-in racket/file file->bytes make-directory* make-temporary-file)
         (only-in racket/port copy-port)
         (only-in net/url string->url get-pure-port)
         (only-in file/gunzip gunzip-through-ports)
         (only-in "../main.rkt" tensor reshape to-dtype))

(provide read-idx
         idx->images
         idx->labels
         load-mnist-fixture
         download-cached
         load-mnist)

;; Parse an IDX (uint8) buffer into (values dims data-bytes). The header is
;; 0x00 0x00 <dtype> <ndim> then ndim big-endian int32 dimension sizes; we
;; only handle the uint8 dtype (0x08) MNIST uses.
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

;; IDX images (N x H x W uint8) -> an [N, 1, H, W] float32 tensor scaled to
;; [0, 1], the NCHW layout conv2d wants.
(define (idx->images bs)
  (define-values (dims data) (read-idx bs))
  (define floats
    (for/list ([b (in-bytes data)]) (/ (exact->inexact b) 255.0)))
  (apply reshape (tensor floats)
         (list (car dims) 1 (cadr dims) (caddr dims))))

;; IDX labels (N uint8) -> an [N] int64 tensor of class indices.
(define (idx->labels bs)
  (define-values (_dims data) (read-idx bs))
  (to-dtype (tensor (for/list ([b (in-bytes data)]) b)) 'int64))

;; --- committed fixture (offline) ----------------------------------------
(define-runtime-path images-fixture "fixtures/mnist-256-images-idx3-ubyte")
(define-runtime-path labels-fixture "fixtures/mnist-256-labels-idx1-ubyte")

;; (values images labels) for the committed 256-image fixture.
(define (load-mnist-fixture)
  (values (idx->images (file->bytes images-fixture))
          (idx->labels (file->bytes labels-fixture))))

;; --- full dataset (download + cache) ------------------------------------
;; The mirror PyTorch's torchvision uses (yann.lecun.com is gone).
(define mnist-mirror "https://ossci-datasets.s3.amazonaws.com/mnist/")

(define (mnist-cache-dir)
  (build-path (find-system-path 'cache-dir) "rktorch" "mnist"))

;; Fetch <name> (a .gz IDX file) into the cache once, then return its
;; gunzipped bytes.
(define (download-cached name)
  (define dest (build-path (mnist-cache-dir) name))
  (unless (file-exists? dest)
    (make-directory* (mnist-cache-dir))
    ;; Download to a temp file in the cache dir, then atomically rename on
    ;; success. So an interrupted/failed fetch can't leave a partial file at
    ;; `dest` that file-exists? would treat as a valid cache entry. dynamic-wind
    ;; guarantees the input port closes and the temp file is removed on error.
    (define tmp (make-temporary-file "mnist-~a.part" #f (mnist-cache-dir)))
    (define in (get-pure-port (string->url (string-append mnist-mirror name))))
    (dynamic-wind
     void
     (lambda ()
       (call-with-output-file tmp #:exists 'truncate
         (lambda (out) (copy-port in out)))
       (rename-file-or-directory tmp dest #t))
     (lambda ()
       (close-input-port in)
       (when (file-exists? tmp) (delete-file tmp)))))
  (define out (open-output-bytes))
  (gunzip-through-ports (open-input-bytes (file->bytes dest)) out)
  (get-output-bytes out))

;; (values images labels) for the real dataset; split is 'train or 'test.
(define (load-mnist [split 'train])
  (define prefix (if (eq? split 'test) "t10k" "train"))
  (values
   (idx->images (download-cached (string-append prefix "-images-idx3-ubyte.gz")))
   (idx->labels (download-cached (string-append prefix "-labels-idx1-ubyte.gz")))))
