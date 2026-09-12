#lang racket/base

(require (only-in ffi/vector
                  f32vector-set! make-f32vector make-s64vector s64vector-set!)
         (only-in file/gunzip gunzip-through-ports)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/contract/base -> ->* any/c cons/c listof or/c)
         (only-in racket/file file->bytes make-directory*)
         (only-in racket/port copy-port)
         ;; whole-module on purpose: the expansion needs bindings only-in
         ;; would strip
         racket/runtime-path
         (only-in "../data/loader.rkt" dataset? tensor-dataset)
         (only-in "../main.rkt" device/c reshape tensor tensor? to)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/util.rkt" with-temporary-file))

(define record-size 3073)
(define image-size 3072)

(define/contract-out cifar10-label-names (listof string?) ;; noqa
  '("airplane" "automobile" "bird" "cat" "deer"
    "dog" "frog" "horse" "ship" "truck"))

;; a record is one label byte, then the red, green and blue planes of a
;; 32x32 image: channel-major already, so the buffer reshapes to NCHW
(define/contract-out (cifar10-records->tensors bs) ;; noqa
  (-> bytes? (values tensor? tensor?))
  (define n (quotient (bytes-length bs) record-size))
  (unless (= (bytes-length bs) (* n record-size))
    (raise-arguments-error 'cifar10-records->tensors
                           "not a whole number of 3073-byte records"
                           "bytes" (bytes-length bs)))
  (define images (make-f32vector (* n image-size)))
  (define labels (make-s64vector n))
  (for ([r (in-range n)])
    (define at (* r record-size))
    (s64vector-set! labels r (bytes-ref bs at))
    (define base (* r image-size))
    (for ([k (in-range image-size)])
      (f32vector-set! images (+ base k)
                      (- (/ (bytes-ref bs (+ at 1 k)) 127.5) 1.0))))
  (values (reshape (tensor images) n 3 32 32) (tensor labels)))

(define (octal-field bs start len)
  (define s (bytes->string/latin-1 (subbytes bs start (+ start len))))
  (define digits (regexp-match #rx"^[ 0]*([0-7]*)" s))
  (define text (cadr digits))
  (if (string=? text "") 0 (string->number text 8)))

(define (name-field bs start len)
  (define raw (subbytes bs start (+ start len)))
  (define end (or (for/first ([i (in-range len)] #:when (zero? (bytes-ref raw i))) i)
                  len))
  (bytes->string/utf-8 (subbytes raw 0 end)))

;; the entries of an uncompressed tar: name and contents per regular file
(define/contract-out (tar-entries bs) ;; noqa
  (-> bytes? (listof (cons/c string? bytes?)))
  (let loop ([at 0] [acc '()])
    (cond
      [(> (+ at 512) (bytes-length bs)) (reverse acc)]
      [(for/and ([i (in-range at (+ at 512))]) (zero? (bytes-ref bs i)))
       (reverse acc)]
      [else
       (define size (octal-field bs (+ at 124) 12))
       (define type (bytes-ref bs (+ at 156)))
       (define prefix (name-field bs (+ at 345) 155))
       (define base (name-field bs at 100))
       (define name (if (string=? prefix "") base (string-append prefix "/" base)))
       (define data-at (+ at 512))
       (define next (+ data-at (* 512 (quotient (+ size 511) 512))))
       (define regular? (or (= type 0) (= type (char->integer #\0))))
       (loop next
             (if regular?
                 (cons (cons name (subbytes bs data-at (+ data-at size))) acc)
                 acc))])))

(define cifar10-mirror "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
(define archive-name "cifar-10-binary.tar.gz")

(define (cifar10-cache-dir)
  (define override (getenv "RKTORCH_CIFAR10_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "cifar10")))

(define (archive-path)
  (build-path (cifar10-cache-dir) archive-name))

(define/contract-out (cifar10-cached?) ;; noqa
  (-> boolean?)
  (file-exists? (archive-path)))

(define (download-cached)
  (define dest (archive-path))
  (unless (file-exists? dest)
    (make-directory* (cifar10-cache-dir))
    ;; temp file + atomic rename: an interrupted fetch must not poison the cache
    (with-temporary-file (tmp #:template "cifar10-~a.part"
                              #:directory (cifar10-cache-dir))
      (call/input-url (string->url cifar10-mirror)
                      (lambda (url) (get-pure-port url #:redirections 5))
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))
                        (unless (gzip-file? tmp)
                          (raise-arguments-error 'load-cifar10
                                                 "the mirror did not answer with the archive"
                                                 "url" cifar10-mirror))
                        (rename-file-or-directory tmp dest #t)))))
  dest)

(define (gzip-file? path)
  (define head (call-with-input-file path (lambda (in) (read-bytes 2 in))))
  (equal? head #"\37\213"))

(define (basename name)
  (define parts (regexp-split #rx"/" name))
  (car (reverse parts)))

;; every file of the archive by its base name, unpacked in memory
(define/contract-out (cifar10-archive-files) ;; noqa
  (-> (listof (cons/c string? bytes?)))
  (define out (open-output-bytes))
  (gunzip-through-ports (open-input-bytes (file->bytes (download-cached))) out)
  (for/list ([entry (in-list (tar-entries (get-output-bytes out)))])
    (cons (basename (car entry)) (cdr entry))))

(define (archive-file files name)
  (define entry (assoc name files))
  (unless entry
    (raise-arguments-error 'load-cifar10 "file missing from the archive"
                           "name" name))
  (cdr entry))

(define/contract-out (load-cifar10 [split 'train]) ;; noqa
  (->* [] [(or/c 'train 'test)] (values tensor? tensor?))
  (define files (cifar10-archive-files))
  (define names
    (if (eq? split 'test)
        '("test_batch.bin")
        (for/list ([i (in-range 1 6)]) (format "data_batch_~a.bin" i))))
  (cifar10-records->tensors
   (apply bytes-append (for/list ([name (in-list names)])
                         (archive-file files name)))))

(define/contract-out (cifar10-dataset [split 'train] #:device [device #f]) ;; noqa
  (->* [] [(or/c 'train 'test) #:device (or/c #f device/c)] dataset?)
  (define-values (images labels) (load-cifar10 split))
  (if device
      (tensor-dataset (to images device) (to labels device))
      (tensor-dataset images labels)))

(define-runtime-path fixture "fixtures/cifar10-256.bin")

(define/contract-out (load-cifar10-fixture) ;; noqa
  (-> (values tensor? tensor?))
  (cifar10-records->tensors (file->bytes fixture)))
