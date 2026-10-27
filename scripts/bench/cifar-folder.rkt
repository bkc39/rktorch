#lang racket/base

(require (only-in file/gzip deflate)
         (only-in file/sha1 bytes->hex-string)
         (only-in racket/file
                  delete-directory/files file->bytes file->string make-directory*
                  make-temporary-directory)
         (only-in racket/format ~r)
         (only-in racket/string string-join string-trim)
         (only-in threading ~>)
         (only-in torch/vision/cifar10 cifar10-archive-files cifar10-label-names))

(provide cifar10-folder
         cifar10-folder-digest
         ensure-cifar10-folder!)

(define record-size 3073)
(define side 32)
(define plane (* side side))
(define stride (add1 (* 3 side)))

(define (cifar10-folder-base)
  (define override (getenv "RKTORCH_CIFAR10_PNG_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "cifar10-png")))

(define (cifar10-folder dataset #:base [base (cifar10-folder-base)])
  (build-path base dataset))

(define (digest-path root) (build-path root "digest"))

(define (cifar10-folder-digest dataset #:base [base (cifar10-folder-base)])
  (define root (cifar10-folder dataset #:base base))
  (and (file-exists? (digest-path root))
       (~> root digest-path file->string string-trim)))

(define crc-table
  (for/vector #:length 256 ([n (in-range 256)])
    (for/fold ([c n]) ([_ (in-range 8)])
      (if (odd? c)
          (bitwise-xor #xEDB88320 (arithmetic-shift c -1))
          (arithmetic-shift c -1)))))

(define (crc32 bs)
  (bitwise-xor #xFFFFFFFF
               (for/fold ([c #xFFFFFFFF]) ([b (in-bytes bs)])
                 (bitwise-xor (vector-ref crc-table (bitwise-and (bitwise-xor c b) #xFF))
                              (arithmetic-shift c -8)))))

(define (adler32 bs)
  (define-values (a b)
    (for/fold ([a 1] [b 0]) ([x (in-bytes bs)])
      (define a* (modulo (+ a x) 65521))
      (values a* (modulo (+ b a*) 65521))))
  (bitwise-ior (arithmetic-shift b 16) a))

(define (u32 n) (integer->integer-bytes n 4 #f #t))

(define (chunk type data)
  (define typed (bytes-append type data))
  (bytes-append (u32 (bytes-length data)) typed (u32 (crc32 typed))))

(define (zlib bs)
  (define out (open-output-bytes))
  (deflate (open-input-bytes bs) out)
  (bytes-append #"\x78\x9c" (get-output-bytes out) (u32 (adler32 bs))))

(define (scanlines record at)
  (define out (make-bytes (* side stride) 0))
  (for* ([y (in-range side)] [x (in-range side)] [c (in-range 3)])
    (bytes-set! out (+ (* y stride) 1 (* 3 x) c)
                (bytes-ref record (+ at 1 (* c plane) (* y side) x))))
  out)

(define (png-bytes record [at 0])
  (bytes-append #"\x89PNG\r\n\x1a\n"
                (chunk #"IHDR" (bytes-append (u32 side) (u32 side) (bytes 8 2 0 0 0)))
                (chunk #"IDAT" (zlib (scanlines record at)))
                (chunk #"IEND" #"")))

(define (split-records dataset)
  (cond
    [(equal? dataset "fixture")
     (define bs (file->bytes (collection-file-path "fixtures/cifar10-256.bin" "torch" "vision")))
     (list (cons "train" bs) (cons "test" bs))]
    [else
     (define files (cifar10-archive-files))
     (define (joined names) (apply bytes-append (map (lambda (n) (cdr (assoc n files))) names)))
     (list (cons "train" (joined (for/list ([i (in-range 1 6)]) (format "data_batch_~a.bin" i))))
           (cons "test" (joined '("test_batch.bin"))))]))

(define (write-split! dir split bs)
  (for/list ([i (in-range (quotient (bytes-length bs) record-size))])
    (define at (* i record-size))
    (define relative
      (build-path split (list-ref cifar10-label-names (bytes-ref bs at))
                  (format "~a.png" (~r i #:min-width 5 #:pad-string "0"))))
    (define png (png-bytes bs at))
    (define file (build-path dir relative))
    (make-directory* (let-values ([(parent _name _dir?) (split-path file)]) parent))
    (call-with-output-file file (lambda (o) (write-bytes png o)))
    (cons (path->string relative) (bytes->hex-string (sha256-bytes png)))))

(define (write-tree! dir dataset)
  (define sums
    (sort (for*/list ([split (in-list (split-records dataset))]
                      [entry (in-list (write-split! dir (car split) (cdr split)))])
            entry)
          string<? #:key car))
  (define manifest
    (string-join (map (lambda (e) (format "~a  ~a\n" (cdr e) (car e))) sums) ""))
  (call-with-output-file (build-path dir "SHA256SUMS") (lambda (o) (write-string manifest o)))
  (call-with-output-file (digest-path dir)
    (lambda (o) (~> manifest string->bytes/utf-8 sha256-bytes bytes->hex-string (displayln o)))))

(define (ensure-cifar10-folder! dataset #:base [base (cifar10-folder-base)])
  (define root (cifar10-folder dataset #:base base))
  (cond
    [(file-exists? (digest-path root)) (cifar10-folder-digest dataset #:base base)]
    [else
     (make-directory* base)
     (define partial (make-temporary-directory (format ".~a-partial-~~a" dataset) #:base-dir base))
     (with-handlers ([(lambda (_) #t)
                      (lambda (e) (delete-directory/files partial #:must-exist? #f) (raise e))])
       (write-tree! partial dataset)
       (cond
         [(directory-exists? root) (delete-directory/files partial)]
         [else (rename-file-or-directory partial root)]))
     (cifar10-folder-digest dataset #:base base)]))

(module+ main
  (require (only-in racket/cmdline command-line))
  (define dataset
    (command-line #:args ([dataset "full"])
                  (unless (member dataset '("full" "fixture"))
                    (raise-user-error 'cifar-folder "the dataset is full or fixture, got ~s"
                                      dataset))
                  dataset))
  (define digest (ensure-cifar10-folder! dataset))
  (eprintf "cifar-folder: ~a at ~a\n" dataset (cifar10-folder dataset))
  (displayln digest))

(module+ test
  (require (only-in racket/file find-files)
           (only-in racket/list take)
           rackunit
           (only-in torch dtype reshape shape tensor->list to-dtype)
           (only-in torch/vision/image decode-image read-image)
           (only-in torch/vision/image-folder image-folder-classes image-folder-samples)
           (only-in "harness.rkt" with-temporary-path))

  (test-case "the PNG checksums are zlib's and PNG's"
    (check-equal? (crc32 #"IEND") #xAE426082)
    (check-equal? (adler32 #"Wikipedia") #x11E60398))

  (test-case "a record decodes back to its pixels, channel by channel"
    (define record (make-bytes record-size))
    (for ([k (in-range 1 record-size)]) (bytes-set! record k (modulo (* 7 k) 256)))
    (define image (decode-image (png-bytes record) #:mode 'rgb))
    (check-equal? (shape image) '(3 32 32))
    (check-equal? (dtype image) 'uint8)
    (check-equal? (~> image (to-dtype 'int64) (reshape -1) tensor->list)
                  (for/list ([k (in-range 1 record-size)]) (modulo (* 7 k) 256))))

  (test-case "the fixture tree is written once, labelled by class, with its digest"
    (with-temporary-path (base (make-temporary-directory "cifar-folder-~a"))
      (define digest (ensure-cifar10-folder! "fixture" #:base base))
      (define root (cifar10-folder "fixture" #:base base))
      (check-regexp-match #px"^[0-9a-f]{64}$" digest)
      (check-equal? (image-folder-classes (build-path root "train")) cifar10-label-names)
      (define samples (image-folder-samples (build-path root "train")))
      (check-equal? (length samples) 256)
      (define bs (file->bytes (collection-file-path "fixtures/cifar10-256.bin" "torch" "vision")))
      (for ([s (in-list (take samples 3))])
        (define i (~> (car s) path->string (regexp-match #rx"([0-9]+)\\.png$" _) cadr
                      string->number))
        (check-equal? (cdr s) (bytes-ref bs (* i record-size)))
        (check-equal? (~> s car (read-image #:mode 'rgb) (to-dtype 'int64) (reshape -1)
                          tensor->list)
                      (bytes->list (subbytes bs (add1 (* i record-size))
                                             (* (add1 i) record-size)))))
      (define stamp (file-or-directory-modify-seconds (digest-path root)))
      (check-equal? (ensure-cifar10-folder! "fixture" #:base base) digest)
      (check-equal? (file-or-directory-modify-seconds (digest-path root)) stamp)
      (check-equal? (length (find-files (lambda (p) (regexp-match? #rx"partial" (path->string p)))
                                        base))
                    0)
      (with-temporary-path (again (make-temporary-directory "cifar-folder-~a"))
        (check-equal? (ensure-cifar10-folder! "fixture" #:base again) digest
                      "the digest is the tree's, wherever it is written")))))
