#lang racket/base

(module+ test
  (require (only-in racket/file delete-directory/files make-temporary-file)
           (only-in racket/list argmin group-by range remove-duplicates)
           (only-in racket/match match-define)
           (only-in racket/sequence in-slice)
           rackunit
           (only-in "../data/loader.rkt" dataset-ref)
           (only-in "../main.rkt"
                    cuda-available? cuda-device device dtype length
                    mps-available? mps-device ref shape tensor->list)
           (only-in "../vision/cifar10.rkt"
                    cifar10-archive-files
                    cifar10-cached?
                    cifar10-dataset
                    cifar10-label-names
                    cifar10-records->tensors
                    load-cifar10-fixture
                    tar-entries)
           "private/class-means.rkt")

  (define-values (imgs lbls) (load-cifar10-fixture))
  (define labels (tensor->list lbls))
  (define records
    (for/list ([record (in-slice 3072 (tensor->list imgs))])
      (list->vector record)))

  (test-case "the committed fixture parses to NCHW in [-1, 1] and int64 labels"
    (check-equal? (shape imgs) '(256 3 32 32))
    (check-equal? (shape lbls) '(256))
    (check-equal? (dtype lbls) 'int64)
    (define all (tensor->list imgs))
    (check-true (>= (apply min all) -1.0) "pixel below -1")
    (check-true (<= (apply max all) 1.0) "pixel above 1")
    (check-true (< (apply min all) 0.0) "no dark pixel"))

  (test-case "every one of the ten classes is drawn, and only those"
    (check-equal? (sort (remove-duplicates labels) <) (range 10)))

  (define (channel-means record)
    (for/list ([c (in-range 3)])
      (/ (for/sum ([k (in-range (* c 1024) (* (add1 c) 1024))])
           (vector-ref record k))
         1024)))

  (test-case "each class is a colour: a record's mean colour is nearest its own class's"
    (define colours (map channel-means records))
    (define means (class-means colours labels))
    (for ([colour (in-list colours)] [label (in-list labels)] [i (in-naturals)])
      (check-equal? (nearest-class means colour squared-distance) label
                    (format "record ~a" i))))

  (define shifts
    (for*/list ([dy (in-range 4)]
                [dx (in-range -3 4)]
                #:when (or (positive? dy) (positive? dx))
                #:when (= 1 (gcd dx dy)))
      (cons dx dy)))

  (define (shift-difference record shift)
    (match-define (cons dx dy) shift)
    (define pairs
      (for*/list ([c (in-range 3)]
                  [y (in-range (- 32 dy))]
                  [x (in-range (max 0 (- dx)) (min 32 (- 32 dx)))])
        (define (px y x) (vector-ref record (+ (* c 1024) (* y 32) x)))
        (abs (- (px y x) (px (+ y dy) (+ x dx))))))
    (/ (apply + pairs) (length pairs)))

  (define (stripe-shift record)
    (argmin (lambda (shift) (shift-difference record shift)) shifts))

  (test-case "each class is a stripe direction: one per class, a different one for each"
    (define by-class
      (group-by car (map cons labels (map stripe-shift records))))
    (define per-class (map (lambda (g) (remove-duplicates (map cdr g))) by-class))
    (for ([g (in-list per-class)])
      (check-equal? (length g) 1 (format "a class with stripes along ~a" g)))
    (check-equal? (length (remove-duplicates (map car per-class))) 10))

  (test-case "records land on the requested device"
    (for ([dev (in-list (list (and (cuda-available?) (cuda-device))
                              (and (mps-available?) (mps-device))))]
          #:when dev)
      (define-values (imgs lbls)
        (cifar10-records->tensors (make-bytes (* 2 3073) 9) #:device dev))
      (check-equal? (device imgs) dev)
      (check-equal? (device lbls) dev)
      (check-equal? (shape imgs) '(2 3 32 32))))

  (test-case "records must come whole"
    (check-exn #rx"whole number of 3073-byte records"
               (lambda () (cifar10-records->tensors (make-bytes 100 0))))
    (define-values (imgs lbls) (cifar10-records->tensors (make-bytes 3073 7)))
    (check-equal? (shape imgs) '(1 3 32 32))
    (check-equal? (tensor->list lbls) '(7)))

  (define (tar-header name size type)
    (define h (make-bytes 512 0))
    (bytes-copy! h 0 (string->bytes/utf-8 name))
    (bytes-copy! h 100 #"0000644\0")
    (bytes-copy! h 124 (string->bytes/utf-8 (string-append (number->string size 8) "\0")))
    (bytes-set! h 156 (char->integer type))
    (bytes-copy! h 257 #"ustar\0")
    h)
  (define (padded bs)
    (bytes-append bs (make-bytes (modulo (- 512 (modulo (bytes-length bs) 512)) 512) 0)))

  (test-case "tar-entries reads regular files and skips directories"
    (define archive
      (bytes-append (tar-header "batches/" 0 #\5)
                    (tar-header "batches/a.bin" 5 #\0) (padded #"hello")
                    (tar-header "batches/b.txt" 600 #\0) (padded (make-bytes 600 65))
                    (make-bytes 1024 0)))
    (define entries (tar-entries archive))
    (check-equal? (map car entries) '("batches/a.bin" "batches/b.txt"))
    (check-equal? (cdr (car entries)) #"hello")
    (check-equal? (bytes-length (cdr (cadr entries))) 600)
    (check-equal? (tar-entries (make-bytes 1024 0)) '()))

  (define (string-trim-newlines s)
    (regexp-replace #rx"\n+$" s ""))

  (test-case "a cache holding anything but the whole archive is refused"
    (define dir (make-temporary-file "cifar10-cache-~a" 'directory))
    (define fake (build-path dir "cifar-10-binary.tar.gz"))
    (call-with-output-file fake
      (lambda (out) (write-bytes #"<!DOCTYPE HTML>" out)))
    (parameterize ([current-environment-variables
                    (environment-variables-copy (current-environment-variables))])
      (putenv "RKTORCH_CIFAR10_DIR" (path->string dir))
      (check-true (cifar10-cached?))
      (check-exn #rx"not the CIFAR-10 binary archive, or cut short"
                 (lambda () (cifar10-archive-files))))
    (delete-directory/files dir))

  (define files (and (cifar10-cached?) (cifar10-archive-files)))
  (cond
    [files
     (test-case "the archive carries five training batches, a test batch and the names"
       (check-equal? (sort (map car files) string<?)
                     '("batches.meta.txt" "data_batch_1.bin" "data_batch_2.bin"
                       "data_batch_3.bin" "data_batch_4.bin" "data_batch_5.bin"
                       "readme.html" "test_batch.bin"))
       (check-equal? (regexp-split #rx"\n" (string-trim-newlines
                                            (bytes->string/utf-8
                                             (cdr (assoc "batches.meta.txt" files)))))
                     cifar10-label-names)
       (define ds (cifar10-dataset 'test))
       (check-equal? (length ds) 10000)
       (define-values (x y) (dataset-ref ds 0))
       (check-equal? (shape x) '(3 32 32))
       (check-equal? (tensor->list y) '(3) "the first test image is a cat"))
     (displayln "[cifar10-test] archive OK (5 x 10000 train, 10000 test)")]
    [else
     (displayln "[cifar10-test] skipped archive (not cached; load-cifar10 fetches it)")]))
