#lang racket/base

(module+ test
  (require (only-in racket/list take)
           rackunit
           (only-in "../data/loader.rkt" dataset-ref)
           (only-in "../main.rkt" dtype length ref shape tensor->list)
           (only-in "../vision/cifar10.rkt"
                    cifar10-archive-files
                    cifar10-cached?
                    cifar10-dataset
                    cifar10-label-names
                    cifar10-records->tensors
                    load-cifar10-fixture
                    tar-entries))

  (test-case "the committed fixture parses to NCHW in [-1, 1] and int64 labels"
    (define-values (imgs lbls) (load-cifar10-fixture))
    (check-equal? (shape imgs) '(256 3 32 32))
    (check-equal? (shape lbls) '(256))
    (check-equal? (dtype lbls) 'int64)
    ;; the canonical first ten labels of data_batch_1
    (check-equal? (take (tensor->list lbls) 10) '(6 9 9 4 1 1 2 7 8 3))
    (check-equal? (map (lambda (l) (list-ref cifar10-label-names l))
                       (take (tensor->list lbls) 3))
                  '("frog" "truck" "truck"))
    ;; record 0 opens with red bytes 59 43 50, green 62 and blue 63 at pixel 0
    (define (px c h w) (ref imgs 0 c h w))
    (check-= (px 0 0 0) (- (/ 59 127.5) 1.0) 1e-6)
    (check-= (px 0 0 1) (- (/ 43 127.5) 1.0) 1e-6)
    (check-= (px 0 0 2) (- (/ 50 127.5) 1.0) 1e-6)
    (check-= (px 1 0 0) (- (/ 62 127.5) 1.0) 1e-6)
    (check-= (px 2 0 0) (- (/ 63 127.5) 1.0) 1e-6)
    (define all (tensor->list imgs))
    (check-true (>= (apply min all) -1.0) "pixel below -1")
    (check-true (<= (apply max all) 1.0) "pixel above 1")
    (check-true (< (apply min all) 0.0) "no dark pixel"))

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

  ;; the archive itself, only when the cache already has it: never a fetch
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
