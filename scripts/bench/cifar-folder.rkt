#lang racket/base

(require (only-in ffi/unsafe _fun _int get-ffi-obj lookup-errno saved-errno)
         (only-in file/sha1 bytes->hex-string)
         (only-in racket/contract/base ->* or/c)
         (only-in racket/file
                  delete-directory/files file->bytes file->string make-directory*
                  make-temporary-directory)
         (only-in racket/format ~r)
         (only-in racket/list append-map)
         (only-in racket/match match match-lambda)
         (only-in racket/os getpid)
         (only-in racket/path path-only)
         (only-in racket/string string-join string-trim)
         (only-in threading ~>)
         (only-in torch/vision/cifar10 cifar10-archive-files cifar10-label-names)
         (only-in "../../torch/private/contract.rkt" define/contract-out)
         (only-in "harness.rkt" with-temporary-path)
         (only-in "png.rkt" rgb8-png))

(define record-size 3073)
(define side 32)
(define plane (* side side))
(define labels (list->vector cifar10-label-names))

(define dataset/c (or/c "full" "fixture"))

(define (cifar10-folder-base)
  (define override (getenv "RKTORCH_CIFAR10_PNG_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "cifar10-png")))

(define/contract-out (cifar10-folder dataset #:base [base (cifar10-folder-base)])
  (->* (dataset/c) (#:base path-string?) path?)
  (build-path base dataset))

(define (digest-path root) (build-path root "digest"))

(define/contract-out (cifar10-folder-digest dataset #:base [base (cifar10-folder-base)])
  (->* (dataset/c) (#:base path-string?) (or/c #f string?))
  (define root (cifar10-folder dataset #:base base))
  (and (file-exists? (digest-path root))
       (~> root digest-path file->string string-trim)))

(define (split-records dataset)
  (match dataset
    ["fixture"
     (define bs (file->bytes (collection-file-path "fixtures/cifar10-256.bin" "torch" "vision")))
     (list (cons "train" bs) (cons "test" bs))]
    ["full"
     (define archive (make-immutable-hash (cifar10-archive-files)))
     (define (joined names) (apply bytes-append (map (lambda (n) (hash-ref archive n)) names)))
     (list (cons "train" (joined (for/list ([i (in-range 1 6)]) (format "data_batch_~a.bin" i))))
           (cons "test" (joined '("test_batch.bin"))))]))

(define (record-png bs at)
  (rgb8-png side side (lambda (x y c) (bytes-ref bs (+ at 1 (* c plane) (* y side) x)))))

(define (write-split! dir split bs)
  (for/list ([i (in-range (quotient (bytes-length bs) record-size))])
    (define at (* i record-size))
    (define relative
      (build-path split (vector-ref labels (bytes-ref bs at))
                  (format "~a.png" (~r i #:min-width 5 #:pad-string "0"))))
    (define png (record-png bs at))
    (define file (build-path dir relative))
    (make-directory* (path-only file))
    (call-with-output-file file (lambda (o) (write-bytes png o)))
    (cons (path->string relative) (bytes->hex-string (sha256-bytes png)))))

(define (write-tree! dir dataset)
  (define sums
    (sort (append-map (match-lambda [(cons split bs) (write-split! dir split bs)])
                      (split-records dataset))
          string<? #:key car))
  (define manifest
    (~> sums (map (match-lambda [(cons file sum) (format "~a  ~a\n" sum file)]) _) (string-join "")))
  (call-with-output-file (build-path dir "SHA256SUMS") (lambda (o) (write-string manifest o)))
  (call-with-output-file (digest-path dir)
    (lambda (o) (~> manifest string->bytes/utf-8 sha256-bytes bytes->hex-string (displayln o)))))

(define (partial-prefix dataset) (format ".~a-partial-" dataset))

(define (partial-template dataset) (format "~apid~a-~~a" (partial-prefix dataset) (getpid)))

(define signal-process
  (get-ffi-obj "kill" #f (_fun #:save-errno 'posix _int _int -> _int)))

(define (process-alive? pid)
  (or (zero? (signal-process pid 0)) (= (saved-errno) (lookup-errno 'EPERM))))

(define (partial-owner prefix name)
  (match (~> (string-append "^" (regexp-quote prefix) "(pid([0-9]+)-)?") pregexp (regexp-match name))
    [#f #f]
    [(list _ _ #f) 'none]
    [(list _ _ pid) (string->number pid)]))

(define (remove-stale-partials! base dataset)
  (define prefix (partial-prefix dataset))
  (for ([p (in-list (directory-list base))])
    (match (partial-owner prefix (path->string p))
      [#f (void)]
      [(? exact-positive-integer? pid) #:when (process-alive? pid) (void)]
      [_ (delete-directory/files (build-path base p) #:must-exist? #f)])))

(define (publish! partial root)
  (with-handlers ([exn:fail:filesystem? (lambda (e)
                                          (unless (file-exists? (digest-path root))
                                            (raise e)))])
    (unless (directory-exists? root)
      (rename-file-or-directory partial root))))

(define/contract-out (ensure-cifar10-folder! dataset #:base [base (cifar10-folder-base)])
  (->* (dataset/c) (#:base path-string?) string?)
  (define root (cifar10-folder dataset #:base base))
  (cond
    [(file-exists? (digest-path root)) (void)]
    [else
     (make-directory* base)
     (remove-stale-partials! base dataset)
     (with-temporary-path (partial (~> dataset partial-template
                                       (make-temporary-directory #:base-dir base)))
       (write-tree! partial dataset)
       (publish! partial root))])
  (cifar10-folder-digest dataset #:base base))

(module+ main
  (require (only-in racket/cmdline command-line)
           (only-in racket/contract/base contract))
  (define dataset
    (contract dataset/c (command-line #:args ([dataset "full"]) dataset)
              'command-line 'cifar-folder 'dataset #f))
  (define digest (ensure-cifar10-folder! dataset))
  (eprintf "cifar-folder: ~a at ~a\n" dataset (cifar10-folder dataset))
  (displayln digest))

(module+ test
  (require (only-in racket/file find-files)
           (only-in racket/list take)
           rackunit
           (only-in torch reshape tensor->list to-dtype)
           (only-in torch/vision/image read-image)
           (only-in torch/vision/image-folder image-folder-classes image-folder-samples))

  (define fixture-records
    (file->bytes (collection-file-path "fixtures/cifar10-256.bin" "torch" "vision")))

  (define (partials base)
    (find-files (lambda (p) (regexp-match? #rx"partial" (path->string p))) base))

  (define (sample-index s)
    (match (regexp-match #rx"([0-9]+)\\.png$" (path->string (car s)))
      [(list _ n) (string->number n)]))

  (test-case "the fixture tree is labelled by class, each image its record's pixels"
    (with-temporary-path (base (make-temporary-directory "cifar-folder-~a"))
      (define digest (ensure-cifar10-folder! "fixture" #:base base))
      (define root (cifar10-folder "fixture" #:base base))
      (check-regexp-match #px"^[0-9a-f]{64}$" digest)
      (check-equal? (image-folder-classes (build-path root "train")) cifar10-label-names)
      (define samples (image-folder-samples (build-path root "train")))
      (check-equal? (length samples) 256)
      (for ([s (in-list (take samples 3))])
        (define at (* (sample-index s) record-size))
        (check-equal? (cdr s) (bytes-ref fixture-records at))
        (check-equal? (~> s car (read-image #:mode 'rgb) (to-dtype 'int64) (reshape -1)
                          tensor->list)
                      (bytes->list (subbytes fixture-records (add1 at) (+ at record-size)))))
      (check-equal? (partials base) '())
      (with-temporary-path (again (make-temporary-directory "cifar-folder-~a"))
        (check-equal? (ensure-cifar10-folder! "fixture" #:base again) digest
                      "the digest is the tree's, wherever it is written"))))

  (test-case "a written tree is left alone: a read-only cache still gives its digest"
    (with-temporary-path (base (make-temporary-directory "cifar-folder-~a"))
      (define digest (ensure-cifar10-folder! "fixture" #:base base))
      (file-or-directory-permissions base #o555)
      (define again (with-handlers ([exn:fail? exn-message])
                      (ensure-cifar10-folder! "fixture" #:base base)))
      (file-or-directory-permissions base #o755)
      (check-equal? again digest)))

  (test-case "a write killed before its rename leaves nothing for the next write"
    (with-temporary-path (base (make-temporary-directory "cifar-folder-~a"))
      (define stale (build-path base ".fixture-partial-1234" "train" "airplane"))
      (make-directory* stale)
      (call-with-output-file (build-path stale "00000.png") void)
      (ensure-cifar10-folder! "fixture" #:base base)
      (check-equal? (partials base) '())))

  (define (spawn-racket expr)
    (define-values (p out in err)
      (subprocess #f #f #f (find-system-path 'exec-file) "-e" expr))
    (close-input-port out)
    (close-input-port err)
    (values p in))

  (test-case "a write removes dead writers' partials and leaves a live writer's"
    (define-values (live live-in) (spawn-racket "(read-line)"))
    (define-values (dead dead-in) (spawn-racket "(void)"))
    (close-output-port dead-in)
    (subprocess-wait dead)
    (dynamic-wind
     void
     (lambda ()
       (with-temporary-path (base (make-temporary-directory "cifar-folder-~a"))
         (define live-partial (format ".fixture-partial-pid~a-live" (subprocess-pid live)))
         (define scenario
           `((,live-partial kept)
             (,(format ".fixture-partial-pid~a-dead" (subprocess-pid dead)) removed)
             (".fixture-partial-pid0-none" removed)
             (".fixture-partial-1234" removed)))
         (for ([s (in-list scenario)])
           (define dir (build-path base (car s) "train" "airplane"))
           (make-directory* dir)
           (call-with-output-file (build-path dir "00000.png") void))
         (ensure-cifar10-folder! "fixture" #:base base)
         (check-equal? (for/list ([s (in-list scenario)])
                         (list (car s)
                               (if (directory-exists? (build-path base (car s))) 'kept 'removed)))
                       scenario)
         (check-true (file-exists? (build-path base live-partial "train" "airplane" "00000.png")))))
     (lambda ()
       (close-output-port live-in)
       (subprocess-kill live #t)
       (subprocess-wait live)))))

