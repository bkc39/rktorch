#lang racket/base

(module+ test
  (require (only-in json jsexpr->string string->jsexpr)
           (only-in racket/file file->bytes)
           (only-in threading ~>)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../main.rkt" copy! dtype full-like with-no-grad)
           (only-in "../nn.rkt" BatchNorm1d Linear Sequential load-state! save-state! state-dict)
           (only-in "../nn/safetensors.rkt"
                    call-with-atomic-output-file read-safetensors safetensors-header
                    safetensors-metadata write-safetensors!)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "private/comparable.rkt" layer-bytes))

  (define-runtime-path master-file "master-save-state.safetensors")

  (define (fresh-model)
    (Sequential (Linear 3 2) (BatchNorm1d 2)))

  (define (fixture-model)
    (define net (fresh-model))
    (with-no-grad
      (for ([e (in-list (state-dict net))] [k (in-naturals 1)])
        (copy! (cdr e) (full-like (cdr e) (if (~> (cdr e) dtype (eq? 'int64)) k (* 1.5 k))))))
    net)

  (test-case "a file written by master's save-state! loads"
    (define net (fresh-model))
    (load-state! net master-file)
    (check-equal? (layer-bytes net) (layer-bytes (fixture-model))))

  (test-case "save-state! writes master's bytes, so master's load-state! reads them"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (save-state! (fixture-model) path)
      (check-equal? (file->bytes path) (file->bytes master-file))))

  (test-case "__metadata__ is a string map, and load-state! reads past it"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (define meta (hash "format" "rktorch-checkpoint" "version" "1"))
      (write-safetensors! 'test path (state-dict (fixture-model)) #:metadata meta)
      (define file (read-safetensors path))
      (check-equal? (safetensors-metadata file) meta)
      (check-false (hash-has-key? (safetensors-header file) '__metadata__))
      (check-false (safetensors-metadata (read-safetensors master-file)))
      (define net (fresh-model))
      (load-state! net path)
      (check-equal? (layer-bytes net) (layer-bytes (fixture-model)))))

  (define (directory-names dir)
    (sort (map path->string (directory-list dir)) string<?))

  (test-case "a write that raises leaves the target as it was and no temporary file"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (save-state! (fixture-model) path)
      (check-exn #rx"half-way"
                 (lambda ()
                   (call-with-atomic-output-file
                    path
                    (lambda (out)
                      (write-bytes #"partial" out)
                      (error 'test "half-way")))))
      (check-equal? (file->bytes path) (file->bytes master-file))
      (check-equal? (directory-names dir) '("m.safetensors"))))

  (test-case "a write broken half-way leaves the target as it was and no temporary file"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define path (build-path dir "m.safetensors"))
      (save-state! (fixture-model) path)
      (define started (make-semaphore))
      (define writer
        (thread (lambda ()
                  (call-with-atomic-output-file path
                                                (lambda (out)
                                                  (write-bytes #"partial" out)
                                                  (semaphore-post started)
                                                  (sync never-evt))))))
      (semaphore-wait started)
      (break-thread writer)
      (thread-wait writer)
      (check-equal? (file->bytes path) (file->bytes master-file))
      (check-equal? (directory-names dir) '("m.safetensors"))))

  (test-case "save-state! writes through a symlink at the target, as master did"
    (with-temporary-directory (dir #:template "save-state-~a")
      (make-directory (build-path dir "real"))
      (define target (build-path dir "real" "m.safetensors"))
      (call-with-output-file target (lambda (out) (write-bytes #"old" out)))
      (define link (build-path dir "m.safetensors"))
      (make-file-or-directory-link (build-path "real" "m.safetensors") link)
      (save-state! (fixture-model) link)
      (check-true (link-exists? link) "the link is still a link")
      (check-equal? (file->bytes target) (file->bytes master-file))
      (check-equal? (directory-names (build-path dir "real")) '("m.safetensors"))))

  (test-case "save-state! follows a dangling link, and a relative path, as master did"
    (with-temporary-directory (dir #:template "save-state-~a")
      (make-directory (build-path dir "real"))
      (define link (build-path dir "dangling.safetensors"))
      (make-file-or-directory-link (build-path "real" "new.safetensors") link)
      (save-state! (fixture-model) link)
      (check-true (link-exists? link) "the link is still a link")
      (check-equal? (file->bytes (build-path dir "real" "new.safetensors"))
                    (file->bytes master-file))
      (parameterize ([current-directory dir])
        (save-state! (fixture-model) "relative.safetensors"))
      (check-equal? (file->bytes (build-path dir "relative.safetensors"))
                    (file->bytes master-file))))

  (test-case "a missing directory or a link loop raises a filesystem error, as master did"
    (with-temporary-directory (dir #:template "save-state-~a")
      (check-exn exn:fail:filesystem?
                 (lambda () (save-state! (fixture-model) (build-path dir "nope" "m.safetensors"))))
      (make-file-or-directory-link "loop2" (build-path dir "loop1"))
      (make-file-or-directory-link "loop1" (build-path dir "loop2"))
      (check-exn (lambda (e)
                   (and (exn:fail:filesystem? e) (regexp-match? #rx"loop1" (exn-message e))))
                 (lambda () (save-state! (fixture-model) (build-path dir "loop1"))))
      (check-equal? (directory-names dir) '("loop1" "loop2"))))

  (test-case "a target name with a ~ in it saves, and leaves no temporary file"
    (with-temporary-directory (dir #:template "save-state-~a")
      (define names '("run~1.safetensors" "model.safetensors~" "x~a.safetensors" "y~n.safetensors"))
      (for ([name (in-list names)])
        (save-state! (fixture-model) (build-path dir name))
        (check-equal? (file->bytes (build-path dir name)) (file->bytes master-file) name))
      (check-equal? (directory-names dir) (sort names string<?))))

  (test-case "a target ending in a link to .. or in . raises a filesystem error naming it"
    (with-temporary-directory (dir #:template "save-state-~a")
      (make-directory (build-path dir "a"))
      (define up (build-path dir "a" "up"))
      (make-file-or-directory-link ".." up)
      (for ([target (in-list (list up (build-path dir "a" 'same)))])
        (check-exn (lambda (e)
                     (and (exn:fail:filesystem? e)
                          (regexp-match? (regexp-quote (path->string target)) (exn-message e))))
                   (lambda () (save-state! (fixture-model) target))))
      (check-equal? (directory-names dir) '("a"))
      (check-equal? (directory-names (build-path dir "a")) '("up"))))

  (define (with-metadata src dst metadata)
    (define raw (file->bytes src))
    (define start (+ 8 (integer-bytes->integer raw #f #f 0 8)))
    (define header
      (~> (subbytes raw 8 start) bytes->string/utf-8 string->jsexpr
          (hash-set '__metadata__ metadata) jsexpr->string string->bytes/utf-8))
    (call-with-output-file dst
      (lambda (out)
        (~> (bytes-length header) (integer->integer-bytes 8 #f #f) (write-bytes out))
        (write-bytes header out)
        (write-bytes (subbytes raw start) out))))

  (test-case "a __metadata__ that is not a map reads as none, and load-state! reads past it"
    (with-temporary-directory (dir #:template "save-state-~a")
      (for ([metadata (in-list '("x" 5 (1 2)))] [k (in-naturals)])
        (define path (build-path dir (format "m~a.safetensors" k)))
        (with-metadata master-file path metadata)
        (check-false (safetensors-metadata (read-safetensors path)) (format "~s" metadata))
        (define net (fresh-model))
        (load-state! net path)
        (check-equal? (layer-bytes net) (layer-bytes (fixture-model)) (format "~s" metadata))))))
