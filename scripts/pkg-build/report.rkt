#lang racket/base
(require (only-in racket/file file->string)
         (only-in racket/list take-right)
         (only-in racket/string string-split))

(define pkg "torch")

(define (section title)
  (printf "\n==================== ~a ====================\n" title))

(define (show-log work entry key #:tail tail)
  (define log (hash-ref entry key #f))
  (when log
    (section (format "~a: ~a" key log))
    (define lines (string-split (file->string (build-path work log)) "\n"))
    (for-each displayln (if (and tail (> (length lines) tail))
                            (take-right lines tail)
                            lines))))

(define (docs-built work)
  (define doc-dir (build-path work "doc"))
  (if (directory-exists? doc-dir)
      (for/list ([d (in-list (directory-list doc-dir))]
                 #:when (regexp-match? (regexp (string-append "@" pkg "$")) (path->string d)))
        (path->string d))
      '()))

(module+ main
  (require (only-in racket/match match-define))

  (match-define (vector work-arg) (current-command-line-arguments))
  (define work (path->complete-path work-arg))
  (define summary-file (build-path work "summary.rktd"))
  (unless (file-exists? summary-file)
    (raise-user-error 'report "no summary.rktd in ~a; the harness stopped before its summary step"
                      work))
  (define entry (hash-ref (call-with-input-file summary-file read) pkg #f))
  (unless entry
    (raise-user-error 'report "~a has no entry for ~a" summary-file pkg))
  (for ([key (in-list '(failure-log min-failure-log dep-failure-log
                        success-log test-failure-log test-success-log))])
    (show-log work entry key #:tail (and (eq? key 'success-log) 60)))
  (define docs (docs-built work))
  (define columns
    (list (cons "build" (and (hash-ref entry 'success-log #f)
                             (not (hash-ref entry 'failure-log #f))))
          (cons "dependencies" (not (hash-ref entry 'dep-failure-log #f)))
          (cons "docs" (pair? docs))
          (cons "tests" (and (hash-ref entry 'test-success-log #f) #t))))
  (section "columns")
  (for ([column (in-list columns)])
    (printf "~a: ~a\n" (car column) (if (cdr column) "ok" "FAILED")))
  (printf "docs built: ~a\n" (if (null? docs) "none" docs))
  (unless (andmap cdr columns)
    (exit 1)))
