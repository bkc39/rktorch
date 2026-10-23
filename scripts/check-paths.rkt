#lang racket/base

(require (only-in racket/contract/base -> contract-out list/c listof)
         (only-in racket/file file->lines file->value)
         (only-in racket/list filter-not remove-duplicates)
         (only-in racket/port with-output-to-string)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-split)
         (only-in racket/system system*))

(provide (contract-out
          [struct mention ([file string?]
                           [line exact-positive-integer?]
                           [path string?])]
          [stale-mentions (-> path-string? (listof string?) (listof mention?))]
          [check-paths
           (-> path-string? (listof string?) (listof allowance/c)
               (values (listof mention?) (listof allowance/c)))]))

(define-runtime-path repository "..")
(define-runtime-path allowances "check-paths-allow.rktd")

(define allowance/c (list/c string? string?))

(struct mention (file line path) #:transparent)

(define mention-rxs
  (list #px"@filepath\\{([^{}]+)\\}"
        #px"`([^`\\s]+\\.(?:rkt|py))`"))

(define pattern-rx #px"[*?{}<>\\[\\]$~…]|^/|://|\\.\\.\\.")

(define (line-paths line)
  (for*/list ([rx (in-list mention-rxs)]
              [path (in-list (regexp-match* rx line #:match-select cadr))]
              #:unless (regexp-match? pattern-rx path))
    path))

(define (file-mentions root file)
  (for*/list ([(line n) (in-parallel (in-list (file->lines (build-path root file)))
                                     (in-naturals 1))]
              [path (in-list (line-paths line))])
    (mention file n path)))

(define (resolves? root m)
  (define-values (dir _name _dir?)
    (split-path (path->complete-path (build-path root (mention-file m)))))
  (for/or ([base (in-list (list root dir))])
    (define target (build-path base (mention-path m)))
    (or (file-exists? target) (directory-exists? target) (link-exists? target))))

(define (stale-mentions root files)
  (for*/list ([file (in-list files)]
              [m (in-list (file-mentions root file))]
              #:unless (resolves? root m))
    m))

(define (mention-key m)
  (list (mention-file m) (mention-path m)))

(define (check-paths root files allowed)
  (define stale (stale-mentions root files))
  (define keys (remove-duplicates (map mention-key stale)))
  (values (filter-not (lambda (m) (member (mention-key m) allowed)) stale)
          (filter-not (lambda (entry) (member entry keys)) allowed)))

(define (tracked-files root)
  (define listing
    (parameterize ([current-directory root])
      (with-output-to-string
        (lambda ()
          (system* (find-executable-path "git")
                   "ls-files" "-z" "--" "*.rkt" "*.scrbl" "*.md")))))
  (for/list ([file (in-list (string-split listing "\u0000"))]
             #:when (file-exists? (build-path root file)))
    file))

(module+ main
  (define files (tracked-files repository))
  (define-values (stale unused)
    (check-paths repository files (file->value allowances)))
  (for ([m (in-list stale)])
    (printf "~a:~a: ~a\n" (mention-file m) (mention-line m) (mention-path m)))
  (for ([entry (in-list unused)])
    (printf "scripts/check-paths-allow.rktd: ~a no longer names a missing ~a\n"
            (car entry) (cadr entry)))
  (cond
    [(and (null? stale) (null? unused))
     (printf "check-paths: every path named in ~a files resolves\n" (length files))]
    [else
     (printf "check-paths: ~a path(s) resolve from neither the repository root nor their file's directory, ~a stale allowance(s)\n"
             (length stale) (length unused))
     (exit 1)]))
