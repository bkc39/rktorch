#lang racket/base

(require (only-in racket/contract/base -> contract-out list/c listof)
         (only-in racket/file file->lines file->value)
         (only-in racket/list append-map filter-not remove-duplicates take)
         (only-in racket/match match-define)
         (only-in racket/path path-only)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/set for*/set set-member?)
         (only-in racket/string string-join string-split string-trim)
         (only-in racket/system system*))

(provide (contract-out
          [struct mention ([file string?]
                           [line exact-positive-integer?]
                           [path string?])]
          [tracked-files (-> path-string? (listof string?))]
          [stale-mentions
           (-> path-string? (listof string?) (listof string?) (listof mention?))]
          [check-paths
           (-> path-string? (listof string?) (listof string?) (listof allowance/c)
               (values (listof mention?) (listof allowance/c)))]))

(define-runtime-path repository "..")
(define-runtime-path allowances "check-paths-allow.rktd")

(define allowance/c (list/c string? string?))

(struct mention (file line path) #:transparent)

(define filepath-rx #px"@filepath\\{([^{}]+)\\}")
(define span-rx #px"`([^`]+)`")
(define word-span-rx #px"`([^`\\s]+)`")
(define option-rx #px"^--[^=]+=")
(define host-rx #px"^(?:[A-Za-z0-9-]+\\.)+[A-Za-z]{2,}/")
(define fence-rx #px"^\\s*```")
(define source-rx #px"[^/]\\.(?:rkt|py)$")
(define pattern-rx #px"[*?{}<>\\[\\]$~…]|^/|://|\\.\\.\\.")
(define scanned-rx #px"\\.(?:rkt|scrbl|md)$")

(define (source-paths text #:lone [lone? #f])
  (define tokens
    (for/list ([token (in-list (string-split text))]
               #:unless (regexp-match? host-rx token))
      (regexp-replace option-rx token "")))
  (define single? (and lone? (= (length tokens) 1)))
  (filter (lambda (token)
            (and (regexp-match? source-rx token)
                 (or single? (regexp-match? #rx"/" token))))
          tokens))

(define (line-paths line fenced?)
  (define candidates
    (cond
      [fenced? (source-paths line)]
      [else
       (append (regexp-match* filepath-rx line #:match-select cadr)
               (append-map (lambda (span) (source-paths span #:lone #t))
                           (append (regexp-match* span-rx line #:match-select cadr)
                                   (regexp-match* word-span-rx line
                                                  #:match-select cadr))))]))
  (remove-duplicates
   (filter-not (lambda (path) (regexp-match? pattern-rx path)) candidates)))

(define (fenced-flags lines)
  (for/fold ([inside? #f] [flags '()] #:result (reverse flags))
            ([line (in-list lines)])
    (define fence? (regexp-match? fence-rx line))
    (values (if fence? (not inside?) inside?)
            (cons (and inside? (not fence?)) flags))))

(define (file-mentions root file)
  (define lines (file->lines (build-path root file)))
  (for*/list ([(line fenced? n) (in-parallel (in-list lines)
                                            (in-list (fenced-flags lines))
                                            (in-naturals 1))]
              [path (in-list (line-paths line fenced?))])
    (mention file n path)))

(define (prefixes file)
  (define parts (string-split file "/"))
  (for/list ([n (in-range 1 (add1 (length parts)))])
    (string-join (take parts n) "/")))

(define (normalize dir path)
  (string-trim (path->string (simplify-path (build-path dir path) #f))
               "/" #:left? #f))

(define (resolves? known m)
  (define dir (or (path-only (mention-file m)) "."))
  (for/or ([base (in-list (list "." dir))])
    (set-member? known (normalize base (mention-path m)))))

(define (stale-mentions root tracked files)
  (define known
    (for*/set ([file (in-list tracked)]
               [prefix (in-list (prefixes file))])
      prefix))
  (for*/list ([file (in-list files)]
              [m (in-list (file-mentions root file))]
              #:unless (resolves? known m))
    m))

(define (mention-key m)
  (list (mention-file m) (mention-path m)))

(define (check-paths root tracked files allowed)
  (define stale (stale-mentions root tracked files))
  (define keys (remove-duplicates (map mention-key stale)))
  (values (filter-not (lambda (m) (member (mention-key m) allowed)) stale)
          (filter-not (lambda (entry) (member entry keys)) allowed)))

(define (tracked-files root)
  (define git (find-executable-path "git"))
  (define out (open-output-string))
  (define err (open-output-string))
  (define listed?
    (and git
         (parameterize ([current-directory root]
                        [current-output-port out]
                        [current-error-port err])
           (system* git "ls-files" "-z"))))
  (define files
    (for/list ([file (in-list (string-split (get-output-string out) "\u0000"))]
               #:when (or (file-exists? (build-path root file))
                          (link-exists? (build-path root file))))
      file))
  (define problem
    (cond
      [(not git) "git is not on PATH"]
      [(not listed?)
       (format "git ls-files failed in ~a: ~a"
               root (string-trim (get-output-string err)))]
      [(null? files) (format "git ls-files lists no file in ~a" root)]
      [else #f]))
  (when problem
    (raise-user-error 'check-paths "~a" problem))
  files)

(module+ main
  (define tracked (tracked-files repository))
  (define files
    (filter (lambda (file) (regexp-match? scanned-rx file)) tracked))
  (define-values (stale unused)
    (check-paths repository tracked files (file->value allowances)))
  (for ([m (in-list stale)])
    (printf "~a:~a: ~a\n" (mention-file m) (mention-line m) (mention-path m)))
  (for ([entry (in-list unused)])
    (match-define (list file path) entry)
    (printf "scripts/check-paths-allow.rktd: ~a no longer names a missing ~a\n"
            file path))
  (cond
    [(and (null? stale) (null? unused))
     (printf "check-paths: every path named in ~a files is tracked\n" (length files))]
    [else
     (printf "check-paths: ~a path(s) name no tracked file from the repository root or their file's directory, ~a stale allowance(s)\n"
             (length stale) (length unused))
     (exit 1)]))
