#lang racket/base
(require (only-in file/sha1 bytes->hex-string sha1)
         (only-in json jsexpr? write-json)
         (only-in racket/contract/base -> hash/c listof or/c)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->string make-directory*)
         (only-in racket/match match)
         (only-in racket/pretty pretty-write)
         (only-in racket/string string-trim)
         (only-in "../platform/catalog-entry.rkt" catalog-entry)
         (only-in "checks.rkt" pinned-sources read-release-data torch-info))

(provide asset-limit
         check-asset-size
         assemble
         release-notes)

(define asset-limit (expt 2 31))

(define (refuse message . args)
  (apply raise-user-error 'release message args))

(define (check-asset-size file)
  (define size (file-size file))
  (if (< size asset-limit)
      size
      (refuse "~a is ~a bytes; GitHub refuses a release asset of 2 GiB (~a bytes) or more"
              file size asset-limit)))

(define (sha256-hex file)
  (bytes->hex-string (call-with-input-file file sha256-bytes)))

(define (file-name file)
  (let-values ([(_dir name _must-be-dir?) (split-path file)])
    (path->string name)))

(define (download-url repository tag file)
  (format "https://github.com/~a/releases/download/~a/~a" repository tag file))

(define (copy-into file out)
  (copy-file file (build-path out (file-name file)) #t))

(define (built-package repository tag artifacts out name zip-path)
  (define zip (build-path artifacts zip-path))
  (define checksum-file (path-add-checksum zip))
  (define size (and (file-exists? zip) (check-asset-size zip)))
  (define checksum (and size
                        (file-exists? checksum-file)
                        (check-asset-size checksum-file)
                        (string-trim (file->string checksum-file))))
  (define actual (and checksum (call-with-input-file zip sha1)))
  (cond
    [(not size) (refuse "~a: no zip at ~a" name zip)]
    [(not checksum) (refuse "~a: no ~a beside the zip" name checksum-file)]
    [(not (string=? checksum actual))
     (refuse "~a holds ~a, but the zip's sha1 is ~a" checksum-file checksum actual)]
    [else
     (copy-into zip out)
     (copy-into checksum-file out)
     (define file (file-name zip))
     (values (hasheq 'name name
                     'file file
                     'url (download-url repository tag file)
                     'size size
                     'sha256 (sha256-hex zip)
                     'checksum checksum)
             (catalog-entry name (download-url repository tag file) zip))]))

(define (path-add-checksum zip)
  (string->path (format "~a.CHECKSUM" zip)))

(define (attached-source root sources-dir out file)
  (define tarball (build-path sources-dir file))
  (define pinned (hash-ref (pinned-sources root) file))
  (define size (and (file-exists? tarball) (check-asset-size tarball)))
  (define actual (and size (sha256-hex tarball)))
  (cond
    [(not size) (refuse "~a: no source tarball at ~a" file tarball)]
    [(not (equal? actual (car pinned)))
     (refuse "~a has sha256 ~a; scripts/platform/sources.txt pins ~a" tarball actual (car pinned))]
    [else
     (copy-into tarball out)
     (hasheq 'file file 'size size 'sha256 actual 'upstream (cadr pinned))]))

(define (torch-entry root repository tag commit)
  (define info (torch-info root))
  (hash 'name "torch"
        'source (format "https://github.com/~a.git?path=torch#~a" repository tag)
        'checksum commit
        'author ""
        'description ""
        'tags '()
        'dependencies (append (info 'deps (λ () '())) (info 'build-deps (λ () '())))
        'modules '()))

(define (carried-entry previous-catalog name)
  (or (hash-ref previous-catalog name #f)
      (refuse "the previous catalog-entry.rktd has no entry for ~a" name)))

(define (built-natives root repository tag artifacts sources-dir out packages sources)
  (define-values (entries catalog-entries)
    (for/lists (entries catalog-entries) ([package (in-list packages)])
      (built-package repository tag artifacts out (car package) (cadr package))))
  (hasheq 'natives-tag tag
          'platform-packages entries
          'catalog (for/hash ([entry (in-list catalog-entries)])
                     (values (hash-ref entry 'name) entry))
          'sources (for/list ([file (in-list sources)])
                     (attached-source root sources-dir out file))))

(define (carried-natives previous previous-catalog packages)
  (hasheq 'natives-tag (hash-ref previous 'natives-tag)
          'platform-packages (hash-ref previous 'platform-packages)
          'catalog (for/hash ([package (in-list packages)])
                     (values (car package) (carried-entry previous-catalog (car package))))
          'sources (hash-ref previous 'sources)))

(define (write-sha256sums out)
  (define files
    (sort (for/list ([f (in-list (directory-list out))]
                     #:unless (equal? (path->string f) "SHA256SUMS"))
            (path->string f))
          string<?))
  (with-output-to-file (build-path out "SHA256SUMS") #:exists 'truncate
    (λ ()
      (for ([f (in-list files)])
        (printf "~a  ~a\n" (sha256-hex (build-path out f)) f)))))

(define/contract (assemble root checked
                           #:repository repository
                           #:commit commit
                           #:artifacts artifacts
                           #:sources sources-dir
                           #:previous previous
                           #:previous-catalog previous-catalog
                           #:out out)
  (-> path-string? (hash/c symbol? string?)
      #:repository string?
      #:commit string?
      #:artifacts (or/c #f path-string?)
      #:sources path-string?
      #:previous (or/c #f jsexpr?)
      #:previous-catalog (or/c #f hash?)
      #:out path-string?
      jsexpr?)
  (define tag (hash-ref checked 'tag))
  (define-values (packages sources) (read-release-data root))
  (make-directory* out)
  (define natives
    (cond
      [artifacts (built-natives root repository tag artifacts sources-dir out packages sources)]
      [(and previous previous-catalog) (carried-natives previous previous-catalog packages)]
      [else
       (refuse "no build artifacts, and no previous manifest.json and catalog-entry.rktd to reuse")]))
  (define manifest
    (hasheq 'tag tag
            'commit commit
            'version (hash-ref checked 'version)
            'native-version (hash-ref checked 'native-version)
            'abi-stamp (hash-ref checked 'abi-stamp)
            'natives-tag (hash-ref natives 'natives-tag)
            'platform-packages (hash-ref natives 'platform-packages)
            'sources (hash-ref natives 'sources)))
  (with-output-to-file (build-path out "catalog-entry.rktd") #:exists 'truncate
    (λ () (pretty-write (hash-set (hash-ref natives 'catalog) "torch" (torch-entry root repository tag commit)))))
  (with-output-to-file (build-path out "manifest.json") #:exists 'truncate
    (λ () (write-json manifest #:indent 2) (newline)))
  (write-sha256sums out)
  (for ([f (in-list (directory-list out #:build? #t))])
    (check-asset-size f))
  manifest)

(define (megabytes bytes)
  (real->decimal-string (/ bytes 1e6) 1))

(define (natives-paragraph manifest)
  (define packages (hash-ref manifest 'platform-packages))
  (define sources (hash-ref manifest 'sources))
  (if (equal? (hash-ref manifest 'natives-tag) (hash-ref manifest 'tag))
      (string-append
       "This release builds the platform packages:\n\n"
       (apply string-append
              (for/list ([p (in-list packages)])
                (format "- `~a`, ~a MB\n" (hash-ref p 'file) (megabytes (hash-ref p 'size)))))
       "\nThe sources of the LGPL libraries they ship are attached beside them:\n\n"
       (apply string-append
              (for/list ([s (in-list sources)])
                (format "- `~a`, from ~a\n" (hash-ref s 'file) (hash-ref s 'upstream)))))
      (format (string-append
               "The platform packages are unchanged since ~a. Their catalog entries keep"
               " pointing at that release's assets, beside which their LGPL sources are"
               " attached.\n")
              (hash-ref manifest 'natives-tag))))

(define/contract (release-notes template manifest)
  (-> path-string? jsexpr? string?)
  (define fields
    (hash "tag" (hash-ref manifest 'tag)
          "version" (hash-ref manifest 'version)
          "native-version" (hash-ref manifest 'native-version)
          "abi-stamp" (hash-ref manifest 'abi-stamp)
          "natives" (natives-paragraph manifest)))
  (regexp-replace* #px"\\{\\{([a-z-]+)\\}\\}"
                   (file->string template)
                   (λ (all name)
                     (hash-ref fields name (λ () (refuse "~a: no field ~a" template all))))))
