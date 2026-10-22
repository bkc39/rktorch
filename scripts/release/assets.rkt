#lang racket/base
(require (only-in file/sha1 bytes->hex-string sha1)
         (only-in json jsexpr? write-json)
         (only-in racket/contract/base -> hash/c or/c)
         (only-in racket/contract/region define/contract)
         (only-in racket/file display-lines-to-file file->string make-directory*)
         (only-in racket/match match-define)
         (only-in racket/path file-name-from-path)
         (only-in racket/pretty pretty-write)
         (only-in racket/string string-append* string-trim)
         (only-in "../../nix/racket-deps/racket-deps.rkt" info-dependencies)
         (only-in "../platform/catalog-entry.rkt" catalog-entry checksum-path zip-catalog-entry)
         (only-in "checks.rkt" pinned-sources read-release-data refuse release-data-sources
                  release-packages torch-info)
         (only-in "github.rkt" download-url))

(provide asset-limit
         check-asset-size
         assemble
         release-notes)

(define asset-limit (expt 2 31))

(define/contract (check-asset-size file)
  (-> path-string? exact-nonnegative-integer?)
  (define size (file-size file))
  (if (< size asset-limit)
      size
      (refuse "~a is ~a bytes; GitHub refuses a release asset of 2 GiB (~a bytes) or more"
              file size asset-limit)))

(define (sha256-hex file)
  (bytes->hex-string (call-with-input-file file sha256-bytes)))

(define (file-name file)
  (path->string (file-name-from-path file)))

(define (copy-into file out)
  (copy-file file (build-path out (file-name file)) #t))

(define (built-package #:repository repository
                       #:tag tag
                       #:artifacts artifacts
                       #:out out
                       #:name name
                       #:zip zip-path)
  (define zip (build-path artifacts zip-path))
  (define checksum-file (checksum-path zip))
  (define size (and (file-exists? zip) (check-asset-size zip)))
  (define checksum (and size
                        (file-exists? checksum-file)
                        (check-asset-size checksum-file)
                        (string-trim (file->string checksum-file))))
  (define actual (and checksum (call-with-input-file zip sha1)))
  (define file (file-name zip))
  (define url (download-url repository tag file))
  (cond
    [(not size) (refuse "~a: no zip at ~a" name zip)]
    [(not checksum) (refuse "~a: no ~a beside the zip" name checksum-file)]
    [(not (string=? checksum actual))
     (refuse "~a holds ~a, but the zip's sha1 is ~a" checksum-file checksum actual)]
    [else
     (copy-into zip out)
     (copy-into checksum-file out)
     (values (hasheq 'name name
                     'file file
                     'url url
                     'size size
                     'sha256 (sha256-hex zip)
                     'checksum checksum)
             (zip-catalog-entry name url zip))]))

(define (attached-source root sources-dir out file)
  (define tarball (build-path sources-dir file))
  (match-define (list pinned-sha256 upstream) (hash-ref (pinned-sources root) file))
  (define size (and (file-exists? tarball) (check-asset-size tarball)))
  (define actual (and size (sha256-hex tarball)))
  (cond
    [(not size) (refuse "~a: no source tarball at ~a" file tarball)]
    [(not (equal? actual pinned-sha256))
     (refuse "~a has sha256 ~a; scripts/platform/sources.txt pins ~a" tarball actual pinned-sha256)]
    [else
     (copy-into tarball out)
     (hasheq 'file file 'size size 'sha256 actual 'upstream upstream)]))

(define (torch-entry root repository tag commit)
  (catalog-entry "torch"
                 (format "https://github.com/~a.git?path=torch#~a" repository tag)
                 #:checksum commit
                 #:dependencies (info-dependencies (torch-info root))))

(define (carried-entry previous-catalog name)
  (or (hash-ref previous-catalog name #f)
      (refuse "the previous catalog-entry.rktd has no entry for ~a" name)))

(define (built-natives root repository tag artifacts sources-dir out)
  (define-values (entries catalog-entries)
    (for/lists (entries catalog-entries) ([package (in-list (release-packages root))])
      (match-define (list name zip) package)
      (built-package #:repository repository #:tag tag #:artifacts artifacts #:out out
                     #:name name #:zip zip)))
  (hasheq 'natives-tag tag
          'platform-packages entries
          'catalog (for/hash ([entry (in-list catalog-entries)])
                     (values (hash-ref entry 'name) entry))
          'sources (for/list ([file (in-list (release-data-sources (read-release-data root)))])
                     (attached-source root sources-dir out file))))

(define (carried-natives root previous previous-catalog)
  (match-define (hash-table ['natives-tag natives-tag]
                            ['platform-packages platform-packages]
                            ['sources sources]
                            [_ _] ...)
    previous)
  (hasheq 'natives-tag natives-tag
          'platform-packages platform-packages
          'catalog (for/hash ([package (in-list (release-packages root))])
                     (match-define (list name _zip) package)
                     (values name (carried-entry previous-catalog name)))
          'sources sources))

(define (write-sha256sums out)
  (define files
    (sort (for/list ([f (in-list (directory-list out))]
                     #:unless (equal? (path->string f) "SHA256SUMS"))
            (path->string f))
          string<?))
  (display-lines-to-file (for/list ([f (in-list files)])
                           (format "~a  ~a" (sha256-hex (build-path out f)) f))
                         (build-path out "SHA256SUMS")
                         #:exists 'truncate))

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
  (match-define (hash-table ['tag tag]
                            ['version version]
                            ['native-version native-version]
                            ['abi-stamp stamp]
                            ['natives-inputs inputs]
                            [_ _] ...)
    checked)
  (make-directory* out)
  (define natives
    (cond
      [artifacts (built-natives root repository tag artifacts sources-dir out)]
      [(and previous previous-catalog) (carried-natives root previous previous-catalog)]
      [else
       (refuse "no build artifacts, and no previous manifest.json and catalog-entry.rktd to reuse")]))
  (match-define (hash-table ['natives-tag natives-tag]
                            ['platform-packages platform-packages]
                            ['catalog catalog]
                            ['sources sources])
    natives)
  (define manifest
    (hasheq 'tag tag
            'commit commit
            'version version
            'native-version native-version
            'abi-stamp stamp
            'natives-inputs inputs
            'natives-tag natives-tag
            'platform-packages platform-packages
            'sources sources))
  (with-output-to-file (build-path out "catalog-entry.rktd") #:exists 'truncate
    (λ () (pretty-write (hash-set catalog "torch" (torch-entry root repository tag commit)))))
  (with-output-to-file (build-path out "manifest.json") #:exists 'truncate
    (λ () (write-json manifest #:indent 2) (newline)))
  (write-sha256sums out)
  (for ([f (in-list (directory-list out #:build? #t))])
    (check-asset-size f))
  manifest)

(define (megabytes bytes)
  (real->decimal-string (/ bytes 1e6) 1))

(define (natives-paragraph manifest)
  (match-define (hash-table ['tag tag]
                            ['natives-tag natives-tag]
                            ['platform-packages packages]
                            ['sources sources]
                            [_ _] ...)
    manifest)
  (if (equal? natives-tag tag)
      (string-append
       "This release builds the platform packages:\n\n"
       (string-append*
        (for/list ([p (in-list packages)])
          (format "- `~a`, ~a MB\n" (hash-ref p 'file) (megabytes (hash-ref p 'size)))))
       "\nThe sources of the LGPL libraries they ship are attached beside them:\n\n"
       (string-append*
        (for/list ([s (in-list sources)])
          (format "- `~a`, from ~a\n" (hash-ref s 'file) (hash-ref s 'upstream)))))
      (format (string-append
               "The platform packages are unchanged since ~a. Their catalog entries keep"
               " pointing at that release's assets, beside which their LGPL sources are"
               " attached.\n")
              natives-tag)))

(define/contract (release-notes template manifest)
  (-> path-string? jsexpr? string?)
  (match-define (hash-table ['tag tag]
                            ['version version]
                            ['native-version native-version]
                            ['abi-stamp stamp]
                            [_ _] ...)
    manifest)
  (define fields
    (hash "tag" tag
          "version" version
          "native-version" native-version
          "abi-stamp" stamp
          "natives" (natives-paragraph manifest)))
  (regexp-replace* #px"\\{\\{([a-z-]+)\\}\\}"
                   (file->string template)
                   (λ (all name)
                     (hash-ref fields name (λ () (refuse "~a: no field ~a" template all))))))
