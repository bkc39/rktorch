#lang racket/base
(require (only-in file/sha1 bytes->hex-string sha1)
         (only-in file/zip zip)
         (only-in json read-json)
         (only-in racket/contract/combinator exn:fail:contract:blame?)
         (only-in racket/file
                  display-lines-to-file display-to-file file->lines file->string file->value
                  make-directory*)
         (only-in racket/match == match-define)
         (only-in racket/string string-split)
         (only-in rackunit check-equal? check-exn check-match check-regexp-match test-case)
         (only-in "../../torch/private/util.rkt" with-temporary-directory)
         (only-in "assets.rkt" asset-limit assemble check-asset-size release-notes)
         (only-in "checks.rkt" check-release)
         (only-in "fixture.rkt" make-root packages refused stamp-a tarball-bytes tarball-sha256
                  write-forms))

(define (sparse-file file size)
  (call-with-output-file file #:exists 'truncate
    (λ (out)
      (file-position out (sub1 size))
      (write-bytes #"\0" out))))

(test-case "an asset must be under GitHub's 2 GiB limit"
  (with-temporary-directory (dir)
    (define under (build-path dir "under.zip"))
    (define at (build-path dir "at.zip"))
    (sparse-file under (sub1 asset-limit))
    (sparse-file at asset-limit)
    (check-equal? (check-asset-size under) (sub1 asset-limit))
    (refused #rx"at.zip is 2147483648 bytes; GitHub refuses a release asset of 2 GiB"
             (λ () (check-asset-size at)))
    (check-exn exn:fail:contract:blame? (λ () (check-asset-size 'at.zip)))))

(define (write-zip artifacts name #:checksum [checksum #f])
  (define stage (build-path artifacts "stage" name))
  (write-forms (build-path stage "info.rkt") "#lang info"
               '((define collection "torch-native") (define version "0.1") (define deps '("base"))))
  (define out (build-path artifacts name "pkg"))
  (make-directory* out)
  (define zip-file (build-path out (string-append name ".zip")))
  (when (file-exists? zip-file)
    (delete-file zip-file))
  (parameterize ([current-directory stage])
    (zip zip-file "info.rkt"))
  (display-to-file (or checksum (call-with-input-file zip-file sha1))
                   (format "~a.CHECKSUM" zip-file)
                   #:exists 'truncate)
  zip-file)

(define (write-sources dir #:bytes [bytes tarball-bytes])
  (define sources (build-path dir "sources"))
  (make-directory* sources)
  (display-to-file bytes (build-path sources "lame-3.100.tar.gz") #:exists 'truncate)
  sources)

(define (checked-release dir tag)
  (hash-set (check-release dir tag) 'natives-inputs "inputs-a"))

(define (assemble-built dir #:artifacts artifacts #:sources sources)
  (assemble dir (checked-release dir "v0.0.1-alpha")
            #:repository "owner/repo"
            #:commit "c0ffee"
            #:artifacts artifacts
            #:sources sources
            #:previous #f
            #:previous-catalog #f
            #:out (build-path dir "dist")))

(define (dist-files dir)
  (sort (map path->string (directory-list (build-path dir "dist"))) string<?))

(test-case "a release that builds the natives carries zips, checksums, sources, a manifest and entries"
  (with-temporary-directory (dir)
    (make-root dir)
    (define artifacts (build-path dir "artifacts"))
    (for-each (λ (name) (write-zip artifacts name)) packages)
    (define manifest (assemble-built dir #:artifacts artifacts #:sources (write-sources dir)))
    (check-equal? (dist-files dir)
                  '("SHA256SUMS" "catalog-entry.rktd" "lame-3.100.tar.gz" "manifest.json"
                    "torch-aarch64-macosx.zip" "torch-aarch64-macosx.zip.CHECKSUM"
                    "torch-x86_64-linux.zip" "torch-x86_64-linux.zip.CHECKSUM"))
    (check-equal? (call-with-input-file (build-path dir "dist" "manifest.json") read-json) manifest)
    (check-match manifest
                 (hash-table ['natives-tag "v0.0.1-alpha"]
                             ['abi-stamp (== stamp-a)]
                             ['native-version "0.1"]
                             ['natives-inputs "inputs-a"]
                             [_ _] ...))
    (check-match (hash-ref manifest 'platform-packages)
                 (list (hash-table ['name "torch-aarch64-macosx"] [_ _] ...)
                       (hash-table ['name "torch-x86_64-linux"]
                                   ['url "https://github.com/owner/repo/releases/download/v0.0.1-alpha/torch-x86_64-linux.zip"]
                                   [_ _] ...)))
    (check-match (hash-ref manifest 'sources)
                 (list (hash-table ['file "lame-3.100.tar.gz"] ['sha256 (== tarball-sha256)] [_ _] ...)))
    (define catalog (file->value (build-path dir "dist" "catalog-entry.rktd")))
    (check-equal? (sort (hash-keys catalog) string<?) (cons "torch" packages))
    (check-match (hash-ref catalog "torch")
                 (hash-table ['source "https://github.com/owner/repo.git?path=torch#v0.0.1-alpha"]
                             ['checksum "c0ffee"]
                             ['dependencies (list (list "base" '#:version "9.1") _ ...)]
                             [_ _] ...))
    (check-equal? (hash-ref (hash-ref catalog "torch-x86_64-linux") 'checksum)
                  (file->string (build-path dir "dist" "torch-x86_64-linux.zip.CHECKSUM")))
    (assemble-built dir #:artifacts artifacts #:sources (write-sources dir))
    (define sums (file->lines (build-path dir "dist" "SHA256SUMS")))
    (check-equal? (length sums) 7 "a second run into the same directory sums only the assets")
    (for ([line (in-list sums)])
      (match-define (list sum file) (string-split line))
      (check-equal? sum (bytes->hex-string
                         (call-with-input-file (build-path dir "dist" file) sha256-bytes))))))

(test-case "an oversized zip fails before anything is hashed"
  (with-temporary-directory (dir)
    (make-root dir)
    (define artifacts (build-path dir "artifacts"))
    (for-each (λ (name) (write-zip artifacts name)) packages)
    (sparse-file (build-path artifacts "torch-x86_64-linux" "pkg" "torch-x86_64-linux.zip")
                 asset-limit)
    (refused #rx"torch-x86_64-linux.zip is 2147483648 bytes"
             (λ () (assemble-built dir #:artifacts artifacts #:sources (write-sources dir))))))

(test-case "a missing zip, a stale .CHECKSUM or an unpinned tarball fails"
  (with-temporary-directory (dir)
    (make-root dir)
    (define artifacts (build-path dir "artifacts"))
    (write-zip artifacts "torch-x86_64-linux")
    (refused #rx"torch-aarch64-macosx: no zip at"
             (λ () (assemble-built dir #:artifacts artifacts #:sources (write-sources dir))))
    (write-zip artifacts "torch-aarch64-macosx" #:checksum (make-string 40 #\0))
    (refused #rx"torch-aarch64-macosx.zip.CHECKSUM holds 0+, but the zip's sha1 is"
             (λ () (assemble-built dir #:artifacts artifacts #:sources (write-sources dir))))
    (write-zip artifacts "torch-aarch64-macosx")
    (refused #rx"lame-3.100.tar.gz has sha256 .*; scripts/platform/sources.txt pins"
             (λ () (assemble-built dir #:artifacts artifacts
                                   #:sources (write-sources dir #:bytes #"tampered\n"))))))

(define (assemble-carried dir previous previous-catalog)
  (assemble dir (checked-release dir "v0.0.2")
            #:repository "owner/repo"
            #:commit "beef"
            #:artifacts #f
            #:sources (build-path dir "no-sources")
            #:previous previous
            #:previous-catalog previous-catalog
            #:out (build-path dir "dist")))

(test-case "a release with unchanged natives reuses the earlier release's assets and entries"
  (with-temporary-directory (dir)
    (make-root dir #:version "0.0.2")
    (define previous
      (hasheq 'native-version "0.1" 'abi-stamp stamp-a 'natives-tag "v0.0.1-alpha"
              'natives-inputs "inputs-a"
              'platform-packages (list (hasheq 'name "torch-x86_64-linux"))
              'sources (list (hasheq 'file "lame-3.100.tar.gz"))))
    (define previous-catalog
      (for/hash ([name (in-list (cons "torch" packages))])
        (values name (hash 'name name 'source (string-append "https://old/" name)))))
    (define manifest (assemble-carried dir previous previous-catalog))
    (check-equal? (dist-files dir) '("SHA256SUMS" "catalog-entry.rktd" "manifest.json"))
    (check-match manifest
                 (hash-table ['natives-tag "v0.0.1-alpha"]
                             ['tag "v0.0.2"]
                             ['natives-inputs "inputs-a"]
                             [_ _] ...))
    (check-equal? (hash-ref manifest 'platform-packages) (hash-ref previous 'platform-packages))
    (define catalog (file->value (build-path dir "dist" "catalog-entry.rktd")))
    (check-equal? (hash-ref (hash-ref catalog "torch-x86_64-linux") 'source)
                  "https://old/torch-x86_64-linux")
    (check-equal? (hash-ref (hash-ref catalog "torch") 'source)
                  "https://github.com/owner/repo.git?path=torch#v0.0.2")
    (refused #rx"no entry for torch-aarch64-macosx"
             (λ () (assemble-carried dir previous (hash-remove previous-catalog "torch-aarch64-macosx"))))
    (refused #rx"no build artifacts, and no previous manifest.json"
             (λ () (assemble-carried dir #f #f)))))

(test-case "the notes say whether this release built the natives"
  (with-temporary-directory (dir)
    (define template (build-path dir "notes.md"))
    (display-lines-to-file '("{{tag}} {{version}} {{native-version}} {{abi-stamp}}" "{{natives}}")
                           template)
    (define built
      (hasheq 'tag "v0.0.1" 'version "0.0.1" 'native-version "0.1" 'abi-stamp "s"
              'natives-tag "v0.0.1"
              'platform-packages (list (hasheq 'file "torch-x86_64-linux.zip" 'size 104500000))
              'sources (list (hasheq 'file "lame-3.100.tar.gz" 'upstream "https://u/lame"))))
    (define notes (release-notes template built))
    (check-regexp-match #rx"^v0.0.1 0.0.1 0.1 s\n" notes)
    (check-regexp-match #rx"- `torch-x86_64-linux.zip`, 104.5 MB" notes)
    (check-regexp-match #rx"- `lame-3.100.tar.gz`, from https://u/lame" notes)
    (check-regexp-match #rx"unchanged since v0.0.1-alpha"
                        (release-notes template (hash-set* built 'tag "v0.0.2"
                                                           'natives-tag "v0.0.1-alpha")))
    (display-lines-to-file '("{{tag}} {{nonsense}}") template #:exists 'truncate)
    (refused #rx"no field \\{\\{nonsense\\}\\}" (λ () (release-notes template built)))))
