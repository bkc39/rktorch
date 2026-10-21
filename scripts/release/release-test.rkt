#lang racket/base

(require (for-syntax (only-in syntax/parse/pre ...+ expr id))
         (only-in file/sha1 bytes->hex-string sha1)
         (only-in file/zip zip)
         (only-in json read-json)
         (only-in racket/file
                  delete-directory/files display-lines-to-file file->lines file->string
                  file->value make-directory* make-temporary-directory)
         (only-in racket/match == match)
         (only-in racket/string string-split)
         (only-in rackunit
                  check-equal? check-exn check-false check-match check-regexp-match check-true
                  test-case)
         (only-in syntax/parse/define define-syntax-parse-rule)
         (only-in "assets.rkt" asset-limit assemble check-asset-size release-notes)
         (only-in "checks.rkt" check-release natives-plan previous-release))

(define-syntax-parse-rule (with-temporary-directory (dir:id) body:expr ...+)
  (let ([dir (make-temporary-directory)])
    (dynamic-wind void
                  (lambda () body ...)
                  (lambda () (delete-directory/files dir #:must-exist? #f)))))

(define stamp-a (make-string 64 #\a))
(define stamp-b (make-string 64 #\b))
(define packages '("torch-aarch64-macosx" "torch-x86_64-linux"))
(define tarball-bytes #"pretend LGPL source tarball\n")
(define tarball-sha256 (bytes->hex-string (sha256-bytes tarball-bytes)))

(define (write-forms file lang forms)
  (make-directory* (let-values ([(dir _name _d?) (split-path file)]) dir))
  (with-output-to-file file #:exists 'truncate
    (λ ()
      (when lang (displayln lang))
      (for-each writeln forms))))

(define (platform-deps names version)
  (for/list ([name (in-list names)])
    `(,name #:platform ,(substring name 6) #:version ,version)))

(define (make-root dir
                   #:version [version "0.0.1"]
                   #:native-version [native-version "0.1"]
                   #:flake-version [flake-version version]
                   #:skeleton-version [skeleton-version native-version]
                   #:declared [declared packages]
                   #:listed [listed packages]
                   #:lgpl [lgpl '("lame-3.100.tar.gz")]
                   #:stamp [stamp stamp-a])
  (write-forms (build-path dir "torch" "info.rkt") "#lang info"
               `((define collection "torch")
                 (define version ,version)
                 (define native-version ,native-version)
                 (define deps '(("base" #:version "9.1") ,@(platform-deps declared native-version)))))
  (for ([name (in-list declared)])
    (write-forms (build-path dir "pkgs" name "info.rkt") "#lang info"
                 `((define collection "torch-native")
                   (define version ,skeleton-version)
                   (define deps '("base")))))
  (display-lines-to-file (list "{" (format "      version = ~s;" flake-version) "}")
                         (build-path dir "flake.nix"))
  (write-forms (build-path dir "torch" "foreign" "raw" "abi-stamp.rktd") #f (list stamp))
  (make-directory* (build-path dir "scripts" "platform"))
  (display-lines-to-file
   (list "# sha256  file  url"
         (format "~a  lame-3.100.tar.gz  https://example.org/lame-3.100.tar.gz" tarball-sha256))
   (build-path dir "scripts" "platform" "sources.txt"))
  (write-forms (build-path dir "scripts" "release" "release.rktd") #f
               `((platform-packages
                  ,@(for/list ([name (in-list listed)])
                      (list name (format "~a/pkg/~a.zip" name name))))
                 (lgpl-sources ,@lgpl)))
  dir)

(define (refused rx thunk)
  (check-exn (λ (e) (and (exn:fail:user? e) (regexp-match? rx (exn-message e)))) thunk))

(test-case "the tag must name torch/info.rkt's version, with an optional suffix"
  (with-temporary-directory (dir)
    (make-root dir)
    (check-equal? (hash-ref (check-release dir "") 'tag) "v0.0.1")
    (check-equal? (hash-ref (check-release dir "v0.0.1") 'tag) "v0.0.1")
    (check-equal? (hash-ref (check-release dir "v0.0.1-alpha") 'tag) "v0.0.1-alpha")
    (check-equal? (hash-ref (check-release dir "v0.0.1-rc.2") 'tag) "v0.0.1-rc.2")
    (for ([tag (in-list '("v0.0.2" "v0.0.1alpha" "0.0.1" "v0.0.1-" "v0.0.10" "v0.0"))])
      (refused #rx"does not name torch/info.rkt's version 0.0.1" (λ () (check-release dir tag))))))

(test-case "the flake's version must equal torch/info.rkt's"
  (with-temporary-directory (dir)
    (make-root dir #:flake-version "0.0.1.0")
    (refused #rx"flake.nix has version \"0.0.1.0\"; torch/info.rkt has version \"0.0.1\""
             (λ () (check-release dir "")))))

(test-case "the bindings' version and native-version move separately"
  (with-temporary-directory (dir)
    (make-root dir #:version "0.0.7" #:native-version "0.2")
    (define checked (check-release dir "v0.0.7"))
    (check-equal? (hash-ref checked 'version) "0.0.7")
    (check-equal? (hash-ref checked 'native-version) "0.2")
    (check-equal? (hash-ref checked 'abi-stamp) stamp-a)))

(test-case "a platform package whose version is not native-version fails"
  (with-temporary-directory (dir)
    (make-root dir #:skeleton-version "0.0.1")
    (refused #rx"is version \"0.0.1\"; .* has native-version \"0.1\"" (λ () (check-release dir "")))))

(test-case "version must be a Racket version"
  (with-temporary-directory (dir)
    (make-root dir #:version "0.0.1-alpha")
    (refused #rx"has version \"0.0.1-alpha\", which is not a valid Racket version"
             (λ () (check-release dir "")))))

(test-case "native-version must be a Racket version"
  (with-temporary-directory (dir)
    (make-root dir #:native-version "0.1-cuda")
    (refused #rx"native-version \"0.1-cuda\", which is not a valid Racket version"
             (λ () (check-release dir "")))))

(test-case "release.rktd must build exactly torch's #:platform deps"
  (with-temporary-directory (dir)
    (make-root dir #:listed (cons "torch-x86_64-linux-cu130" packages))
    (refused #rx"release.rktd builds .*torch-x86_64-linux-cu130.*#:platform deps name"
             (λ () (check-release dir ""))))
  (with-temporary-directory (dir)
    (make-root dir #:listed '("torch-x86_64-linux"))
    (refused #rx"release.rktd builds" (λ () (check-release dir "")))))

(test-case "an LGPL source must be pinned in sources.txt"
  (with-temporary-directory (dir)
    (make-root dir #:lgpl '("lame-3.100.tar.gz" "mpg123-9.9.tar.bz2"))
    (refused #rx"attaches \\(\"mpg123-9.9.tar.bz2\"\\), which scripts/platform/sources.txt does not pin"
             (λ () (check-release dir "")))))

(define (checked #:native-version [native-version "0.2"] #:stamp [stamp stamp-b] #:tag [tag "v0.0.2"])
  (hasheq 'tag tag 'version "0.0.2" 'native-version native-version 'abi-stamp stamp))

(define (released #:native-version [native-version "0.1"] #:stamp [stamp stamp-a])
  (hasheq 'tag "v0.0.1" 'native-version native-version 'abi-stamp stamp 'natives-tag "v0.0.1-alpha"))

(define (plan . args)
  (call-with-values (λ () (apply natives-plan args)) list))

(test-case "the natives build on the first release, and when native-version moves"
  (check-equal? (plan (checked) #f) '(#t "v0.0.2"))
  (check-equal? (plan (checked) (released)) '(#t "v0.0.2"))
  (check-equal? (plan (checked #:stamp stamp-a) (released)) '(#t "v0.0.2")))

(test-case "unchanged natives are not rebuilt, and keep the tag that built them"
  (check-equal? (plan (checked #:native-version "0.1" #:stamp stamp-a) (released))
                '(#f "v0.0.1-alpha")))

(test-case "a changed ABI stamp under the released native-version fails"
  (refused #rx"bump native-version"
           (λ () (natives-plan (checked #:native-version "0.1" #:stamp stamp-b) (released)))))

(test-case "a native-version older than the released one fails"
  (refused #rx"native-version 0.1 is older than 0.2, released at v0.0.1-alpha"
           (λ () (natives-plan (checked #:native-version "0.1") (released #:native-version "0.2")))))

(define (release tag published #:draft? [draft? #f] #:assets [assets '("manifest.json")])
  (hasheq 'tag_name tag 'draft draft? 'published_at published
          'assets (for/list ([name (in-list assets)]) (hasheq 'name name))))

(test-case "the previous release is the newest published one with a manifest"
  (define releases
    (list (release "v0.0.1" "2026-10-01T00:00:00Z")
          (release "v0.0.3" "2026-10-20T00:00:00Z" #:draft? #t)
          (release "v0.0.2" "2026-10-10T00:00:00Z")
          (release "v0.0.0-rc" "2026-10-15T00:00:00Z" #:assets '("torch-x86_64-linux.zip"))
          (release "v0.0.4" "2026-10-21T00:00:00Z")))
  (check-equal? (previous-release releases "v0.0.5") "v0.0.4")
  (check-equal? (previous-release releases "v0.0.4") "v0.0.2")
  (check-false (previous-release '() "v0.0.1"))
  (check-false (previous-release (list (release "v0.0.1" "2026-10-01T00:00:00Z" #:draft? #t))
                                 "v0.0.2")))

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
             (λ () (check-asset-size at)))))

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
  (with-output-to-file (format "~a.CHECKSUM" zip-file) #:exists 'truncate
    (λ () (display (or checksum (call-with-input-file zip-file sha1)))))
  zip-file)

(define (write-sources dir #:bytes [bytes tarball-bytes])
  (define sources (build-path dir "sources"))
  (make-directory* sources)
  (call-with-output-file (build-path sources "lame-3.100.tar.gz") #:exists 'truncate
    (λ (out) (write-bytes bytes out)))
  sources)

(define (assemble-built dir #:artifacts artifacts #:sources sources)
  (assemble dir (check-release dir "v0.0.1-alpha")
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
    (check-equal? (hash-ref manifest 'natives-tag) "v0.0.1-alpha")
    (check-equal? (hash-ref manifest 'abi-stamp) stamp-a)
    (check-equal? (hash-ref manifest 'native-version) "0.1")
    (check-match (hash-ref manifest 'platform-packages)
                 (list (hash-table ['name "torch-aarch64-macosx"] [_ _] ...)
                       (hash-table ['name "torch-x86_64-linux"]
                                   ['url "https://github.com/owner/repo/releases/download/v0.0.1-alpha/torch-x86_64-linux.zip"]
                                   [_ _] ...)))
    (check-match (hash-ref manifest 'sources)
                 (list (hash-table ['file "lame-3.100.tar.gz"] ['sha256 (== tarball-sha256)] [_ _] ...)))
    (define catalog (file->value (build-path dir "dist" "catalog-entry.rktd")))
    (check-equal? (sort (hash-keys catalog) string<?) (cons "torch" packages))
    (check-equal? (hash-ref (hash-ref catalog "torch") 'source)
                  "https://github.com/owner/repo.git?path=torch#v0.0.1-alpha")
    (check-equal? (hash-ref (hash-ref catalog "torch") 'checksum) "c0ffee")
    (check-equal? (hash-ref (hash-ref catalog "torch-x86_64-linux") 'checksum)
                  (file->string (build-path dir "dist" "torch-x86_64-linux.zip.CHECKSUM")))
    (assemble-built dir #:artifacts artifacts #:sources (write-sources dir))
    (define sums (file->lines (build-path dir "dist" "SHA256SUMS")))
    (check-equal? (length sums) 7 "a second run into the same directory sums only the assets")
    (for ([line (in-list sums)])
      (match (string-split line)
        [(list sum file)
         (check-equal? sum (bytes->hex-string
                            (call-with-input-file (build-path dir "dist" file) sha256-bytes)))]))))

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
  (assemble dir (check-release dir "v0.0.2")
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
              'platform-packages (list (hasheq 'name "torch-x86_64-linux"))
              'sources (list (hasheq 'file "lame-3.100.tar.gz"))))
    (define previous-catalog
      (for/hash ([name (in-list (cons "torch" packages))])
        (values name (hash 'name name 'source (string-append "https://old/" name)))))
    (define manifest (assemble-carried dir previous previous-catalog))
    (check-equal? (dist-files dir) '("SHA256SUMS" "catalog-entry.rktd" "manifest.json"))
    (check-equal? (hash-ref manifest 'natives-tag) "v0.0.1-alpha")
    (check-equal? (hash-ref manifest 'tag) "v0.0.2")
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
