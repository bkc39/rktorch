#lang racket/base
(require (only-in json write-json)
         (only-in racket/file display-lines-to-file display-to-file make-directory*)
         (only-in racket/port open-output-nowhere with-output-to-string)
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-exn check-false check-not-equal? check-true test-case)
         (only-in "../../torch/private/util.rkt" with-temporary-directory)
         (only-in "checks.rkt" abi-stamp check-release natives-inputs natives-inputs-digest natives-plan
                  release-packages torch-info)
         (only-in "fixture.rkt" make-root packages refused stamp-a stamp-b write-forms)
         (only-in "github.rkt" download-root download-url previous-release published-releases
                  unmet-required-checks)
         (only-in "release.rkt" run))

(test-case "the tag must name torch/info.rkt's version, with an optional suffix"
  (with-temporary-directory (dir)
    (make-root dir)
    (check-equal? (hash-ref (check-release dir #f) 'tag) "v0.0.1")
    (check-equal? (hash-ref (check-release dir "v0.0.1") 'tag) "v0.0.1")
    (check-equal? (hash-ref (check-release dir "v0.0.1-alpha") 'tag) "v0.0.1-alpha")
    (check-equal? (hash-ref (check-release dir "v0.0.1-rc.2") 'tag) "v0.0.1-rc.2")
    (for ([tag (in-list '("" "v0.0.2" "v0.0.1alpha" "0.0.1" "v0.0.1-" "v0.0.10" "v0.0"))])
      (refused #rx"does not name torch/info.rkt's version 0.0.1" (λ () (check-release dir tag))))))

(test-case "the flake's version must equal torch/info.rkt's"
  (with-temporary-directory (dir)
    (make-root dir #:flake-version "0.0.1.0")
    (refused #rx"flake.nix has version \"0.0.1.0\"; torch/info.rkt has version \"0.0.1\""
             (λ () (check-release dir #f)))))

(test-case "a flake.nix without a version fails"
  (with-temporary-directory (dir)
    (make-root dir)
    (display-lines-to-file '("{" "  pname = \"torch\";" "}") (build-path dir "flake.nix")
                           #:exists 'truncate)
    (refused #rx"flake.nix declares no version" (λ () (check-release dir #f)))))

(test-case "a root without torch/info.rkt fails"
  (with-temporary-directory (dir)
    (refused #rx"has no torch/info.rkt" (λ () (torch-info dir)))))

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
    (refused #rx"is version \"0.0.1\"; .* has native-version \"0.1\"" (λ () (check-release dir #f)))))

(test-case "version and native-version must be Racket versions"
  (with-temporary-directory (dir)
    (make-root dir #:version "0.0.1-alpha")
    (refused #rx"has version \"0.0.1-alpha\", which is not a valid Racket version"
             (λ () (check-release dir #f))))
  (with-temporary-directory (dir)
    (make-root dir #:native-version "0.1-cuda")
    (refused #rx"native-version \"0.1-cuda\", which is not a valid Racket version"
             (λ () (check-release dir #f)))))

(test-case "the ABI stamp must be a string"
  (with-temporary-directory (dir)
    (make-root dir)
    (write-forms (build-path dir "torch" "foreign" "raw" "abi-stamp.rktd") #f '(abi))
    (refused #rx"abi-stamp.rktd holds abi, not an ABI stamp string" (λ () (abi-stamp dir)))))

(test-case "release.rktd must have its four forms"
  (with-temporary-directory (dir)
    (make-root dir)
    (write-forms (build-path dir "scripts" "release" "release.rktd") #f
                 '((platform-packages ("torch-x86_64-linux" "torch-x86_64-linux.zip"))))
    (refused #rx"release.rktd: expected \\(platform-zips <zip> ...\\)" (λ () (check-release dir #f)))))

(test-case "the packages are torch's #:platform deps, each with one zip in release.rktd"
  (with-temporary-directory (dir)
    (make-root dir)
    (check-equal? (release-packages dir)
                  '(("torch-aarch64-macosx" "torch-aarch64-macosx/pkg/torch-aarch64-macosx.zip")
                    ("torch-x86_64-linux" "torch-x86_64-linux/pkg/torch-x86_64-linux.zip"))))
  (with-temporary-directory (dir)
    (make-root dir #:listed (cons "torch-x86_64-linux-cu130" packages))
    (refused #rx"release.rktd builds .*torch-x86_64-linux-cu130.zip.*#:platform deps name"
             (λ () (check-release dir #f))))
  (with-temporary-directory (dir)
    (make-root dir #:listed '("torch-x86_64-linux"))
    (refused #rx"deps name torch-aarch64-macosx; scripts/release/release.rktd has \\(\\) for it"
             (λ () (check-release dir #f)))))

(test-case "an LGPL source must be pinned in sources.txt"
  (with-temporary-directory (dir)
    (make-root dir #:lgpl '("lame-3.100.tar.gz" "mpg123-9.9.tar.bz2"))
    (refused #rx"attaches \\(\"mpg123-9.9.tar.bz2\"\\), which scripts/platform/sources.txt does not pin"
             (λ () (check-release dir #f)))))

(define (ls-tree-line path object)
  (format "100644 blob ~a\t~a" (make-string 40 object) path))

(test-case "the natives' inputs digest skips the excluded paths and nothing else"
  (define source (ls-tree-line "cpp/src/a.cpp" #\1))
  (define test-source (ls-tree-line "cpp/tests/a-test.cpp" #\2))
  (define info (ls-tree-line "pkgs/info.rkt" #\3))
  (define excluded '("cpp/tests/" "pkgs/info.rkt"))
  (define digest (natives-inputs-digest (list source test-source info) excluded))
  (check-equal? (natives-inputs-digest (list source
                                             (ls-tree-line "cpp/tests/a-test.cpp" #\9)
                                             (ls-tree-line "pkgs/info.rkt" #\9))
                                       excluded)
                digest)
  (check-not-equal? (natives-inputs-digest (list (ls-tree-line "cpp/src/a.cpp" #\9) test-source info)
                                           excluded)
                    digest)
  (check-not-equal? (natives-inputs-digest
                     (list source test-source (ls-tree-line "cpp/testsuite.cpp" #\4) info)
                     excluded)
                    digest
                    "an excluded directory covers only the paths under it")
  (refused #rx"excludes \\(\"cpp/gone.sh\"\\) from the natives' inputs, but git tracks no such file"
           (λ () (natives-inputs-digest (list source test-source info) (cons "cpp/gone.sh" excluded)))))

(define (git dir . args)
  (check-true
   (parameterize ([current-directory dir]
                  [current-output-port (open-output-nowhere)])
     (apply system* (find-executable-path "git")
            "-c" "user.name=release-test" "-c" "user.email=release-test@example.org"
            "-c" "commit.gpgsign=false" "-c" "init.defaultBranch=master"
            args))))

(define (commit-all dir message)
  (git dir "add" "-A")
  (git dir "commit" "-q" "--allow-empty" "-m" message))

(test-case "natives-inputs hashes what git tracks under the inputs at HEAD"
  (with-temporary-directory (dir)
    (make-root dir)
    (define notes (build-path dir "pkgs" "torch-x86_64-linux" "NOTES"))
    (display-to-file "first" notes)
    (git dir "init" "-q")
    (commit-all dir "first")
    (define first (natives-inputs dir))
    (display-to-file "second" notes #:exists 'truncate)
    (commit-all dir "an excluded file")
    (check-equal? (natives-inputs dir) first)
    (display-to-file "outside the inputs" (build-path dir "README"))
    (commit-all dir "outside")
    (check-equal? (natives-inputs dir) first)
    (display-to-file "THIRD-PARTY-NOTICES" (build-path dir "pkgs" "torch-x86_64-linux" "NOTICES"))
    (commit-all dir "a shipped file")
    (define second (natives-inputs dir))
    (check-not-equal? second first)
    (display-to-file "uncommitted" (build-path dir "pkgs" "torch-x86_64-linux" "scratch"))
    (check-equal? (natives-inputs dir) second "only HEAD counts")))

(test-case "natives-inputs fails outside a git checkout"
  (with-temporary-directory (dir)
    (make-root dir)
    (parameterize ([current-error-port (open-output-nowhere)])
      (refused #rx"git ls-tree -r HEAD -- pkgs failed" (λ () (natives-inputs dir))))))

(define (checked #:native-version [native-version "0.2"]
                 #:stamp [stamp stamp-b]
                 #:inputs [inputs "inputs-b"]
                 #:tag [tag "v0.0.2"]
                 #:version [version "0.0.2"])
  (hasheq 'tag tag 'version version 'native-version native-version 'abi-stamp stamp
          'natives-inputs inputs))

(define (released #:native-version [native-version "0.1"]
                  #:stamp [stamp stamp-a]
                  #:inputs [inputs "inputs-a"]
                  #:tag [tag "v0.0.1"]
                  #:version [version "0.0.1"])
  (hasheq 'tag tag 'version version 'native-version native-version 'abi-stamp stamp
          'natives-inputs inputs 'natives-tag "v0.0.1-alpha"))

(define (plan . args)
  (call-with-values (λ () (apply natives-plan args)) list))

(define unchanged (checked #:native-version "0.1" #:stamp stamp-a #:inputs "inputs-a"))

(test-case "the natives build on the first release, and when native-version moves"
  (check-equal? (plan (checked) #f) '(#t "v0.0.2"))
  (check-equal? (plan (checked) (released)) '(#t "v0.0.2"))
  (check-equal? (plan (checked #:stamp stamp-a #:inputs "inputs-a") (released)) '(#t "v0.0.2")))

(test-case "unchanged natives are not rebuilt, and keep the tag that built them"
  (check-equal? (plan unchanged (released)) '(#f "v0.0.1-alpha")))

(test-case "a changed ABI stamp under the released native-version fails"
  (refused #rx"the ABI stamp is b+, .* bump native-version"
           (λ () (natives-plan (hash-set unchanged 'abi-stamp stamp-b) (released)))))

(test-case "changed natives' inputs under the released native-version fail"
  (refused #rx"natives' inputs .* changed since v0.0.1-alpha, but native-version is still 0.1"
           (λ () (natives-plan (hash-set unchanged 'natives-inputs "inputs-b") (released)))))

(test-case "a native-version older than the released one fails"
  (refused #rx"native-version 0.1 is older than 0.2, released at v0.0.1-alpha"
           (λ () (natives-plan (checked #:native-version "0.1") (released #:native-version "0.2")))))

(test-case "the version may not go backwards, and repeats only under a new tag"
  (refused #rx"version 0.0.1 is older than 0.0.2, released at v0.0.2"
           (λ () (natives-plan (checked #:version "0.0.1" #:tag "v0.0.1")
                               (released #:version "0.0.2" #:tag "v0.0.2"))))
  (check-equal? (plan (checked #:version "0.0.1" #:tag "v0.0.1")
                      (released #:version "0.0.1" #:tag "v0.0.1-alpha"))
                '(#t "v0.0.1"))
  (refused #rx"v0.0.1 is the previous release's tag already"
           (λ () (natives-plan (checked #:version "0.0.1" #:tag "v0.0.1") (released)))))

(test-case "a previous manifest without its string fields fails"
  (for ([key (in-list '(tag version native-version abi-stamp natives-inputs natives-tag))])
    (refused #rx"the previous manifest.json needs the strings tag, version"
             (λ () (natives-plan (checked) (hash-set (released) key 1))))
    (refused #rx"the previous manifest.json needs the strings"
             (λ () (natives-plan (checked) (hash-remove (released) key))))))

(define (release tag published #:draft? [draft? #f] #:assets [assets '("manifest.json")])
  (hasheq 'tag_name tag 'draft draft? 'published_at published
          'assets (for/list ([name (in-list assets)]) (hasheq 'name name))))

(define releases
  (list (release "v0.0.1" "2026-10-01T00:00:00Z")
        (release "v0.0.3" "2026-10-20T00:00:00Z" #:draft? #t)
        (release "v0.0.2" "2026-10-10T00:00:00Z" #:assets '("x.zip" "manifest.json" "SHA256SUMS"))
        (release "v0.0.0-rc" "2026-10-15T00:00:00Z" #:assets '("torch-x86_64-linux.zip"))
        (release "v0.0.4" "2026-10-21T00:00:00Z")))

(test-case "a release is published when it is not a draft and carries a manifest"
  (check-equal? (published-releases releases) '("v0.0.1" "v0.0.2" "v0.0.4"))
  (check-equal? (published-releases releases #:except "v0.0.2") '("v0.0.1" "v0.0.4"))
  (check-equal? (published-releases (list (release "v0.0.1" "2026-10-01T00:00:00Z" #:draft? #t)))
                '()))

(define ((manifests native-versions) tag)
  (hasheq 'native-version (hash-ref native-versions tag)))

(test-case "the previous release has the highest native-version, then the latest date"
  (define natives (manifests (hash "v0.0.1" "0.1" "v0.0.2" "0.2" "v0.0.4" "0.1" "v0.0.5" "0.2")))
  (check-equal? (previous-release releases natives) "v0.0.2")
  (check-equal? (previous-release releases natives #:except "v0.0.2") "v0.0.4")
  (check-equal? (previous-release (cons (release "v0.0.5" "2026-10-22T00:00:00Z") releases) natives)
                "v0.0.5")
  (check-equal? (previous-release releases (manifests (hash "v0.0.1" "0.10" "v0.0.2" "0.9"
                                                            "v0.0.4" "0.1")))
                "v0.0.1"
                "native-versions compare as versions, not strings")
  (check-false (previous-release '() natives))
  (refused #rx"the manifest.json of v0.0.1 has no native-version string"
           (λ () (previous-release releases (λ (tag) (hasheq 'native-version 1))))))

(define required-rules
  (list (hasheq 'type "deletion")
        (hasheq 'type "required_status_checks"
                'parameters (hasheq 'required_status_checks
                                    (list (hasheq 'context "build" 'integration_id 15368)
                                          (hasheq 'context "lint" 'integration_id 15368)
                                          (hasheq 'context "drift"))))))

(define (check-run id name conclusion #:app [app 15368] #:status [status "completed"])
  (hasheq 'id id 'name name 'status status 'conclusion conclusion 'app (hasheq 'id app)))

(test-case "the tagged commit needs every required check's latest run to have succeeded"
  (check-equal? (unmet-required-checks
                 required-rules
                 (list (hasheq 'check_runs (list (check-run 1 "build" "failure")
                                                 (check-run 2 "build" "success")))
                       (hasheq 'check_runs (list (check-run 3 "lint" "success")
                                                 (check-run 4 "drift" "success" #:app 1)))))
                '())
  (check-equal? (unmet-required-checks
                 required-rules
                 (list (hasheq 'check_runs (list (check-run 1 "build" "success")
                                                 (check-run 2 "build" "failure")
                                                 (check-run 3 "lint" "success" #:app 99)
                                                 (check-run 4 "drift" 'null
                                                            #:status "in_progress")))))
                '("build (failure)" "lint (no run)" "drift (in_progress)"))
  (refused #rx"master's rules require no status checks"
           (λ () (unmet-required-checks (list (hasheq 'type "deletion")) '()))))

(test-case "release assets live under the tag's download root"
  (check-equal? (download-root "owner/repo" "v0.0.1-alpha")
                "https://github.com/owner/repo/releases/download/v0.0.1-alpha")
  (check-equal? (download-url "owner/repo" "v0.0.1-alpha" "torch-x86_64-linux.zip")
                "https://github.com/owner/repo/releases/download/v0.0.1-alpha/torch-x86_64-linux.zip"))

(test-case "natives-root prints where a release's natives were published"
  (with-temporary-directory (dir)
    (define manifest (build-path dir "manifest.json"))
    (with-output-to-file manifest
      (λ () (write-json (hasheq 'tag "v0.0.2" 'natives-tag "v0.0.1-alpha"))))
    (check-equal? (with-output-to-string
                    (λ () (run (list "natives-root" (path->string manifest) "owner/repo"))))
                  "https://github.com/owner/repo/releases/download/v0.0.1-alpha\n")
    (with-output-to-file manifest #:exists 'truncate (λ () (write-json (hasheq 'tag "v0.0.2"))))
    (refused #rx"manifest.json has no natives-tag string"
             (λ () (run (list "natives-root" (path->string manifest) "owner/repo"))))))

(test-case "assemble's required flags fail with the usage"
  (for ([missing (in-list '("--tag" "--repository" "--commit" "--sources" "--notes"))])
    (define flags
      (for*/list ([flag (in-list '("--tag" "--repository" "--commit" "--sources" "--notes"))]
                  #:unless (equal? flag missing)
                  [arg (in-list (list flag "x"))])
        arg))
    (refused (regexp (string-append missing " is required\nusage: release.rkt"))
             (λ () (run (append '("assemble") flags '("out")))))))

(test-case "an unknown command fails with the usage"
  (refused #rx"^release: usage: release.rkt .*got \\(\"bogus\"\\)" (λ () (run '("bogus")))))
