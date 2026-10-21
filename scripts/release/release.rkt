#lang racket/base
(require (only-in json read-json)
         (only-in racket/cmdline command-line)
         (only-in racket/file display-to-file file->value)
         (only-in racket/list append-map)
         (only-in racket/match match match-define)
         (only-in racket/port port->list)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "assets.rkt" assemble release-notes)
         (only-in "checks.rkt" check-release natives-inputs natives-plan read-release-data refuse
                  release-data-sources)
         (only-in "github.rkt" download-root previous-release published-releases
                  unmet-required-checks))

(provide run)

(define-runtime-path root "../..")
(define-runtime-path notes-template "notes.md")

(define usage
  (string-append
   "usage: release.rkt <command> <arg> ...\n"
   "  check [--tag <tag>]\n"
   "  plan [--tag <tag>] [--previous <manifest.json>]\n"
   "  published [--except <tag>] <releases.json>\n"
   "  previous [--except <tag>] <releases.json> <manifests-dir>\n"
   "  natives-root <manifest.json> <owner/name>\n"
   "  required-checks <rules.json> <check-runs.json>\n"
   "  lgpl-sources\n"
   "  assemble --tag <tag> --repository <owner/name> --commit <sha> --sources <dir>\n"
   "           --notes <file> [--artifacts <dir> | --previous <dir>] <out-dir>"))

(define (read-json-file file)
  (call-with-input-file file read-json))

(define (read-json-values file)
  (append-map (λ (value) (if (list? value) value (list value)))
              (call-with-input-file file (λ (in) (port->list read-json in)))))

(define (print-outputs pairs)
  (for ([pair (in-list pairs)])
    (match-define (cons key value) pair)
    (printf "~a=~a\n" key value)))

(define (required flag value)
  (or value (refuse "~a is required\n~a" flag usage)))

(define (checked-release tag)
  (hash-set (check-release root tag) 'natives-inputs (natives-inputs root)))

(define (check-command args)
  (define tag #f)
  (command-line
   #:program "release.rkt check"
   #:argv args
   #:once-each
   [("--tag") t "The release tag (default: v<torch/info.rkt's version>)" (set! tag t)]
   #:args ()
   (check-release root tag)
   (void)))

(define (plan-command args)
  (define tag #f)
  (define previous #f)
  (command-line
   #:program "release.rkt plan"
   #:argv args
   #:once-each
   [("--tag") t "The release tag (default: v<torch/info.rkt's version>)" (set! tag t)]
   [("--previous") file "The previous release's manifest.json" (set! previous file)]
   #:args ()
   (define checked (checked-release tag))
   (define-values (build? natives-tag)
     (natives-plan checked (and previous (read-json-file previous))))
   (match-define (hash-table ['tag release-tag]
                             ['version version]
                             ['native-version native-version]
                             ['abi-stamp stamp]
                             ['natives-inputs inputs])
     checked)
   (print-outputs
    (list (cons 'tag release-tag)
          (cons 'version version)
          (cons 'native-version native-version)
          (cons 'abi-stamp stamp)
          (cons 'natives-inputs inputs)
          (cons 'build-natives (if build? "true" "false"))
          (cons 'natives-tag natives-tag)))))

(define (published-command args)
  (define except #f)
  (command-line
   #:program "release.rkt published"
   #:argv args
   #:once-each
   [("--except") t "Leave out this tag" (set! except t)]
   #:args (releases)
   (for-each displayln (published-releases (read-json-values releases) #:except except))))

(define (previous-command args)
  (define except #f)
  (command-line
   #:program "release.rkt previous"
   #:argv args
   #:once-each
   [("--except") t "Leave out this tag" (set! except t)]
   #:args (releases manifests)
   (define previous
     (previous-release (read-json-values releases)
                       (λ (tag) (read-json-file (build-path manifests tag "manifest.json")))
                       #:except except))
   (print-outputs (list (cons 'previous (or previous ""))))))

(define (natives-root-command manifest repository)
  (match (read-json-file manifest)
    [(hash-table ['natives-tag (? string? natives-tag)] [_ _] ...)
     (displayln (download-root repository natives-tag))]
    [_ (refuse "~a has no natives-tag string" manifest)]))

(define (required-checks-command rules check-runs)
  (match (unmet-required-checks (read-json-values rules) (read-json-values check-runs))
    ['() (void)]
    [unmet (refuse "the tagged commit's required checks have not all passed: ~a" unmet)]))

(define (assemble-command args)
  (define tag #f)
  (define repository #f)
  (define commit #f)
  (define artifacts #f)
  (define sources #f)
  (define previous #f)
  (define notes #f)
  (command-line
   #:program "release.rkt assemble"
   #:argv args
   #:once-each
   [("--tag") t "The release tag" (set! tag t)]
   [("--repository") r "owner/name on GitHub" (set! repository r)]
   [("--commit") c "The tagged commit" (set! commit c)]
   [("--artifacts") dir "The downloaded build artifacts, when the natives are built"
                    (set! artifacts dir)]
   [("--sources") dir "The fetched source tarballs" (set! sources dir)]
   [("--previous") dir "The previous release's manifest.json and catalog-entry.rktd"
                   (set! previous dir)]
   [("--notes") file "Where to write the release notes" (set! notes file)]
   #:args (out)
   (define release-tag (required "--tag" tag))
   (define release-repository (required "--repository" repository))
   (define release-commit (required "--commit" commit))
   (define sources-dir (required "--sources" sources))
   (define notes-file (required "--notes" notes))
   (define manifest
     (assemble root (checked-release release-tag)
               #:repository release-repository
               #:commit release-commit
               #:artifacts artifacts
               #:sources sources-dir
               #:previous (and previous (read-json-file (build-path previous "manifest.json")))
               #:previous-catalog (and previous
                                       (file->value (build-path previous "catalog-entry.rktd")))
               #:out out))
   (display-to-file (release-notes notes-template manifest) notes-file #:exists 'truncate)))

(define (run args)
  (match args
    [(cons "check" args) (check-command args)]
    [(cons "plan" args) (plan-command args)]
    [(cons "published" args) (published-command args)]
    [(cons "previous" args) (previous-command args)]
    [(list "natives-root" manifest repository) (natives-root-command manifest repository)]
    [(list "required-checks" rules check-runs) (required-checks-command rules check-runs)]
    [(list "lgpl-sources")
     (for-each displayln (release-data-sources (read-release-data root)))]
    [(cons "assemble" args) (assemble-command args)]
    [args (refuse "~a\ngot ~s" usage args)]))

(module+ main
  (run (vector->list (current-command-line-arguments))))
