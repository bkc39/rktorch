#lang racket/base
(require (only-in json read-json)
         (only-in racket/cmdline command-line)
         (only-in racket/file file->value)
         (only-in racket/match match)
         (only-in racket/port port->list)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in "assets.rkt" assemble release-notes)
         (only-in "checks.rkt" check-release natives-plan previous-release read-release-data))

(define-runtime-path root "../..")
(define-runtime-path notes-template "notes.md")

(define (read-json-file file)
  (call-with-input-file file read-json))

(define (read-releases file)
  (for*/list ([value (in-list (call-with-input-file file (λ (in) (port->list read-json in))))]
              [release (in-list (if (list? value) value (list value)))])
    release))

(define (print-outputs pairs)
  (for ([pair (in-list pairs)])
    (printf "~a=~a\n" (car pair) (cdr pair))))

(define (plan tag previous)
  (define checked (check-release root tag))
  (define-values (build? natives-tag) (natives-plan checked previous))
  (print-outputs
   (list (cons 'tag (hash-ref checked 'tag))
         (cons 'version (hash-ref checked 'version))
         (cons 'native-version (hash-ref checked 'native-version))
         (cons 'abi-stamp (hash-ref checked 'abi-stamp))
         (cons 'build-natives (if build? "true" "false"))
         (cons 'natives-tag natives-tag))))

(define (assemble-command args)
  (define tag "")
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
   (define checked (check-release root tag))
   (define manifest
     (assemble root checked
               #:repository repository
               #:commit commit
               #:artifacts artifacts
               #:sources sources
               #:previous (and previous (read-json-file (build-path previous "manifest.json")))
               #:previous-catalog (and previous
                                       (file->value (build-path previous "catalog-entry.rktd")))
               #:out out))
   (call-with-output-file notes #:exists 'truncate
     (λ (port) (write-string (release-notes notes-template manifest) port)))
   (void)))

(module+ main
  (match (vector->list (current-command-line-arguments))
    [(list "plan" tag) (plan tag #f)]
    [(list "plan" tag manifest) (plan tag (read-json-file manifest))]
    [(list "previous" tag releases)
     (print-outputs (list (cons 'previous (or (previous-release (read-releases releases) tag) ""))))]
    [(list "lgpl-sources")
     (define-values (_packages sources) (read-release-data root))
     (for-each displayln sources)]
    [(cons "assemble" args) (assemble-command args)]
    [args
     (raise-user-error
      'release
      (string-append
       "usage: plan <tag> [<previous manifest.json>] | previous <tag> <releases.json> | lgpl-sources\n"
       "     | assemble --tag <tag> --repository <owner/name> --commit <sha> [--artifacts <dir>]\n"
       "                --sources <dir> [--previous <dir>] --notes <file> <out-dir>\n"
       "got ~s")
      args)]))
