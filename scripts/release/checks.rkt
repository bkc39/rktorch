#lang racket/base
(require (only-in json jsexpr?)
         (only-in racket/contract/base -> hash/c list/c listof or/c)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->lines file->string)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match match-define)
         (only-in racket/port port->list)
         (only-in racket/string string-prefix? string-split)
         (only-in setup/getinfo get-info/full)
         (only-in version/utils valid-version? version<?)
         (only-in "../../nix/racket-deps/racket-deps.rkt" platform-packages))

(provide abi-stamp
         check-release
         natives-plan
         pinned-sources
         previous-release
         read-release-data
         torch-info)

(define (refuse message . args)
  (apply raise-user-error 'release message args))

(define (data-file root)
  (build-path root "scripts" "release" "release.rktd"))

(define/contract (read-release-data root)
  (-> path-string? (values (listof (list/c string? string?)) (listof string?)))
  (match (call-with-input-file (data-file root) (λ (in) (port->list read in)))
    [(list `(platform-packages (,(? string? names) ,(? string? zips)) ...)
           `(lgpl-sources ,(? string? sources) ...))
     (values (map list names zips) sources)]
    [_
     (refuse "~a: expected (platform-packages (<name> <zip>) ...) (lgpl-sources <file> ...)"
             (data-file root))]))

(define/contract (torch-info root)
  (-> path-string? procedure?)
  (or (get-info/full (build-path root "torch"))
      (refuse "~a has no torch/info.rkt" root)))

(define (info-string info key root)
  (match (info key (λ () #f))
    [(? string? s) #:when (valid-version? s) s]
    [v (refuse "torch/info.rkt in ~a has ~a ~s, which is not a valid Racket version" root key v)]))

(define (flake-version root)
  (define flake (build-path root "flake.nix"))
  (match (regexp-match #px"(?m:^\\s*version = \"([^\"]*)\";)" (file->string flake))
    [(list _ version) version]
    [#f (refuse "~a declares no version = \"...\";" flake)]))

(define/contract (abi-stamp root)
  (-> path-string? string?)
  (define file (build-path root "torch" "foreign" "raw" "abi-stamp.rktd"))
  (match (call-with-input-file file read)
    [(? string? stamp) stamp]
    [v (refuse "~a holds ~s, not an ABI stamp string" file v)]))

(define (platform-dep-names info)
  (define deps (append (info 'deps (λ () '())) (info 'build-deps (λ () '()))))
  (sort (remove-duplicates
         (for/list ([dep (in-list deps)]
                    #:when (and (pair? dep) (memq '#:platform dep)))
           (car dep)))
        string<?))

(define/contract (pinned-sources root)
  (-> path-string? (hash/c string? (list/c string? string?)))
  (for*/hash ([line (in-list (file->lines (build-path root "scripts" "platform" "sources.txt")))]
              [fields (in-value (string-split line))]
              #:when (and (= (length fields) 3) (not (string-prefix? line "#"))))
    (match-define (list sha256 file url) fields)
    (values file (list sha256 url))))

(define tag-suffix #px"^(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$")

(define (tag-names-version? tag version)
  (define prefix (string-append "v" version))
  (and (string-prefix? tag prefix)
       (regexp-match? tag-suffix (substring tag (string-length prefix)))))

(define/contract (check-release root tag)
  (-> path-string? string? (hash/c symbol? string?))
  (define info (torch-info root))
  (define version (info-string info 'version root))
  (define native-version (info-string info 'native-version root))
  (define flake (flake-version root))
  (define-values (packages sources) (read-release-data root))
  (define listed (sort (map car packages) string<?))
  (define declared (platform-dep-names info))
  (define pinned (pinned-sources root))
  (define unpinned (filter (λ (s) (not (hash-has-key? pinned s))) sources))
  (cond
    [(and (not (string=? tag "")) (not (tag-names-version? tag version)))
     (refuse "tag ~a does not name torch/info.rkt's version ~a (want v~a, or v~a-<suffix>)"
             tag version version version)]
    [(not (string=? flake version))
     (refuse "flake.nix has version ~s; torch/info.rkt has version ~s" flake version)]
    [(not (equal? listed declared))
     (refuse "scripts/release/release.rktd builds ~s; torch/info.rkt's #:platform deps name ~s"
             listed declared)]
    [(pair? unpinned)
     (refuse "scripts/release/release.rktd attaches ~s, which scripts/platform/sources.txt does not pin"
             unpinned)]
    [else
     (platform-packages (build-path root "torch") (build-path root "pkgs"))
     (hasheq 'tag (if (string=? tag "") (string-append "v" version) tag)
             'version version
             'native-version native-version
             'abi-stamp (abi-stamp root))]))

(define (manifest-ref manifest key)
  (match (hash-ref manifest key #f)
    [(? string? s) s]
    [v (refuse "the previous manifest.json has ~a ~s; expected a string" key v)]))

(define/contract (natives-plan checked previous)
  (-> (hash/c symbol? string?) (or/c #f jsexpr?) (values boolean? string?))
  (define tag (hash-ref checked 'tag))
  (define native-version (hash-ref checked 'native-version))
  (define stamp (hash-ref checked 'abi-stamp))
  (define released (and previous (manifest-ref previous 'native-version)))
  (define released-stamp (and previous (manifest-ref previous 'abi-stamp)))
  (define natives-tag (and previous (manifest-ref previous 'natives-tag)))
  (cond
    [(not previous) (values #t tag)]
    [(and (string=? released native-version) (string=? released-stamp stamp))
     (values #f natives-tag)]
    [(string=? released native-version)
     (refuse
      (string-append
       "the ABI stamp is ~a, but the natives released at ~a have ~a and the same"
       " native-version, ~a; bump native-version in torch/info.rkt, torch's #:platform"
       " deps and pkgs/*/info.rkt")
      stamp natives-tag released-stamp native-version)]
    [(version<? released native-version) (values #t tag)]
    [else
     (refuse "native-version ~a is older than ~a, released at ~a" native-version released natives-tag)]))

(define (release-has-manifest? release)
  (for/or ([asset (in-list (hash-ref release 'assets '()))])
    (equal? (hash-ref asset 'name #f) "manifest.json")))

(define/contract (previous-release releases tag)
  (-> (listof jsexpr?) string? (or/c #f string?))
  (define candidates
    (for/list ([release (in-list releases)]
               #:unless (hash-ref release 'draft #t)
               #:unless (equal? (hash-ref release 'tag_name #f) tag)
               #:when (release-has-manifest? release))
      release))
  (match (sort candidates string>? #:key (λ (r) (hash-ref r 'published_at "")))
    ['() #f]
    [(cons newest _) (hash-ref newest 'tag_name)]))
