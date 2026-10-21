#lang racket/base
(require (only-in file/sha1 bytes->hex-string)
         (only-in json jsexpr?)
         (only-in racket/contract/base -> hash/c list/c listof or/c)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->lines file->list file->string file->value)
         (only-in racket/list filter-map filter-not remove-duplicates)
         (only-in racket/match match match-define match-lambda)
         (only-in racket/path file-name-from-path)
         (only-in racket/string string-join string-prefix? string-split string-suffix?)
         (only-in racket/system system*)
         (only-in setup/getinfo get-info/full)
         (only-in version/utils version<?)
         (only-in "../../nix/racket-deps/racket-deps.rkt"
                  info-dependencies info-version platform-packages))

(provide abi-stamp
         check-release
         natives-inputs
         natives-inputs-digest
         natives-plan
         pinned-sources
         read-release-data
         refuse
         release-packages
         (struct-out release-data)
         torch-info)

(define (refuse message . args)
  (apply raise-user-error 'release message args))

(struct release-data (zips sources inputs excluded))

(define (data-file root)
  (build-path root "scripts" "release" "release.rktd"))

(define/contract (read-release-data root)
  (-> path-string? release-data?)
  (match (file->list (data-file root))
    [(list `(platform-zips ,(? string? zips) ...)
           `(lgpl-sources ,(? string? sources) ...)
           `(natives-inputs ,(? string? inputs) ..1)
           `(natives-inputs-exclude ,(? string? excluded) ...))
     (release-data zips sources inputs excluded)]
    [_
     (refuse (string-append
              "~a: expected (platform-zips <zip> ...) (lgpl-sources <file> ...)"
              " (natives-inputs <dir> ...+) (natives-inputs-exclude <path> ...)")
             (data-file root))]))

(define/contract (torch-info root)
  (-> path-string? procedure?)
  (or (get-info/full (build-path root "torch"))
      (refuse "~a has no torch/info.rkt" root)))

(define (flake-version root)
  (define flake (build-path root "flake.nix"))
  (match (regexp-match #px"(?m:^\\s*version = \"([^\"]*)\";)" (file->string flake))
    [(list _ version) version]
    [#f (refuse "~a declares no version = \"...\";" flake)]))

(define/contract (abi-stamp root)
  (-> path-string? string?)
  (define file (build-path root "torch" "foreign" "raw" "abi-stamp.rktd"))
  (match (file->value file)
    [(? string? stamp) stamp]
    [v (refuse "~a holds ~s, not an ABI stamp string" file v)]))

(define (platform-dep-names info)
  (sort (remove-duplicates
         (filter-map (match-lambda
                       [(list (? string? name) _ ... '#:platform _ _ ...) name]
                       [_ #f])
                     (info-dependencies info)))
        string<?))

(define (zip-package zip)
  (path->string (path-replace-extension (file-name-from-path zip) #"")))

(define/contract (release-packages root)
  (-> path-string? (listof (list/c string? string?)))
  (define declared (platform-dep-names (torch-info root)))
  (define zips (release-data-zips (read-release-data root)))
  (define undeclared (filter-not (λ (zip) (member (zip-package zip) declared)) zips))
  (cond
    [(pair? undeclared)
     (refuse "scripts/release/release.rktd builds ~s; torch/info.rkt's #:platform deps name ~s"
             undeclared declared)]
    [else
     (for/list ([name (in-list declared)])
       (match (filter (λ (zip) (string=? (zip-package zip) name)) zips)
         [(list zip) (list name zip)]
         [found
          (refuse "torch/info.rkt's #:platform deps name ~a; scripts/release/release.rktd has ~s for it, not one zip"
                  name found)]))]))

(define/contract (pinned-sources root)
  (-> path-string? (hash/c string? (list/c string? string?)))
  (for/fold ([pinned (hash)])
            ([line (in-list (file->lines (build-path root "scripts" "platform" "sources.txt")))])
    (match (string-split line)
      [(list sha256 file url) #:when (not (string-prefix? line "#"))
       (hash-set pinned file (list sha256 url))]
      [_ pinned])))

(define tag-suffix #px"^(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$")

(define (tag-names-version? tag version)
  (define prefix (string-append "v" version))
  (and (string-prefix? tag prefix)
       (regexp-match? tag-suffix (substring tag (string-length prefix)))))

(define/contract (check-release root tag)
  (-> path-string? (or/c #f string?) (hash/c symbol? string?))
  (define torch-dir (build-path root "torch"))
  (define info (torch-info root))
  (define version (info-version 'release info torch-dir 'version))
  (define native-version (info-version 'release info torch-dir 'native-version))
  (define flake (flake-version root))
  (define pinned (pinned-sources root))
  (define unpinned
    (filter-not (λ (source) (hash-has-key? pinned source))
                (release-data-sources (read-release-data root))))
  (cond
    [(and tag (not (tag-names-version? tag version)))
     (refuse "tag ~a does not name torch/info.rkt's version ~a (want v~a, or v~a-<suffix>)"
             tag version version version)]
    [(not (string=? flake version))
     (refuse "flake.nix has version ~s; torch/info.rkt has version ~s" flake version)]
    [(pair? unpinned)
     (refuse "scripts/release/release.rktd attaches ~s, which scripts/platform/sources.txt does not pin"
             unpinned)]
    [else
     (release-packages root)
     (platform-packages torch-dir (build-path root "pkgs"))
     (hasheq 'tag (or tag (string-append "v" version))
             'version version
             'native-version native-version
             'abi-stamp (abi-stamp root))]))

(define (excluded? path excluded)
  (for/or ([entry (in-list excluded)])
    (if (string-suffix? entry "/")
        (string-prefix? path entry)
        (string=? path entry))))

(define (ls-tree-path line)
  (match (string-split line "\t")
    [(list _ path) path]
    [_ (refuse "git ls-tree printed ~s, not <mode> <type> <object>\\t<path>" line)]))

(define/contract (natives-inputs-digest lines excluded)
  (-> (listof string?) (listof string?) string?)
  (define paths (map ls-tree-path lines))
  (define stale
    (filter-not (λ (entry) (for/or ([path (in-list paths)]) (excluded? path (list entry))))
                excluded))
  (cond
    [(pair? stale)
     (refuse "scripts/release/release.rktd excludes ~s from the natives' inputs, but git tracks no such file"
             stale)]
    [else
     (define kept
       (for/list ([line (in-list lines)]
                  [path (in-list paths)]
                  #:unless (excluded? path excluded))
         line))
     (bytes->hex-string (sha256-bytes (string->bytes/utf-8 (string-join kept "\n"))))]))

(define/contract (natives-inputs root)
  (-> path-string? string?)
  (match-define (release-data _ _ inputs excluded) (read-release-data root))
  (define git (find-executable-path "git"))
  (define out (open-output-string))
  (define listed?
    (and git
         (parameterize ([current-directory root]
                        [current-output-port out])
           (apply system* git "ls-tree" "-r" "HEAD" "--" inputs))))
  (cond
    [(not listed?)
     (refuse "git ls-tree -r HEAD -- ~a failed in ~a" (string-join inputs " ") root)]
    [else
     (natives-inputs-digest (string-split (get-output-string out) "\n") excluded)]))

(define/contract (natives-plan checked previous)
  (-> (hash/c symbol? string?) (or/c #f jsexpr?) (values boolean? string?))
  (match-define (hash-table ['tag tag]
                           ['version version]
                           ['native-version native-version]
                           ['abi-stamp stamp]
                           ['natives-inputs inputs]
                           [_ _] ...)
    checked)
  (match previous
    [#f (values #t tag)]
    [(hash-table ['tag (? string? released-tag)]
                 ['version (? string? released-version)]
                 ['native-version (? string? released)]
                 ['abi-stamp (? string? released-stamp)]
                 ['natives-inputs (? string? released-inputs)]
                 ['natives-tag (? string? natives-tag)]
                 [_ _] ...)
     (cond
       [(version<? version released-version)
        (refuse "version ~a is older than ~a, released at ~a" version released-version released-tag)]
       [(string=? tag released-tag)
        (refuse "~a is the previous release's tag already" tag)]
       [(version<? native-version released)
        (refuse "native-version ~a is older than ~a, released at ~a"
                native-version released natives-tag)]
       [(version<? released native-version) (values #t tag)]
       [(not (string=? stamp released-stamp))
        (refuse
         (string-append
          "the ABI stamp is ~a, but the natives released at ~a have ~a and the same"
          " native-version, ~a; bump native-version in torch/info.rkt, torch's #:platform"
          " deps and pkgs/*/info.rkt")
         stamp natives-tag released-stamp native-version)]
       [(not (string=? inputs released-inputs))
        (refuse
         (string-append
          "the natives' inputs (scripts/release/release.rktd's natives-inputs) changed since ~a,"
          " but native-version is still ~a; bump it in torch/info.rkt, torch's #:platform deps"
          " and pkgs/*/info.rkt")
         natives-tag native-version)]
       [else (values #f natives-tag)])]
    [_
     (refuse (string-append
              "the previous manifest.json needs the strings tag, version, native-version,"
              " abi-stamp, natives-inputs and natives-tag; it has ~s")
             previous)]))
