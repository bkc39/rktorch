#lang racket/base
(require (only-in pkg/lib installed-pkg-names pkg-directory)
         (only-in pkg/name package-source->name+type)
         (only-in racket/contract/base ->)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->lines)
         (only-in racket/list partition remove-duplicates)
         (only-in racket/match match)
         (only-in racket/path file-name-from-path path-has-extension?)
         (only-in racket/sequence sequence->list)
         (only-in racket/string string-join)
         (only-in setup/getinfo get-info/full)
         (only-in version/utils version<=?))

(define manifest-name ".info-deps")

(define (dep->string dep)
  (format "~s" dep))

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define (dep-option dep kw)
  (match dep
    [(list _ (? string? version)) (and (eq? kw '#:version) version)]
    [(cons _ options)
     (match (member kw options)
       [(list* _ value _) value]
       [_ #f])]
    [_ #f]))

(define (platform-dep? dep)
  (and (dep-option dep '#:platform) #t))

(define (source-dep? dep)
  (define-values (_name type) (package-source->name+type (dep-name dep) #f))
  (not (eq? type 'name)))

(define (info-deps info-dir)
  (define info
    (or (get-info/full info-dir)
        (raise-user-error 'racket-deps "no info.rkt in ~a" info-dir)))
  (append (info 'deps (λ () '())) (info 'build-deps (λ () '()))))

(define (declared-deps info-dir)
  (define have (installed-pkg-names #:scope 'installation))
  (partition (λ (dep) (not (member (dep-name dep) have)))
             (remove-duplicates (info-deps info-dir))))

(define/contract (catalog-deps info-dir)
  (-> directory-exists? list?)
  (define-values (wanted _installed) (declared-deps info-dir))
  (define deps (sort wanted string<? #:key dep->string))
  (define platform-dep (findf platform-dep? deps))
  (define source-dep (findf source-dep? deps))
  (cond
    [(null? deps)
     (raise-user-error 'racket-deps
                       "~a declares no package outside the Racket distribution"
                       info-dir)]
    [source-dep
     (raise-user-error 'racket-deps
                       "~a is a package source; racket-deps fetches catalog packages by name"
                       (dep-name source-dep))]
    [platform-dep
     (raise-user-error 'racket-deps
                       "~a has #:platform ~s; platform-qualified deps are not fetched from the catalog"
                       (dep-name platform-dep)
                       (dep-option platform-dep '#:platform))]
    [else deps]))

(define (check-version dep pkg-dir where)
  (define required (dep-option dep '#:version))
  (define info (and required pkg-dir (get-info/full pkg-dir)))
  (define found (and info (info 'version (λ () "0.0"))))
  (cond
    [(not required) (void)]
    [(and found (version<=? required found)) (void)]
    [else
     (raise-user-error 'racket-deps
                       "~a requires #:version ~a; found ~a ~a"
                       (dep-name dep)
                       required
                       (or found "no package")
                       where)]))

(define/contract (check-current info-dir deps-dir)
  (-> directory-exists? directory-exists? void?)
  (define manifest (build-path deps-dir manifest-name))
  (define built (if (file-exists? manifest) (file->lines manifest) '()))
  (define-values (_wanted installed) (declared-deps info-dir))
  (define deps (catalog-deps info-dir))
  (define declared (map dep->string deps))
  (define fetched-from (format "in ~a" deps-dir))
  (cond
    [(equal? built declared)
     (for ([dep (in-list installed)])
       (check-version dep (pkg-directory (dep-name dep)) "(installation)"))
     (for ([dep (in-list deps)])
       (check-version dep (build-path deps-dir (dep-name dep)) fetched-from))]
    [else
     (raise-user-error 'racket-deps
                       (string-append
                        "info.rkt changed; bump racket-deps outputHash in flake.nix\n"
                        "  declared: ~a\n  built:    ~a")
                       (string-join declared " ")
                       (string-join built " "))]))

(define/contract (base-version info-dir)
  (-> directory-exists? string?)
  (define base (findf (λ (dep) (equal? (dep-name dep) "base")) (info-deps info-dir)))
  (or (and base (dep-option base '#:version))
      (raise-user-error 'racket-deps "~a declares no (\"base\" #:version ...) floor" info-dir)))

(define (build-output? p)
  (or (link-exists? p)
      (equal? (file-name-from-path p) (string->path "compiled"))
      (ormap (λ (ext) (path-has-extension? p ext))
             '(#".zo" #".dep" #".sxref"))))

(define/contract (guard tree)
  (-> directory-exists? void?)
  (define tree-paths (in-directory tree (λ (d) (not (link-exists? d)))))
  (match (filter build-output? (sequence->list tree-paths))
    ['() (void)]
    [found
     (for-each displayln found)
     (raise-user-error 'racket-deps "the paths above are build output, not source")]))

(module+ main
  (match (current-command-line-arguments)
    [(vector "manifest" info-dir)
     (for-each (λ (dep) (displayln (dep->string dep))) (catalog-deps info-dir))]
    [(vector "names" info-dir)
     (for-each (λ (dep) (displayln (dep-name dep))) (catalog-deps info-dir))]
    [(vector "base-version" info-dir) (displayln (base-version info-dir))]
    [(vector "installed")
     (for-each displayln (installed-pkg-names #:scope 'user))]
    [(vector "check" info-dir deps-dir) (check-current info-dir deps-dir)]
    [(vector "guard" tree) (guard tree)]
    [args
     (raise-user-error
      'racket-deps
      (string-append
       "usage: manifest <info-dir> | names <info-dir> | base-version <info-dir> | installed\n"
       "     | check <info-dir> <deps-dir> | guard <tree>\n"
       "got ~s")
      args)]))
