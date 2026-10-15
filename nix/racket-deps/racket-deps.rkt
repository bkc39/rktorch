#lang racket/base
(require (only-in racket/contract/base ->)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->lines)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in racket/path file-name-from-path path-has-extension?)
         (only-in racket/sequence sequence->list)
         (only-in racket/string string-join)
         (only-in pkg/lib installed-pkg-names)
         (only-in setup/getinfo get-info/full))

(define manifest-name ".info-deps")

(define (dep->string dep)
  (format "~s" dep))

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define/contract (catalog-deps info-dir)
  (-> directory-exists? list?)
  (define info
    (or (get-info/full info-dir)
        (raise-user-error 'racket-deps "no info.rkt in ~a" info-dir)))
  (define have (installed-pkg-names #:scope 'installation))
  (define wanted
    (for/list ([dep (append (info 'deps (λ () '())) (info 'build-deps (λ () '())))]
               #:unless (member (dep-name dep) have))
      dep))
  (match (sort (remove-duplicates wanted) string<? #:key dep->string)
    ['()
     (raise-user-error 'racket-deps
                       "~a declares no package outside the Racket distribution"
                       info-dir)]
    [deps deps]))

(define/contract (check-current info-dir deps-dir)
  (-> directory-exists? directory-exists? void?)
  (define manifest (build-path deps-dir manifest-name))
  (define built (if (file-exists? manifest) (file->lines manifest) '()))
  (define declared (map dep->string (catalog-deps info-dir)))
  (unless (equal? built declared)
    (raise-user-error 'racket-deps
                      (string-append
                       "info.rkt changed; bump racket-deps outputHash in flake.nix\n"
                       "  declared: ~a\n  built:    ~a")
                      (string-join declared " ")
                      (string-join built " "))))

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
    [(vector "installed")
     (for-each displayln (installed-pkg-names #:scope 'user))]
    [(vector "check" info-dir deps-dir) (check-current info-dir deps-dir)]
    [(vector "guard" tree) (guard tree)]
    [args
     (raise-user-error
      'racket-deps
      (string-append
       "usage: manifest <info-dir> | names <info-dir> | installed\n"
       "     | check <info-dir> <deps-dir> | guard <tree>\n"
       "got ~s")
      args)]))
