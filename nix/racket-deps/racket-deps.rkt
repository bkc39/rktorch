#lang racket/base
(require (only-in racket/contract/base -> listof)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->lines)
         (only-in racket/list remove-duplicates)
         (only-in racket/match match)
         (only-in racket/path file-name-from-path path-has-extension?)
         (only-in racket/string string-join)
         (only-in pkg/lib installed-pkg-names)
         (only-in setup/getinfo get-info/full))

(define manifest-name ".info-deps")

(define (dep-name dep)
  (match dep
    [(cons name _) name]
    [name name]))

(define/contract (catalog-deps info-dir)
  (-> directory-exists? (listof string?))
  (define info
    (or (get-info/full info-dir)
        (raise-user-error 'racket-deps "no info.rkt in ~a" info-dir)))
  (define have (installed-pkg-names #:scope 'installation))
  (define wanted
    (for/list ([dep (append (info 'deps (λ () '())) (info 'build-deps (λ () '())))]
               #:unless (member (dep-name dep) have))
      (format "~s" dep)))
  (match (sort (remove-duplicates wanted) string<?)
    ['()
     (raise-user-error 'racket-deps
                       "~a declares no package outside the Racket distribution"
                       info-dir)]
    [specs specs]))

(define/contract (check-current info-dir deps-dir)
  (-> directory-exists? directory-exists? void?)
  (define manifest (build-path deps-dir manifest-name))
  (define built (if (file-exists? manifest) (file->lines manifest) '()))
  (define declared (catalog-deps info-dir))
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
      (for/or ([ext '(#".zo" #".dep" #".sxref")])
        (path-has-extension? p ext))))

(define/contract (guard tree)
  (-> directory-exists? void?)
  (match (for/list ([p (in-directory tree (λ (d) (not (link-exists? d))))]
                    #:when (build-output? p))
           p)
    ['() (void)]
    [found
     (for-each displayln found)
     (raise-user-error 'racket-deps "the paths above are build output, not source")]))

(module+ main
  (match (current-command-line-arguments)
    [(vector "manifest" info-dir) (for-each displayln (catalog-deps info-dir))]
    [(vector "names" info-dir)
     (for ([spec (catalog-deps info-dir)])
       (displayln (dep-name (read (open-input-string spec)))))]
    [(vector "check" info-dir deps-dir) (check-current info-dir deps-dir)]
    [(vector "guard" tree) (guard tree)]
    [args
     (raise-user-error 'racket-deps
                       "usage: manifest|names <info-dir>, check <info-dir> <deps-dir>, guard <tree>; got ~s"
                       args)]))
