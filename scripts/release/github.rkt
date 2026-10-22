#lang racket/base
(require (only-in json jsexpr?)
         (only-in racket/contract/base -> ->* listof or/c)
         (only-in racket/contract/region define/contract)
         (only-in racket/list append-map filter-map)
         (only-in racket/match == match match*)
         (only-in version/utils version<?)
         (only-in "checks.rkt" refuse))

(provide download-root
         download-url
         previous-release
         published-releases
         unmet-required-checks)

(define/contract (download-root repository tag)
  (-> string? string? string?)
  (format "https://github.com/~a/releases/download/~a" repository tag))

(define/contract (download-url repository tag file)
  (-> string? string? string? string?)
  (string-append (download-root repository tag) "/" file))

(define (published-release release)
  (match release
    [(hash-table ['draft #f]
                 ['tag_name (? string?)]
                 ['assets (list _ ... (hash-table ['name "manifest.json"] [_ _] ...) _ ...)]
                 [_ _] ...)
     release]
    [_ #f]))

(define (published releases except)
  (filter-map (λ (release)
                (match (published-release release)
                  [(hash-table ['tag_name (== except)] [_ _] ...) #f]
                  [found found]))
              releases))

(define/contract (published-releases releases #:except [except #f])
  (->* [(listof jsexpr?)] [#:except (or/c #f string?)] (listof string?))
  (map (λ (release) (hash-ref release 'tag_name)) (published releases except)))

(define (native-version-of manifest-of tag)
  (match (manifest-of tag)
    [(hash-table ['native-version (? string? native-version)] [_ _] ...) native-version]
    [manifest (refuse "the manifest.json of ~a has no native-version string: ~s" tag manifest)]))

(define/contract (previous-release releases manifest-of #:except [except #f])
  (->* [(listof jsexpr?) (-> string? jsexpr?)] [#:except (or/c #f string?)] (or/c #f string?))
  (define candidates
    (for/list ([release (in-list (published releases except))])
      (define tag (hash-ref release 'tag_name))
      (list tag (native-version-of manifest-of tag) (hash-ref release 'published_at ""))))
  (define (newer? a b)
    (match* (a b)
      [((list _ native-a date-a) (list _ native-b date-b))
       (or (version<? native-b native-a)
           (and (string=? native-a native-b) (string>? date-a date-b)))]))
  (match (sort candidates newer?)
    ['() #f]
    [(cons (list tag _ _) _) tag]))

(define (required-contexts rules)
  (append-map (λ (rule)
                (match rule
                  [(hash-table ['type "required_status_checks"]
                               ['parameters (hash-table ['required_status_checks checks] [_ _] ...)]
                               [_ _] ...)
                   checks]
                  [_ '()]))
              rules))

(define (latest-run runs context)
  (define matching
    (filter (λ (run)
              (and (equal? (hash-ref run 'name #f) (hash-ref context 'context #f))
                   (app-matches? run context)))
            runs))
  (match (sort matching > #:key (λ (run) (hash-ref run 'id 0)))
    ['() #f]
    [(cons latest _) latest]))

(define (app-matches? run context)
  (match* (context run)
    [((hash-table ['integration_id (? exact-integer? app)] [_ _] ...)
      (hash-table ['app (hash-table ['id app-id] [_ _] ...)] [_ _] ...))
     (equal? app app-id)]
    [((hash-table ['integration_id (? exact-integer?)] [_ _] ...) _) #f]
    [(_ _) #t]))

(define/contract (unmet-required-checks rules check-run-pages)
  (-> (listof jsexpr?) (listof jsexpr?) (listof string?))
  (define runs (append-map (λ (page) (hash-ref page 'check_runs '())) check-run-pages))
  (match (required-contexts rules)
    ['() (refuse "master's rules require no status checks, so nothing shows the tagged commit passed CI")]
    [contexts
     (for*/list ([context (in-list contexts)]
                 [run (in-value (latest-run runs context))]
                 #:unless (match run
                            [(hash-table ['conclusion "success"] [_ _] ...) #t]
                            [_ #f]))
       (format "~a (~a)"
               (hash-ref context 'context)
               (match run
                 [#f "no run"]
                 [(hash-table ['status "completed"] ['conclusion conclusion] [_ _] ...) conclusion]
                 [(hash-table ['status status] [_ _] ...) status])))]))
