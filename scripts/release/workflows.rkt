#lang racket/base
(require (only-in racket/contract/base -> cons/c listof)
         (only-in racket/contract/region define/contract)
         (only-in racket/list append* dropf remove-duplicates splitf-at takef)
         (only-in racket/match match match*)
         (only-in racket/string string-join string-split string-trim)
         (only-in "checks.rkt" refuse))

(provide (struct-out job)
         called-workflows
         unpinned-actions
         unguarded-jobs
         workflow-jobs)

(struct job (name needs condition uses writes? actions) #:transparent)

(define (indentation line)
  (string-length (car (regexp-match #px"^ *" line))))

(define (content? line)
  (not (regexp-match? #px"^\\s*(#.*)?$" line)))

(define (jobs-block lines)
  (match (dropf lines (λ (line) (not (regexp-match? #px"^jobs:\\s*(#.*)?$" line))))
    ['() (refuse "the workflow has no top-level jobs: key")]
    [(cons _ rest)
     (filter content? (takef rest (λ (line) (or (not (content? line)) (positive? (indentation line))))))]))

(define job-header #px"^  ([A-Za-z_][A-Za-z0-9_-]*):\\s*(#.*)?$")

(define (split-jobs lines)
  (for/fold ([jobs '()]
             #:result (reverse (for/list ([j (in-list jobs)])
                                 (cons (car j) (reverse (cdr j))))))
            ([line (in-list lines)])
    (match* ((regexp-match job-header line) jobs)
      [((list _ name _) _) (cons (list name) jobs)]
      [(#f (cons (cons name body) more))
       #:when (>= (indentation line) 4)
       (cons (list* name line body) more)]
      [(_ _) (refuse "expected a job key indented by two spaces under jobs:, got ~s" line)])))

(define (job-keys body)
  (match body
    ['() (hash)]
    [(cons line rest)
     (define-values (continued more) (splitf-at rest (λ (next) (> (indentation next) 4))))
     (define keys (job-keys more))
     (match (regexp-match #px"^    ([A-Za-z_-]+):\\s*(.*)$" line)
       [(list _ key inline) (hash-set keys key (cons (string-trim inline) continued))]
       [#f (refuse "expected a job's key: indented by four spaces, got ~s" line)])]))

(define (unquote-scalar text)
  (match (regexp-match #px"^\\s*([\"']?)(.*?)\\1\\s*(#.*)?$" text)
    [(list _ _ value _) value]))

(define (needs-value value)
  (match value
    [#f '()]
    [(cons "" items)
     (for/list ([item (in-list items)])
       (match (regexp-match #px"^\\s*-\\s*(.*)$" item)
         [(list _ name) (unquote-scalar name)]
         [#f (refuse "expected a - item under needs:, got ~s" item)]))]
    [(cons (pregexp #px"^\\[(.*)\\]\\s*(#.*)?$" (list _ items _)) _)
     (map unquote-scalar (string-split items ","))]
    [(cons (and inline (pregexp #px"^\\[")) _)
     (refuse "expected needs: [...] closed on its line, got ~s" inline)]
    [(cons inline _) (list (unquote-scalar inline))]))

(define (scalar-value value)
  (match value
    [#f #f]
    [(cons inline continued)
     (define head (if (regexp-match? #px"^[>|][+-]?$" inline) '() (list inline)))
     (string-join (append head (map string-trim continued)) " ")]))

(define/contract (workflow-jobs lines)
  (-> (listof string?) (listof job?))
  (for/list ([entry (in-list (split-jobs (jobs-block lines)))])
    (match entry
      [(cons name body)
       (define keys (job-keys body))
       (define uses (scalar-value (hash-ref keys "uses" #f)))
       (job name
            (needs-value (hash-ref keys "needs" #f))
            (scalar-value (hash-ref keys "if" #f))
            (and uses (unquote-scalar uses))
            (regexp-match? #px"\\bwrite" (or (scalar-value (hash-ref keys "permissions" #f)) ""))
            (step-actions body))])))

(define (step-actions body)
  (for*/list ([line (in-list body)]
              [found (in-value (regexp-match #px"^ {6,}(?:- +)?uses:\\s*(.*)$" line))]
              #:when found)
    (unquote-scalar (cadr found))))

(define (needs-any? j names)
  (for/or ([need (in-list (job-needs j))])
    (member need names)))

(define (skippable jobs)
  (let grow ([names (for/list ([j (in-list jobs)] #:when (job-condition j)) (job-name j))])
    (define more
      (for/list ([j (in-list jobs)]
                 #:unless (member (job-name j) names)
                 #:when (needs-any? j names))
        (job-name j)))
    (if (null? more) names (grow (append names more)))))

(define (guarded? condition)
  (and (regexp-match? #px"\\b(always|cancelled|failure)\\s*\\(" condition)
       (not (regexp-match? #px"\\bsuccess\\s*\\(" condition))))

(define (skip-conditions by-name name seen)
  (match (hash-ref by-name name #f)
    [#f '()]
    [(job _ _ (pregexp #px"\\balways\\s*\\(") _ _ _) '()]
    [(job _ _ (? string? condition) _ _ _) (list condition)]
    [(job _ needs #f _ _ _)
     (remove-duplicates
      (append* (for/list ([need (in-list needs)] #:unless (member need seen))
                 (skip-conditions by-name need (cons need seen)))))]))

(define/contract (unguarded-jobs jobs)
  (-> (listof job?) (listof string?))
  (define names (skippable jobs))
  (define by-name (for/hash ([j (in-list jobs)]) (values (job-name j) j)))
  (for/list ([j (in-list jobs)]
             #:when (match j
                      [(job name _ #f _ _ _)
                       (>= (length (skip-conditions by-name name (list name))) 2)]
                      [(job _ _ condition _ _ _)
                       (and (not (guarded? condition)) (needs-any? j names))]))
    (job-name j)))

(define/contract (called-workflows jobs)
  (-> (listof job?) (listof string?))
  (remove-duplicates
   (for*/list ([j (in-list jobs)]
               [uses (in-value (job-uses j))]
               #:when (and uses (regexp-match? #px"^\\./\\.github/workflows/[^/]+\\.ya?ml$" uses)))
     (substring uses 2))))

(define/contract (unpinned-actions jobs)
  (-> (listof job?) (listof (cons/c string? string?)))
  (for*/list ([j (in-list jobs)]
              [action (in-list (job-actions j))]
              #:unless (regexp-match? #px"^\\./|@[0-9a-f]{40}$" action))
    (cons (job-name j) action)))
