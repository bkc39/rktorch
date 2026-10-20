#lang racket/base
(require (only-in racket/date find-seconds)
         (only-in racket/file file->string)
         (only-in racket/list first index-of last take-right)
         (only-in racket/match match-define)
         (only-in racket/string string-split))

(define pkg "torch")
(define test-limit-seconds 600)

(define clock-rx
  #px"The time is now \\w+, (\\d+) (\\w+) (\\d+) (\\d+):(\\d+):(\\d+) ([-+])(\\d\\d)(\\d\\d)")
(define months '("Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"))

(define (section title)
  (printf "\n==================== ~a ====================\n" title))

(define (show-log work entry key #:tail tail)
  (define log (hash-ref entry key #f))
  (when log
    (section (format "~a: ~a" key log))
    (define lines (string-split (file->string (build-path work log)) "\n"))
    (for-each displayln (if (and tail (> (length lines) tail))
                            (take-right lines tail)
                            lines))))

(define (clock->utc-seconds clock)
  (match-define (list _ day month year hour minute second sign zone-hours zone-minutes) clock)
  (define zone-offset
    (* (if (string=? sign "-") -1 1)
       (+ (* 3600 (string->number zone-hours)) (* 60 (string->number zone-minutes)))))
  (- (find-seconds (string->number second)
                   (string->number minute)
                   (string->number hour)
                   (string->number day)
                   (add1 (index-of months month))
                   (string->number year)
                   #f)
     zone-offset))

(define (test-seconds work entry)
  (define log (or (hash-ref entry 'test-success-log #f) (hash-ref entry 'test-failure-log #f)))
  (define clocks
    (if log
        (regexp-match* clock-rx (file->string (build-path work log)) #:match-select values)
        '()))
  (and (>= (length clocks) 2)
       (- (clock->utc-seconds (last clocks)) (clock->utc-seconds (first clocks)))))

(define (rendered-here? doc)
  (eq? (prefab-struct-key doc) 'doc/main))

(module+ main
  (match-define (vector work-arg) (current-command-line-arguments))
  (define work (path->complete-path work-arg))
  (define summary-file (build-path work "summary.rktd"))
  (unless (file-exists? summary-file)
    (raise-user-error 'report "no summary.rktd in ~a; the harness stopped before its summary step"
                      work))
  (define entry (hash-ref (call-with-input-file summary-file read) pkg #f))
  (unless entry
    (raise-user-error 'report "~a has no entry for ~a" summary-file pkg))
  (for ([key (in-list '(failure-log min-failure-log dep-failure-log
                        success-log test-failure-log test-success-log))])
    (show-log work entry key #:tail (and (eq? key 'success-log) 60)))
  (define docs (hash-ref entry 'docs '()))
  (define conflicts-log (hash-ref entry 'conflicts-log #f))
  (define seconds (test-seconds work entry))
  (define build-ok? (and (hash-ref entry 'success-log #f)
                         (not (hash-ref entry 'failure-log #f))
                         #t))
  (define dependencies-ok? (and build-ok? (not (hash-ref entry 'dep-failure-log #f))))
  (define conflicts-ok? (not conflicts-log))
  (define docs-ok? (and (pair? docs) (andmap rendered-here? docs)))
  (define tests-ok? (and (hash-ref entry 'test-success-log #f)
                         seconds
                         (<= seconds test-limit-seconds)
                         #t))
  (section "columns")
  (for ([column (in-list `(("build" ,build-ok?)
                           ("dependencies" ,dependencies-ok?)
                           ("conflicts" ,conflicts-ok?)
                           ("docs" ,docs-ok?)
                           ("tests" ,tests-ok?)))])
    (match-define (list name ok?) column)
    (printf "~a: ~a\n" name (if ok? "ok" "FAILED")))
  (printf "docs: ~s\n" docs)
  (when conflicts-log
    (printf "conflicts: ~s\n" conflicts-log))
  (printf "test time: ~a (the server stops tests at ~a s)\n"
          (if seconds (format "~a s" seconds) "unknown")
          test-limit-seconds)
  (unless (and build-ok? dependencies-ok? conflicts-ok? docs-ok? tests-ok?)
    (exit 1)))
