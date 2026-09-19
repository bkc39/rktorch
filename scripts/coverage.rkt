#! /usr/bin/env racket
#lang racket/base

;; Expression coverage for the library, measured by `raco cover` driving the
;; repo's own test suite.  Run it from the repository root inside a dev shell:
;;
;;   nix develop .#ci --command racket scripts/coverage.rkt
;;   nix develop .#ci --command racket scripts/coverage.rkt --changed
;;
;; `cover` writes only HTML, so the numbers come back out of its report.

(require (only-in racket/cmdline command-line) ;; noqa
         (only-in racket/file file->string)
         (only-in racket/format ~a ~r)
         (only-in racket/list first second take)
         (only-in racket/port with-output-to-string)
         (only-in racket/string string-join string-prefix? string-split
                  string-trim)
         (only-in racket/system system* system*/exit-code))

;; The floor CI and the local loop hold the library to. Raise it as the
;; categories in #173 land; never lower it to make a change pass.
(define coverage-floor 96.0)

(define report-dir (make-parameter "coverage"))
(define floor% (make-parameter coverage-floor))
(define changed-only? (make-parameter #f))
(define run? (make-parameter #t))

(define (die fmt . args)
  (apply eprintf fmt args)
  (exit 1))

(define (raco)
  (or (find-executable-path "raco") (die "coverage: no raco on PATH\n")))

;; A failed git call warns and contributes nothing: loud, because a quietly
;; shorter list understates what --changed covers, but not fatal, because the
;; merge-base comparison has nothing to compare against in a shallow clone
;; (CI checks out at fetch-depth 1) and --changed is a report either way.
(define (git . args)
  (define ok? #f)
  (define out
    (with-output-to-string
      (lambda ()
        (set! ok? (apply system*
                         (or (find-executable-path "git") (die "no git\n"))
                         args)))))
  (cond
    [ok? (string-split out "\n")]
    [else
     (eprintf "coverage: git ~a failed; --changed lists only what it could see\n"
              (string-join args " "))
     '()]))

;;; Running

(define (run-cover!)
  (unless (getenv "PLTCOLLECTS")
    (putenv "PLTCOLLECTS" (string-append (path->string (current-directory)) ":")))
  (displayln "running raco cover over torch/ (instrumented; runs the suite)")
  (zero? (system*/exit-code (raco) "cover" "-b" "-d" (report-dir) "torch/")))

;;; Reading the report

(struct entry (path pct covered missed total) #:transparent)

;; `\\s*` at every tag boundary and an optional `%`: cover emits this table
;; compactly today, and the gate should survive it pretty-printing tomorrow.
;; In a pregexp `\\s` spans newlines, so no (?s:) is needed here.
(define (cell name pattern)
  (string-append "\\s*<td class=\"" name "\">\\s*" pattern "\\s*</td>"))

(define row-px
  (pregexp
   (string-append
    "<tr class=\"file-info\">\\s*<td class=\"file-name\">\\s*"
    "<a href=\"[^\"]*\">([^<]+)</a>\\s*</td>"
    (cell "coverage-percentage" "([0-9.]+)%?")
    (cell "covered-expressions" "([0-9]+)")
    (cell "uncovered-expressions" "([0-9]+)")
    (cell "total-expressions" "([0-9]+)"))))

(define (parse-entries html)
  (for/list ([r (in-list (regexp-match* row-px html #:match-select values))])
    (entry (second r)
           (string->number (list-ref r 2))
           (string->number (list-ref r 3))
           (string->number (list-ref r 4))
           (string->number (list-ref r 5)))))

(define (read-entries)
  (define index (build-path (report-dir) "index.html"))
  (unless (file-exists? index)
    (die "coverage: no report at ~a\n" index))
  (define entries (parse-entries (file->string index)))
  (when (null? entries)
    (die "coverage: could not parse ~a (did cover's html change?)\n" index))
  entries)

(define (area path)
  (define parts (string-split path "/"))
  (cond
    [(string-prefix? path "torch/foreign/raw/") "torch/foreign/raw"]
    [(= 2 (length parts)) "torch (facades)"]
    [else (string-join (take parts 2) "/")]))

;; The lines of a file that hold at least one uncovered expression.
;; (?s:) so a line's spans may straddle newlines.
(define line-px
  (pregexp "(?s:<div\\s+class=\"line\"\\s+id=\"([0-9]+)\"\\s*>(.*?)</div>)"))

(define (parse-uncovered-lines html)
  (for/list ([m (in-list (regexp-match* line-px html #:match-select values))]
             ;; m is (whole-match line-number line-content)
             #:when (regexp-match? #rx"class=\"uncovered\"" (list-ref m 2)))
    (string->number (second m))))

;; A list of line numbers, or 'missing when cover wrote no report for the
;; file: an empty list there would read as "nothing left to test".
(define (uncovered-lines path)
  (define html
    (build-path (report-dir)
                (string-append (substring path 0 (- (string-length path) 4))
                               ".html")))
  (if (file-exists? html)
      (parse-uncovered-lines (file->string html))
      'missing))

;;; Reporting

(define (pct covered total)
  (if (zero? total) 100.0 (* 100.0 (/ covered total))))

(define (print-areas entries)
  (define areas
    (for/fold ([h (hash)]) ([e (in-list entries)])
      (define k (area (entry-path e)))
      (define prev (hash-ref h k (list 0 0 0)))
      (hash-set h k (list (+ (first prev) (entry-covered e))
                          (+ (second prev) (entry-total e))
                          (add1 (list-ref prev 2))))))
  (printf "\n~a ~a ~a ~a\n" (~a "area" #:min-width 22) (~a "files" #:width 6)
          (~a "cov%" #:width 8) (~a "missed" #:width 7))
  (for ([k (in-list (sort (hash-keys areas) string<?))])
    (define v (hash-ref areas k))
    (printf "~a ~a ~a ~a\n"
            (~a k #:min-width 22)
            (~a (list-ref v 2) #:width 6)
            (~a (~r (pct (first v) (second v)) #:precision '(= 1)) #:width 8)
            (~a (- (second v) (first v)) #:width 7))))

(define (print-entry-row e)
  (printf "  ~a ~a  ~a missed of ~a\n"
          (~a (entry-path e) #:min-width 42)
          (~a (~r (entry-pct e) #:precision '(= 1)) #:width 6)
          (entry-missed e) (entry-total e)))

(define (print-worst entries n)
  (define worst
    (take (sort entries > #:key entry-missed)
          (min n (length entries))))
  (display "\nfiles with the most uncovered expressions\n")
  (for ([e (in-list worst)] #:unless (zero? (entry-missed e)))
    (print-entry-row e)))

(define (print-changed entries)
  ;; Two queries, not three: `diff HEAD` already covers the index as well as
  ;; the working tree.
  (define touched
    (for/list ([f (in-list (append (git "diff" "--name-only" "origin/master...HEAD")
                                   (git "diff" "--name-only" "HEAD")))]
               #:when (regexp-match? #rx"^torch/.*[.]rkt$" (string-trim f)))
      (string-trim f)))
  (define mine
    (for/list ([e (in-list entries)] #:when (member (entry-path e) touched)) e))
  (cond
    [(null? mine)
     (display "\nno changed library file appears in the report\n")]
    [else
     (display "\nchanged files\n")
     (for ([e (in-list (sort mine > #:key entry-missed))])
       (print-entry-row e)
       (define ls (uncovered-lines (entry-path e)))
       (cond
         [(eq? ls 'missing)
          (printf "      uncovered lines: unknown, cover wrote no report\n")]
         [(null? ls) (void)]
         [else (printf "      uncovered lines: ~a\n"
                       (string-join (map number->string ls) " "))]))]))

;;; Main

(module+ main
  (command-line
   #:program "coverage"
   #:once-each
   [("-d" "--directory") dir "report directory (default: coverage)"
                             (report-dir dir)]
   [("--floor") f "fail below this percentage"
                  (define n (string->number f))
                  (unless (real? n) (die "coverage: --floor needs a number, got ~a\n" f))
                  (floor% n)]
   [("--changed") "also list uncovered lines in files this branch touches"
                  (changed-only? #t)]
   [("--no-run") "read an existing report instead of running the suite"
                 (run? #f)]
   #:args ()
   (define suite-ok? (or (not (run?)) (run-cover!)))
   (define entries (read-entries))
   (define covered (for/sum ([e (in-list entries)]) (entry-covered e)))
   (define total (for/sum ([e (in-list entries)]) (entry-total e)))
   (define overall (pct covered total))
   (printf "\nexpression coverage ~a%  (~a of ~a expressions, ~a files)\n"
           (~r overall #:precision '(= 2)) covered total (length entries))
   (print-areas entries)
   (print-worst entries 12)
   (when (changed-only?) (print-changed entries))
   (printf "\nreport: ~a/index.html\n" (report-dir))
   ;; cover runs every file in ONE process; raco test forks per file. A test
   ;; that asserts on accumulated ledger or GC state can fail here and pass
   ;; there, so `raco test` stays the authority on correctness and this only
   ;; says the numbers may be short.
   (unless suite-ok?
     (display "note: the suite reported failures under cover; run raco test\n")
     (display "      to judge them, and treat the numbers above as a floor\n"))
   (cond
     [(< overall (floor%))
      (eprintf "\ncoverage ~a% is below the ~a% floor\n"
               (~r overall #:precision '(= 2)) (floor%))
      (exit 1)]
     [else
      (printf "floor ~a%: met\n" (floor%))])))

;;; Tests
;;
;; The report is scraped, so the shapes it is scraped from are pinned here:
;; run with `raco test scripts/coverage.rkt`.

(module+ test
  (require (only-in rackunit check-equal? test-case))

  (define (row name percent cov unc total)
    (string-append
     "<tr class=\"file-info\"><td class=\"file-name\">"
     "<a href=\"" name ".html\">" name ".rkt</a></td>"
     "<td class=\"coverage-percentage\">" percent "</td>"
     "<td class=\"covered-expressions\">" cov "</td>"
     "<td class=\"uncovered-expressions\">" unc "</td>"
     "<td class=\"total-expressions\">" total "</td></tr>"))

  (test-case "the index table parses as cover emits it today"
    (define es
      (parse-entries
       (string-append "<tbody>" (row "torch/nn/linear" "100" "185" "0" "185")
                      (row "torch/foreign/ops" "94.8" "1830" "100" "1930")
                      "</tbody>")))
    (check-equal? (map entry-path es)
                  '("torch/nn/linear.rkt" "torch/foreign/ops.rkt"))
    (check-equal? (map entry-covered es) '(185 1830))
    (check-equal? (map entry-missed es) '(0 100))
    (check-equal? (map entry-total es) '(185 1930))
    (check-equal? (map entry-pct es) '(100 94.8)))

  (test-case "and survives a pretty-printed table with percent signs"
    (define es
      (parse-entries
       (string-append
        "<tr class=\"file-info\">\n  <td class=\"file-name\">\n"
        "    <a href=\"x.html\">torch/nn/x.rkt</a>\n  </td>\n"
        "  <td class=\"coverage-percentage\"> 81.5% </td>\n"
        "  <td class=\"covered-expressions\"> 110 </td>\n"
        "  <td class=\"uncovered-expressions\"> 25 </td>\n"
        "  <td class=\"total-expressions\"> 135 </td>\n</tr>")))
    (check-equal? (map entry-path es) '("torch/nn/x.rkt"))
    (check-equal? (map entry-missed es) '(25))
    (check-equal? (map entry-pct es) '(81.5)))

  (test-case "only the lines carrying an uncovered span are reported"
    (check-equal?
     (parse-uncovered-lines
      (string-append
       "<div class=\"line\" id=\"24\"><span class=\"covered\">(if</span></div>"
       "<div class=\"line\" id=\"25\"><span class=\"uncovered\">(to-device</span>"
       "<span class=\"uncovered\">x)</span></div>"
       "<div class=\"line\" id=\"26\"><span class=\"irrelevant\"> </span></div>"))
     '(25)))

  (test-case "a file's area is its directory, with raw/ kept apart"
    (check-equal? (map area '("torch/main.rkt"
                              "torch/nn/linear.rkt"
                              "torch/foreign/ops.rkt"
                              "torch/foreign/raw/memory.rkt"))
                  '("torch (facades)" "torch/nn" "torch/foreign"
                    "torch/foreign/raw"))))
