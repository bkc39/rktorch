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
         (only-in racket/match match match*)
         (only-in racket/port with-output-to-string)
         (only-in racket/string string-join string-prefix? string-split
                  string-suffix? string-trim)
         (only-in racket/system system* system*/exit-code)
         (only-in setup/getinfo get-info/full))

;; The floor CI and the local loop hold the library to. Raise it as the
;; categories in #173 land; never lower it to make a change pass.
(define coverage-floor 96.0)

(define report-dir (make-parameter "coverage"))
(define floor% (make-parameter coverage-floor))
(define changed-only? (make-parameter #f))
(define run? (make-parameter #t))
(define file-seconds (make-parameter 300))

;; Instrumented code runs slower than `raco test`'s, so a file's bound under
;; cover is its `raco test` bound times this.
(define cover-slowdown 3)

(define (die fmt . args)
  (apply eprintf fmt args)
  (exit 1))

(define (racket)
  (or (find-executable-path "racket") (die "coverage: no racket on PATH\n")))

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

(struct overran (file seconds) #:transparent)

;; 0 and 1 are cover's own answers; anything else ended it before it wrote a
;; report, and a report left from an earlier run must not stand in for one.
(define (cover-outcome result)
  (match result
    [0 'passed]
    [1 'failed]
    [142 "raco cover was ended by SIGALRM: a test's process deadline fired (its notice is above)"]
    [(overran file seconds)
     (format "~a ran past its ~a s bound under raco cover, so the run and its children were ended"
             file seconds)]
    [code
     #:when (> code 128)
     (format "raco cover was ended by signal ~a" (- code 128))]
    [code (format "raco cover exited with status ~a" code)]))

;; cover runs every test file in one process with no per-file timeout, and
;; logs each file's start on a block-buffered stdout: line buffering makes
;; the start lines arrive when the files do, so a file past its bound is seen.
(define (cover-command)
  (list (racket) "-l" "racket/base"
        "-e" "(file-stream-buffer-mode (current-output-port) 'line)"
        "-l-" "raco" "cover" "-b" "-d" (report-dir) "torch/"))

(define (test-timeouts)
  (match (get-info/full (build-path "torch"))
    [#f '()]
    [info (info 'test-timeouts (lambda () '()))]))

(define ((bound-for timeouts) file)
  (* cover-slowdown
     (or (for/first ([entry (in-list timeouts)]
                     #:when (string-suffix? file (string-append "/torch/" (first entry))))
           (second entry))
         (file-seconds))))

(define running-px #px"^cover: running file: (.*)$")

(define (echo-lines in out running)
  (for ([line (in-lines in)])
    (displayln line out)
    (flush-output out)
    (match (regexp-match running-px line)
      [(list _ file) (set-box! running (cons file (current-inexact-monotonic-milliseconds)))]
      [#f (void)])))

(define (watch sp running bound)
  (let poll ()
    (match* ((sync/timeout 1 sp) (unbox running))
      [(#f (cons file started))
       #:when (> (- (current-inexact-monotonic-milliseconds) started) (* 1000 (bound file)))
       (subprocess-kill sp #t)
       (sync/timeout 10 sp)
       (overran file (bound file))]
      [(#f _) (poll)]
      [(_ _) (subprocess-status sp)])))

;; The command runs in a process group of its own, so the children a test
;; started die with it when its file overruns. Its stderr shares the stdout
;; pipe, so a test's stderr lands after the start line of the file it ran in.
(define (run-watched command #:bound bound)
  (define-values (sp out in _err) (apply subprocess #f #f 'stdout 'new command))
  (close-output-port in)
  (define running (box #f))
  (define stdout (current-output-port))
  (define pump (thread (lambda () (echo-lines out stdout running))))
  (begin0 (watch sp running bound)
          (sync/timeout 10 pump)
          (close-input-port out)))

;; Racket also exits 1 on a signal or an uncaught error, so an earlier
;; run's report goes first: a cover that wrote nothing then has nothing to read.
(define (index-path)
  (build-path (report-dir) "index.html"))

(define (run-cover!)
  (when (file-exists? (index-path)) (delete-file (index-path)))
  (displayln "running raco cover over torch/ (instrumented; runs the suite)")
  (define outcome
    (cover-outcome (run-watched (cover-command) #:bound (bound-for (test-timeouts)))))
  (when (string? outcome)
    (die "coverage: ~a\n" outcome))
  (eq? outcome 'passed))

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

;; How many rows the report claims, independent of whether each one parsed.
;; A markup change that breaks only some rows would otherwise drop those
;; files from the totals with nothing to show for it.
(define (count-rows html)
  (length (regexp-match* #px"<tr\\s+class=\"file-info\"" html)))

(define (read-entries)
  (define index (index-path))
  (unless (file-exists? index)
    (die "coverage: no report at ~a\n" index))
  (define html (file->string index))
  (define entries (parse-entries html))
  (define declared (count-rows html))
  (when (null? entries)
    (die "coverage: could not parse ~a (did cover's html change?)\n" index))
  (unless (= declared (length entries))
    (die "coverage: parsed ~a of the ~a rows in ~a; cover's html changed\n"
         (length entries) declared index))
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
  ;; `diff HEAD` covers the index as well as the working tree, but git diff
  ;; only knows about tracked content: a new file nobody has added yet is
  ;; precisely the uncovered code this flag exists to find.
  (define touched
    (for/list ([f (in-list (append (git "diff" "--name-only" "origin/master...HEAD")
                                   (git "diff" "--name-only" "HEAD")
                                   (git "ls-files" "--others" "--exclude-standard")))]
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
          (displayln "      uncovered lines: unknown, cover wrote no report")]
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
   [("--file-seconds") n "a test file's raco test bound, scaled for cover (default 300)"
                       (define seconds (string->number n))
                       (unless (exact-positive-integer? seconds)
                         (die "coverage: --file-seconds needs a positive integer, got ~a\n" n))
                       (file-seconds seconds)]
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
  (require (only-in racket/file delete-directory/files make-temporary-directory
                    make-temporary-file)
           (only-in racket/port open-output-nowhere)
           (only-in rackunit check-equal? check-false check-pred check-regexp-match check-true
                    test-case))

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

  (test-case "a row that stops matching is visible against the declared count"
    (define good (row "torch/nn/a" "100" "5" "0" "5"))
    ;; as if a later cover renamed one cell's class
    (define drifted
      (regexp-replace "covered-expressions" (row "torch/nn/b" "50" "1" "1" "2")
                      "covered-exprs"))
    (define html (string-append good drifted))
    (check-equal? (length (parse-entries html)) 1)
    (check-equal? (count-rows html) 2
                  "the discrepancy is what read-entries refuses to average over"))

  (test-case "a cover run ended by a signal is an error, not a report"
    (check-equal? (map cover-outcome '(0 1)) '(passed failed))
    (check-regexp-match #rx"SIGALRM: a test's process deadline fired" (cover-outcome 142))
    (check-regexp-match #rx"ended by signal 9" (cover-outcome 137))
    (check-regexp-match #rx"exited with status 2" (cover-outcome 2)))

  (define this-script (variable-reference->module-source (#%variable-reference)))

  ;; a stub `racket` that exits 1 having written nothing, as a killed cover can
  (define (gate-over-stale-report)
    (define dir (make-temporary-directory "coverage-test~a"))
    (define bin (build-path dir "bin"))
    (define report (build-path dir "report"))
    (make-directory bin)
    (make-directory report)
    (with-output-to-file (build-path bin "racket")
      (lambda () (display "#!/bin/sh\nexit 1\n")))
    (file-or-directory-permissions (build-path bin "racket") #o755)
    (with-output-to-file (build-path report "index.html")
      (lambda () (display (string-append "<tbody>" (row "torch/main" "100" "1" "0" "1")
                                         "</tbody>"))))
    (define racket (find-executable-path (find-system-path 'exec-file)))
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set!
     env #"PATH"
     (bytes-append (path->bytes bin) #":" (or (environment-variables-ref env #"PATH") #"")))
    (begin0
      (parameterize ([current-environment-variables env]
                     [current-output-port (open-output-nowhere)]
                     [current-error-port (open-output-nowhere)])
        (system*/exit-code racket
                           this-script "-d" (path->string report)))
      (delete-directory/files dir)))

  (test-case "a cover that exits 1 with no report fails the gate, whatever report an earlier run left"
    (check-false (zero? (gate-over-stale-report))))

  (define sh (find-executable-path "sh"))

  (define (watched script #:bound [seconds 1])
    (define start (current-inexact-monotonic-milliseconds))
    (define out (open-output-string))
    (define result
      (parameterize ([current-output-port out])
        (run-watched (list sh "-c" script) #:bound (lambda (_file) seconds))))
    (values result (- (current-inexact-monotonic-milliseconds) start) (get-output-string out)))

  (test-case "a file past its bound ends the run and is named"
    (define-values (result ms _out)
      (watched "echo 'cover: running file: /r/torch/tests/slow-test.rkt'; sleep 60"))
    (check-equal? result (overran "/r/torch/tests/slow-test.rkt" 1))
    (check-true (< ms 15000.0) (format "the run ended after ~a ms" ms))
    (check-regexp-match #rx"^/r/torch/tests/slow-test.rkt ran past its 1 s bound"
                        (cover-outcome result)))

  (test-case "the children a test started die with the run"
    (define pid-file (make-temporary-file "coverage-pid~a"))
    (define-values (result _ms _out)
      (watched (format "echo 'cover: running file: /r/a.rkt'; sleep 60 & echo $! > ~a; wait"
                       pid-file)))
    (define pid (call-with-input-file pid-file read))
    (delete-file pid-file)
    (check-pred overran? result)
    (check-false (for/and ([_ (in-range 50)])
                   (sleep 0.1)
                   (directory-exists? (format "/proc/~a" pid)))
                 (format "child ~a outlived the run" pid)))

  (test-case "a run inside its bound returns cover's status, its stderr in order"
    (define-values (failed _ms1 out)
      (watched (string-append "echo 'cover: running file: /r/a.rkt'; echo notice >&2; "
                              "echo 'cover: running file: /r/b.rkt'; exit 1")
               #:bound 30))
    (check-equal? failed 1)
    (check-equal? out "cover: running file: /r/a.rkt\nnotice\ncover: running file: /r/b.rkt\n")
    (define-values (quiet _ms2 _out2) (watched "sleep 2; exit 0"))
    (check-equal? quiet 0 "no start line, no bound"))

  (test-case "a file's bound is its test-timeouts entry, else the default, times the slowdown"
    (define bound (bound-for '(("tests/loader-workers-test.rkt" 600))))
    (check-equal? (bound "/w/torch/tests/loader-workers-test.rkt") (* cover-slowdown 600))
    (check-equal? (bound "/w/torch/tests/race-stress-test.rkt") (* cover-slowdown 300))
    (check-equal? (parameterize ([file-seconds 10]) (bound "/w/torch/tests/a.rkt"))
                  (* cover-slowdown 10)))

  (test-case "a file's area is its directory, with raw/ kept apart"
    (check-equal? (map area '("torch/main.rkt"
                              "torch/nn/linear.rkt"
                              "torch/foreign/ops.rkt"
                              "torch/foreign/raw/memory.rkt"))
                  '("torch (facades)" "torch/nn" "torch/foreign"
                    "torch/foreign/raw"))))
