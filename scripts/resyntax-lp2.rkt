#lang racket/base

(require (only-in racket/cmdline command-line) ;; noqa
         (only-in racket/file display-to-file file->string find-files)
         (only-in racket/format ~a)
         (only-in racket/list append-map)
         (only-in racket/logging with-intercepted-logging)
         (only-in racket/string string-join string-split)
         (only-in rebellion/base/comparator natural<=>)
         (only-in rebellion/base/range unbounded-range)
         (only-in rebellion/collection/range-set
                  range-set range-set-contains?)
         (only-in resyntax resyntax-analyze)
         (only-in resyntax/grimoire/source string-source)
         (only-in resyntax/grimoire/string-replacement
                  string-replacement-apply string-replacement-union)
         (only-in resyntax/private/git git-diff-modified-lines)
         (only-in resyntax/private/refactoring-result
                  refactoring-result-message refactoring-result-new-code
                  refactoring-result-original-code
                  refactoring-result-original-column
                  refactoring-result-original-line refactoring-result-rule-name
                  refactoring-result-set-results
                  refactoring-result-string-replacement)
         (only-in scribble/reader read-syntax-inside)
         (only-in "resyntax-rules.rkt" rktorch-recommendations))

(provide lp2-shadow)

(define lp2-lang "#lang scribble/lp2")
(define shadow-lang "#lang racket/base")

(define files (make-parameter '()))
(define directories (make-parameter '()))
(define git-base (make-parameter #f))
(define timeout-ms (make-parameter 10000))
(define fix? (make-parameter #f))

(define every-line (range-set (unbounded-range #:comparator natural<=>)))

(define (lp2-path? path)
  (and (regexp-match? #rx"[.]rkt$" (path->string path))
       (file-exists? path)
       (equal? (call-with-input-file path (lambda (in) (read-line in 'any)))
               lp2-lang)))

(define (chunk-reference? stx)
  (and (identifier? stx)
       (regexp-match? #px"^<.+>$" (symbol->string (syntax-e stx)))))

(define (chunk-forms stx)
  (define items (syntax->list stx))
  (cond
    [(not items) '()]
    [(and (pair? items)
          (identifier? (car items))
          (eq? (syntax-e (car items)) 'chunk)
          (pair? (cdr items)))
     (cddr items)]
    [else (append-map chunk-forms items)]))

;; The shadow is the file with its prose blanked and every chunk form left at
;; its own character position, so Resyntax's lines, columns and replacements
;; are the literate file's.
(define (lp2-shadow path)
  (define text (file->string path))
  (define in (open-input-string text))
  (port-count-lines! in)
  (read-line in 'any)
  (define keep (make-vector (string-length text) #f))
  (for* ([form (in-list (chunk-forms (read-syntax-inside path in)))]
         #:unless (chunk-reference? form)
         [i (in-range (sub1 (syntax-position form))
                      (+ (sub1 (syntax-position form)) (syntax-span form)))])
    (vector-set! keep i #t))
  (define blanked
    (build-string (string-length text)
                  (lambda (i)
                    (define c (string-ref text i))
                    (if (or (vector-ref keep i) (char=? c #\newline)) c #\space))))
  (string-append shadow-lang (substring blanked (string-length shadow-lang))))

(define (indent text)
  (string-join (for/list ([line (in-list (string-split text "\n" #:trim? #f))])
                 (string-append "  " line))
               "\n"))

(define (report path result)
  (printf "resyntax: ~a:~a:~a [~a]\n" path
          (refactoring-result-original-line result)
          (refactoring-result-original-column result)
          (refactoring-result-rule-name result))
  (printf "\n\n~a\n" (indent (refactoring-result-message result)))
  (printf "\n\n~a\n\n\n~a\n\n\n"
          (indent (~a (refactoring-result-original-code result)))
          (indent (~a (refactoring-result-new-code result)))))

(define (write-fixes! path results)
  (define replacement
    (for/fold ([acc (refactoring-result-string-replacement (car results))])
              ([result (in-list (cdr results))])
      (string-replacement-union acc (refactoring-result-string-replacement result))))
  (display-to-file (string-replacement-apply replacement (file->string path))
                   path #:exists 'truncate/replace))

(define (touches-code? shadow lines)
  (for/or ([text (in-list (cdr (string-split shadow "\n" #:trim? #f)))]
           [line (in-naturals 2)])
    (and (regexp-match? #px"\\S" text) (range-set-contains? lines line))))

(define (lint! path lines)
  (define shadow (lp2-shadow path))
  (define results
    (cond
      [(touches-code? shadow lines)
       (eprintf "resyntax: analyzing ~a through its racket/base shadow\n" path)
       (refactoring-result-set-results
        (resyntax-analyze (string-source shadow)
                          #:suite rktorch-recommendations
                          #:lines lines
                          #:timeout-ms (timeout-ms)))]
      [else '()]))
  (for ([result (in-list results)])
    (report path result))
  (when (and (fix?) (pair? results))
    (write-fixes! path results)
    (printf "resyntax: fixed ~a in ~a\n" (length results) path)))

(define (git-targets)
  (define base (git-base))
  (cond
    [(not base) '()]
    [else
     (define changed
       (parameterize ([current-directory (car base)])
         (git-diff-modified-lines (cdr base))))
     (for*/list ([(path lines) (in-hash changed)]
                 [full (in-value (build-path (car base) path))]
                 #:when (lp2-path? full))
       (cons full lines))]))

(define (targets)
  (append
   (for/list ([file (in-list (reverse (files)))]
              #:when (lp2-path? (string->path file)))
     (cons (string->path file) every-line))
   (for*/list ([directory (in-list (reverse (directories)))]
               [file (in-list (find-files lp2-path? directory))])
     (cons file every-line))
   (git-targets)))

(define (run)
  (define failed? (box #f))
  (with-intercepted-logging
    (lambda (event)
      (eprintf "~a\n" (vector-ref event 1))
      (when (eq? (vector-ref event 0) 'error)
        (set-box! failed? #t)))
    (lambda ()
      (for ([target (in-list (targets))])
        (lint! (path->string (car target)) (cdr target))))
    'warning 'resyntax)
  (exit (if (unbox failed?) 1 0)))

(module+ test
  (require ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in racket/string string-prefix?)
           (only-in rackunit check-equal? check-true test-case))

  (define-runtime-path asr "../examples/racket/07-asr.rkt")
  (define text (file->string asr))
  (define shadow (lp2-shadow asr))
  (define text-lines (string-split text "\n" #:trim? #f))
  (define shadow-lines (string-split shadow "\n" #:trim? #f))

  (define (shadow-line-of prefix)
    (for/first ([original (in-list text-lines)]
                [blanked (in-list shadow-lines)]
                #:when (string-prefix? original prefix))
      (list original blanked)))

  (define (blank? line)
    (regexp-match? #px"^ *$" line))

  (test-case "the shadow keeps every code character at its position"
    (check-equal? (string-length shadow) (string-length text))
    (check-equal? (car shadow-lines) "#lang racket/base ")
    (check-true (for/and ([kept (in-string shadow (string-length lp2-lang))]
                          [original (in-string text (string-length lp2-lang))])
                  (or (char=? kept original) (char=? kept #\space)))))

  (test-case "prose, chunk headers and chunk references are blank"
    (check-true (blank? (cadr (shadow-line-of "@bold{The model.}"))))
    (check-true (blank? (cadr (shadow-line-of "@chunk[<r07-model>"))))
    (check-true (blank? (cadr (shadow-line-of "<r07-model>")))))

  (test-case "code lines are kept verbatim"
    (define model (shadow-line-of "(define-layer asr"))
    (check-equal? (cadr model) (car model))))

(module+ main
  (command-line
   #:program "resyntax-lp2"
   #:multi
   [("--file") file "lint a #lang scribble/lp2 file"
               (files (cons file (files)))]
   [("--directory") directory "lint the #lang scribble/lp2 files under it"
                    (directories (cons directory (directories)))]
   #:once-each
   [("--local-git-repository") directory ref
                               "lint the lines changed since ref"
                               (git-base (cons directory ref))]
   [("--analyzer-timeout") ms "milliseconds each analyzer may take"
                           (timeout-ms (string->number ms))]
   [("--fix") "apply the suggestions to the literate files"
              (fix? #t)]
   #:args ()
   (run)))
