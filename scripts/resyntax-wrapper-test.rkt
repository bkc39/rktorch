#lang racket/base

(require (for-syntax (only-in syntax/parse/pre ...+ expr id))
         (only-in racket/dict in-dict)
         (only-in racket/file
                  delete-directory/files display-lines-to-file display-to-file
                  file->lines make-directory* make-temporary-directory)
         (only-in racket/match == match match-define)
         ;; whole-module: define-runtime-path needs phase-1 bindings
         ;; only-in strips
         racket/runtime-path
         (only-in racket/string string-trim)
         (only-in racket/system process*/ports)
         (only-in rackunit check-equal? check-false check-match test-case)
         (only-in syntax/parse/define define-syntax-parse-rule))

(define-runtime-path wrapper "resyntax")

(define local-args '("analyze" "--local-git-repository" "."))
(define ci-finding #px"resyntax: .*\\.rkt:[0-9]+:[0-9]+ \\[")

(define stub
  '("#!/bin/sh"
    "name=${0##*/}"
    "printf '%s\\n' \"$@\" > \"$STUB_DIR/$name.args\""
    "cat \"$STUB_DIR/$name-$2.out\" 2>/dev/null"
    "exit \"$(cat \"$STUB_DIR/$name-$2.rc\" 2>/dev/null || echo 0)\""))

(struct fixture (root env repo stubs top shas))
(struct outcome (code out err) #:transparent)
(struct linted (code err args gh) #:transparent)

(define-syntax-parse-rule (with-temporary-directory (dir:id) body:expr ...+)
  (let ([dir (make-temporary-directory)])
    (dynamic-wind void
                  (lambda () body ...)
                  (lambda () (delete-directory/files dir #:must-exist? #f)))))

(define (run env dir program . args)
  (define out (open-output-string))
  (define err (open-output-string))
  (match-define (list _ _ _ _ control)
    (parameterize ([current-environment-variables env]
                   [current-directory dir])
      (apply process*/ports out (open-input-string "") err
             (find-executable-path program) args)))
  (define code
    (cond
      [(sync/timeout 60 (thread (lambda () (control 'wait))))
       (control 'exit-code)]
      [else (control 'kill) (control 'wait) 'timeout]))
  (outcome code (get-output-string out) (get-output-string err)))

(define (git env repo . args)
  (match (apply run env repo "git" args)
    [(outcome 0 out _) (string-trim out)]
    [(outcome code _ err) (error 'git "~a exited ~a: ~a" args code err)]))

(define (environment root . extra)
  (define stubs (build-path root "stubs"))
  (apply make-environment-variables
         #"PATH" (bytes-append (path->bytes stubs) #":"
                               (environment-variables-ref
                                (current-environment-variables) #"PATH"))
         #"HOME" (path->bytes root)
         #"STUB_DIR" (path->bytes stubs)
         #"GIT_CONFIG_GLOBAL" #"/dev/null"
         #"GIT_CONFIG_NOSYSTEM" #"1"
         #"GIT_AUTHOR_NAME" #"test"
         #"GIT_AUTHOR_EMAIL" #"test@example.com"
         #"GIT_COMMITTER_NAME" #"test"
         #"GIT_COMMITTER_EMAIL" #"test@example.com"
         extra))

(define (temporary-fixture root)
  (define repo (build-path root "repo"))
  (define stubs (build-path root "stubs"))
  (make-directory repo)
  (make-directory stubs)
  (for ([name (in-list '("gh" "racket" "resyntax"))])
    (define path (build-path stubs name))
    (display-lines-to-file stub path)
    (file-or-directory-permissions path #o755))
  (define env (environment root))
  (define (commit! message)
    (git env repo "commit" "-q" "--allow-empty" "-m" message)
    (git env repo "rev-parse" "HEAD"))
  (git env repo "init" "-q" "-b" "master")
  (define a (commit! "A"))
  (git env repo "checkout" "-q" "-b" "stack-base")
  (define b (commit! "B"))
  (git env repo "checkout" "-q" "-b" "feature")
  (define f (commit! "F"))
  (git env repo "checkout" "-q" "master")
  (define m (commit! "M"))
  (git env repo "update-ref" "refs/remotes/origin/master" m)
  (git env repo "update-ref" "refs/remotes/origin/stack-base" b)
  (define o (git env repo "commit-tree" "-m" "O" "HEAD^{tree}"))
  (fixture root
           env
           repo
           stubs
           (git env repo "rev-parse" "--show-toplevel")
           (hash 'A a 'B b 'F f 'O o)))

(define (stage! path content)
  (cond
    [content (display-to-file content path #:exists 'truncate/replace)]
    [else (delete-directory/files path #:must-exist? #f)]))

(define (lines-of path)
  (and (file-exists? path) (file->lines path)))

(define (lint fx
              #:view [view ""]
              #:view-rc [view-rc 0]
              #:list [listing ""]
              #:checkout [checkout '("feature")]
              #:extra [extra '()]
              #:args [args local-args])
  (match-define (fixture root env repo stubs _ _) fx)
  (apply git env repo "checkout" "-q" checkout)
  (for ([(name content)
         (in-dict `(("gh.args" . #f)
                    ("racket.args" . #f)
                    ("resyntax.args" . #f)
                    ("gh-view.out" . ,view)
                    ("gh-view.rc" . ,(number->string view-rc))
                    ("gh-list.out" . ,listing)))])
    (stage! (build-path stubs name) content))
  (match-define (outcome code _ err)
    (apply run
           (apply environment root extra)
           repo
           "bash"
           (path->string wrapper)
           args))
  (linted code
          err
          (map (lambda (name) (lines-of (build-path stubs name)))
               '("resyntax.args" "racket.args"))
          (lines-of (build-path stubs "gh.args"))))

(define (expected fx base #:dir [dir "."] . flags)
  (match-define (fixture _ _ _ _ top shas) fx)
  (define tail
    `("--local-git-repository" ,dir ,(hash-ref shas base) ,@flags))
  (list `("analyze"
          "--refactoring-suite"
          ,(format "(file \"~a/scripts/resyntax-rules.rkt\")" top)
          "rktorch-recommendations"
          ,@tail)
        `(,(format "~a/scripts/resyntax-lp2.rkt" top) ,@tail)))

(with-temporary-directory (root)
  (define fx (temporary-fixture root))
  (match-define (fixture _ _ repo _ _ (hash-table ('B b) ('F f) ('O o))) fx)

  (for ([row (in-list
              (list
               (list "an open pull request's base is the default ref"
                     "OPEN stack-base" 0 '() 'B)
               (list "a flag after DIR is not taken for REF"
                     "OPEN stack-base" 0 '("--analyzer-timeout" "30000") 'B)
               (list "a base origin does not have falls back to origin/master"
                     "OPEN pruned-base" 0 '() 'A)
               (list "a closed pull request's base is ignored"
                     "MERGED stack-base" 0 '() 'A)
               (list "junk from gh with exit 0 falls back to origin/master"
                     "OPEN ::not a ref::" 0 '() 'A)
               (list "output from a failing gh is not trusted"
                     "OPEN stack-base" 1 '() 'A)))])
    (match-define (list name view rc flags base) row)
    (test-case name
      (check-match (lint fx
                         #:view view
                         #:view-rc rc
                         #:args (append local-args flags))
                   (linted 0
                           _
                           (== (apply expected fx base flags))
                           (list "pr" "view" _ ...)))))

  (test-case "a detached HEAD finds its pull request by head commit"
    (check-match (lint fx
                       #:view-rc 1
                       #:list (format "~a master\n~a stack-base\n" o f)
                       #:checkout '("--detach" "feature"))
                 (linted 0
                         (regexp #rx"linting changes since origin/stack-base")
                         (== (expected fx 'B))
                         (list "pr" "list" _ ...))))

  (test-case "an explicit ancestor SHA resolves to itself without gh"
    (check-match (lint fx
                       #:args (append local-args
                                      (list b "--analyzer-timeout" "30000")))
                 (linted 0
                         _
                         (== (expected fx 'B "--analyzer-timeout" "30000"))
                         #f)))

  (test-case "CDPATH cannot redirect a relative DIR"
    (make-directory (build-path repo "sub"))
    (make-directory* (build-path root "decoy" "sub"))
    (check-match (lint fx
                       #:view "OPEN stack-base"
                       #:extra (list #"CDPATH"
                                     (path->bytes (build-path root "decoy")))
                       #:args '("analyze" "--local-git-repository" "sub"))
                 (linted 0 _ (== (expected fx 'B #:dir "sub")) _)))

  (for ([row (in-list
              (list
               (list "a missing DIR exits 2 with the wrapper's message"
                     '("missing")
                     "no directory missing")
               (list "a ref with no merge base exits 2"
                     (list "." o)
                     (format "no merge base of HEAD and ~a in . ~a"
                             o "(shallow clone or unrelated history?)"))
               (list "--local-git-repository without DIR exits 2"
                     '()
                     "--local-git-repository needs DIR [REF]")
               (list "a flag is not taken for DIR"
                     '("--analyzer-timeout" "30000")
                     "--local-git-repository needs DIR [REF]")))])
    (match-define (list name args message) row)
    (test-case name
      (define failed
        (lint fx #:args `("analyze" "--local-git-repository" ,@args)))
      (check-equal? failed
                    (linted 2
                            (format "resyntax: ~a\n" message)
                            (list #f #f)
                            #f))
      (check-false (regexp-match? ci-finding (linted-err failed))))))
