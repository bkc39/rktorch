#lang racket/base

(require (for-syntax (only-in racket/base ~?)
                     (only-in syntax/parse/pre ...+ ~optional ~seq expr id))
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
         (only-in syntax/parse/define define-syntax-parse-rule))

(define-runtime-path wrapper "resyntax")
(define-runtime-path self "resyntax-wrapper-test.rkt")

(define local-args '("analyze" "--local-git-repository" "."))

(define stub
  '("#!/bin/sh"
    "name=${0##*/}"
    "printf '%s\\n' \"$@\" > \"$STUB_DIR/$name.args\""
    "cat \"$STUB_DIR/$name-$2.out\" 2>/dev/null"
    "exit \"$(cat \"$STUB_DIR/$name-$2.rc\" 2>/dev/null || echo 0)\""))

(struct fixture (root env repo stubs top shas))
(struct outcome (code out err) #:transparent)
(struct linted (code err args gh) #:transparent)

(define-syntax-parse-rule (with-cleanup cleanup:expr body:expr ...+)
  (let* ([clean (lambda () cleanup)]
         [handle (plumber-add-flush! (current-plumber) (lambda (_) (clean)))])
    (dynamic-wind void
                  (lambda () body ...)
                  (lambda ()
                    (plumber-flush-handle-remove! handle)
                    (clean)))))

(define-syntax-parse-rule (with-temporary-directory
                           (dir:id (~optional (~seq #:in base:expr)))
                           body:expr ...+)
  (let ([dir (make-temporary-directory #:base-dir (~? base #f))])
    (with-cleanup (delete-directory/files dir #:must-exist? #f)
      body ...)))

(define (run env dir program #:timeout [seconds 60] . args)
  (define out (open-output-string))
  (define err (open-output-string))
  (match-define (list _ _ _ _ control)
    (parameterize ([current-environment-variables env]
                   [current-directory dir]
                   [subprocess-group-enabled #t])
      (apply process*/ports out (open-input-string "") err
             (find-executable-path program) args)))
  (define done?
    (with-cleanup (control 'kill)
      (sync/timeout seconds (thread (lambda () (control 'wait))))))
  (control 'wait)
  (outcome (if done? (control 'exit-code) 'timeout)
           (get-output-string out)
           (get-output-string err)))

(define (git env repo . args)
  (match (apply run env repo "git" args)
    [(outcome 0 out _) (string-trim out)]
    [(outcome code _ err) (error 'git "~a exited ~a: ~a" args code err)]))

(define (environment root #:extra [extra '()])
  (define stubs (build-path root "stubs"))
  (define (inherited name)
    (environment-variables-ref (current-environment-variables) name))
  (define vars
    (for/hash ([(name value)
                (in-dict
                 `((#"PATH" . ,(bytes-append (path->bytes stubs) #":"
                                             (inherited #"PATH")))
                   (#"TMPDIR" . ,(inherited #"TMPDIR"))
                   (#"HOME" . ,(path->bytes root))
                   (#"STUB_DIR" . ,(path->bytes stubs))
                   (#"GIT_CONFIG_GLOBAL" . #"/dev/null")
                   (#"GIT_CONFIG_NOSYSTEM" . #"1")
                   (#"GIT_AUTHOR_NAME" . #"test")
                   (#"GIT_AUTHOR_EMAIL" . #"test@example.com")
                   (#"GIT_COMMITTER_NAME" . #"test")
                   (#"GIT_COMMITTER_EMAIL" . #"test@example.com")
                   ,@extra))]
               #:when value)
      (values name value)))
  (apply make-environment-variables
         (for*/list ([(name value) (in-hash vars)]
                     [part (in-list (list name value))])
           part)))

(define (script! path lines)
  (display-lines-to-file lines path)
  (file-or-directory-permissions path #o755))

(define (make-fixture root)
  (define repo (build-path root "repo"))
  (define stubs (build-path root "stubs"))
  (make-directory repo)
  (make-directory stubs)
  (for ([name (in-list '("gh" "racket" "resyntax"))])
    (script! (build-path stubs name) stub))
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
  (delete-directory/files path #:must-exist? #f)
  (when content
    (display-to-file content path)))

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
           (environment root #:extra extra)
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

(module+ hang
  (with-temporary-directory (dir)
    (run (current-environment-variables)
         dir
         "sh"
         "-c"
         "echo $$ > \"$1\"; while :; do sleep 1; done"
         "sh"
         (getenv "RESYNTAX_WRAPPER_HANG_PIDFILE")
         #:timeout 600)))

(module+ test
  (require (only-in rackunit
                    check-equal? check-exn check-false check-match check-true
                    test-case))

  (define ci-finding #px"resyntax: .*\\.rkt:[0-9]+:[0-9]+ \\[")
  (define no-args '(#f #f))

  (test-case "a failing fixture leaves no temporary directory behind"
    (with-temporary-directory (parent)
      (check-exn exn:fail:filesystem?
                 (lambda ()
                   (with-temporary-directory (dir #:in parent)
                     (make-directory (build-path dir "repo"))
                     (make-fixture dir))))
      (check-equal? (directory-list parent) '())))

  (with-temporary-directory (root)
    (define fx (make-fixture root))
    (match-define (fixture _ env repo _ _ (hash-table ('F f) ('O o))) fx)

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
                                        (list (hash-ref (fixture-shas fx) 'B)
                                              "--analyzer-timeout"
                                              "30000")))
                   (linted 0
                           _
                           (== (expected fx 'B "--analyzer-timeout" "30000"))
                           #f)))

    (test-case "CDPATH does not leak into the merge base"
      (make-directory (build-path repo "sub"))
      (check-match (lint fx
                         #:view "OPEN stack-base"
                         #:extra `((#"CDPATH" . ,(path->bytes repo)))
                         #:args '("analyze" "--local-git-repository" "sub"))
                   (linted 0 _ (== (expected fx 'B #:dir "sub")) _)))

    (test-case "a missing DIR exits 2 with the wrapper's message"
      (check-equal? (lint fx #:args '("analyze" "--local-git-repository"
                                                "missing"))
                    (linted 2 "resyntax: no directory missing\n" no-args #f)))

    (test-case "a ref with no merge base exits 2 with a message CI cannot mistake"
      (define orphan (lint fx #:args (append local-args (list o))))
      (check-equal? orphan
                    (linted 2
                            (string-append
                             "resyntax: no merge base of HEAD and " o
                             " in . (shallow clone or unrelated history?)\n")
                            no-args
                            #f))
      (check-false (regexp-match? ci-finding (linted-err orphan))))

    (test-case "--local-git-repository without DIR exits 2"
      (for ([args (in-list '(("analyze" "--local-git-repository")
                             ("analyze" "--local-git-repository"
                                        "--analyzer-timeout" "30000")))])
        (check-equal? (lint fx #:args args)
                      (linted 2
                              "resyntax: --local-git-repository needs DIR [REF]\n"
                              no-args
                              #f))))

    (test-case "a hostile inherited environment changes nothing"
      (define caller (build-path root "caller"))
      (define templates (build-path root "templates"))
      (define bash-env (build-path root "bash-env"))
      (make-directory* (build-path templates "hooks"))
      (for ([path (in-list (list (build-path templates "hooks" "pre-commit")
                                 bash-env))])
        (script! path '("#!/bin/sh" "exit 1")))
      (make-directory caller)
      (git env caller "init" "-q" "-b" "main")
      (git env caller "commit" "-q" "--allow-empty" "-m" "C")
      (define (state)
        (map (lambda (args) (apply git env caller args))
             '(("symbolic-ref" "HEAD") ("rev-parse" "HEAD") ("for-each-ref"))))
      (define before (state))
      (parameterize ([current-environment-variables
                      (make-environment-variables
                       #"PATH" (environment-variables-ref
                                (current-environment-variables) #"PATH")
                       #"GIT_DIR" (path->bytes (build-path caller ".git"))
                       #"GIT_TEMPLATE_DIR" (path->bytes templates)
                       #"BASH_ENV" (path->bytes bash-env))])
        (with-temporary-directory (nested #:in root)
          (define nested-fx (make-fixture nested))
          (check-match (lint nested-fx #:view "OPEN stack-base")
                       (linted 0 _ (== (expected nested-fx 'B)) _))))
      (check-equal? (state) before))

    (test-case "a timed-out run kills the whole process group"
      (define start (current-inexact-milliseconds))
      (check-match (run env root "sh" "-c" "sleep 60; :" #:timeout 1)
                   (outcome 'timeout _ _))
      (check-true (< (- (current-inexact-milliseconds) start) 20000)))

    (test-case "a raco test timeout still removes the directory and the group"
      (define tmp (build-path root "hang-tmp"))
      (define pidfile (build-path root "hang-pid"))
      (make-directory tmp)
      (define hung
        (run (environment
              root
              #:extra
              `((#"TMPDIR" . ,(path->bytes tmp))
                (#"RESYNTAX_WRAPPER_HANG_PIDFILE" . ,(path->bytes pidfile))
                (#"PLTUSERHOME"
                 . ,(environment-variables-ref (current-environment-variables)
                                               #"PLTUSERHOME"))))
             root
             "raco"
             "test"
             "--timeout"
             "10"
             "--submodule"
             "hang"
             (path->string self)
             #:timeout 120))
      (define survivor
        (run env
             root
             "sh"
             "-c"
             (string-append
              "pid=$(cat \"$1\") || exit 1; for _ in 1 2 3 4 5; do"
              " kill -0 -- \"-$pid\" 2>/dev/null || exit 0; sleep 1;"
              " done; kill -KILL -- \"-$pid\"; exit 1")
             "sh"
             (path->string pidfile)))
      (check-match (list hung (directory-list tmp) survivor)
                   (list (outcome 2 _ (regexp #rx"timeout after 10 seconds"))
                         '()
                         (outcome 0 _ _))))))
