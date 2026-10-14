#lang racket/base

(require (for-syntax racket/base (only-in syntax/parse/pre ...+ expr id))
         (only-in racket/dict in-dict)
         (only-in racket/file
                  delete-directory/files display-lines-to-file display-to-file
                  file->lines make-temporary-directory)
         (only-in racket/match == match match-define)
         (only-in racket/runtime-path define-runtime-path)
         (only-in racket/string string-trim)
         (only-in racket/system process*/ports)
         (only-in rackunit
                  check-equal? check-exn check-false check-match check-true
                  test-case)
         (only-in syntax/parse/define define-syntax-parse-rule))

(define-runtime-path wrapper "resyntax")

(define ci-finding #px"resyntax: .*\\.rkt:[0-9]+:[0-9]+ \\[")

(define local-args '("analyze" "--local-git-repository" "."))

(define stub
  '("#!/bin/sh"
    "name=${0##*/}"
    "printf '%s\\n' \"$@\" > \"$STUB_DIR/$name.args\""
    "cat \"$STUB_DIR/$name-$2.out\" 2>/dev/null"
    "exit \"$(cat \"$STUB_DIR/$name-$2.rc\" 2>/dev/null || echo 0)\""))

(define inherited-git
  '(#"GIT_ALTERNATE_OBJECT_DIRECTORIES" #"GIT_COMMON_DIR"
    #"GIT_CONFIG_COUNT" #"GIT_CONFIG_PARAMETERS" #"GIT_DIR"
    #"GIT_INDEX_FILE" #"GIT_NAMESPACE" #"GIT_OBJECT_DIRECTORY"
    #"GIT_PREFIX" #"GIT_WORK_TREE"))

(struct fixture (env repo stubs shas))
(struct outcome (code out err) #:transparent)
(struct linted (code err base gh) #:transparent)

(define-syntax-parse-rule (with-temporary-directory (dir:id) body:expr ...+)
  (let ([dir (make-temporary-directory)])
    (dynamic-wind void
                  (lambda () body ...)
                  (lambda () (delete-directory/files dir #:must-exist? #f)))))

(define (run env dir program #:timeout [seconds 60] . args)
  (define out (open-output-string))
  (define err (open-output-string))
  (match-define (list _ _ _ _ control)
    (parameterize ([current-environment-variables env]
                   [current-directory dir]
                   [subprocess-group-enabled #t])
      (apply process*/ports out (open-input-string "") err
             (find-executable-path program) args)))
  (define done? (sync/timeout seconds (thread (lambda () (control 'wait)))))
  (control 'kill)
  (control 'wait)
  (outcome (if done? (control 'exit-code) 'timeout)
           (get-output-string out)
           (get-output-string err)))

(define (git env repo . args)
  (match (apply run env repo "git" args)
    [(outcome 0 out _) (string-trim out)]
    [(outcome code _ err) (error 'git "~a exited ~a: ~a" args code err)]))

(define (environment stubs)
  (define env (environment-variables-copy (current-environment-variables)))
  (define path (environment-variables-ref env #"PATH"))
  (for ([(name value)
         (in-dict `(,@(map (lambda (name) (cons name #f)) inherited-git)
                    (#"PATH" . ,(bytes-append (path->bytes stubs) #":" path))
                    (#"STUB_DIR" . ,(path->bytes stubs))
                    (#"GIT_CONFIG_GLOBAL" . #"/dev/null")
                    (#"GIT_CONFIG_NOSYSTEM" . #"1")
                    (#"GIT_AUTHOR_NAME" . #"test")
                    (#"GIT_AUTHOR_EMAIL" . #"test@example.com")
                    (#"GIT_COMMITTER_NAME" . #"test")
                    (#"GIT_COMMITTER_EMAIL" . #"test@example.com")))])
    (environment-variables-set! env name value))
  env)

(define (make-fixture root)
  (define repo (build-path root "repo"))
  (define stubs (build-path root "stubs"))
  (make-directory repo)
  (make-directory stubs)
  (for ([name (in-list '("gh" "racket" "resyntax"))])
    (define path (build-path stubs name))
    (display-lines-to-file stub path)
    (file-or-directory-permissions path #o755))
  (define env (environment stubs))
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
  (fixture env repo stubs (hash 'A a 'B b 'F f 'O o)))

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
              #:args [args local-args])
  (match-define (fixture env repo stubs _) fx)
  (apply git env repo "checkout" "-q" checkout)
  (for ([(name content)
         (in-dict `(("gh.args" . #f)
                    ("resyntax.args" . #f)
                    ("gh-view.out" . ,view)
                    ("gh-view.rc" . ,(number->string view-rc))
                    ("gh-list.out" . ,listing)))])
    (stage! (build-path stubs name) content))
  (match-define (outcome code _ err)
    (apply run env repo "bash" (path->string wrapper) args))
  (linted code
          err
          (match (lines-of (build-path stubs "resyntax.args"))
            [(list _ ... "--local-git-repository" _ base _ ...) base]
            [_ #f])
          (lines-of (build-path stubs "gh.args"))))

(test-case "a failing fixture leaves no temporary directory behind"
  (define root #f)
  (check-exn exn:fail:filesystem?
             (lambda ()
               (with-temporary-directory (dir)
                 (set! root dir)
                 (make-directory (build-path dir "repo"))
                 (make-fixture dir))))
  (check-false (directory-exists? root)))

(with-temporary-directory (root)
  (define fx (make-fixture root))
  (match-define (fixture env _ _ (hash-table ('A a) ('B b) ('F f) ('O o))) fx)

  (for ([row (in-list
              (list
               (list "an open pull request's base is the default ref"
                     "OPEN stack-base" 0 '() b)
               (list "a flag after DIR is not taken for REF"
                     "OPEN stack-base" 0 '("--analyzer-timeout" "30000") b)
               (list "a base origin does not have falls back to origin/master"
                     "OPEN pruned-base" 0 '() a)
               (list "a closed pull request's base is ignored"
                     "MERGED stack-base" 0 '() a)
               (list "junk from gh with exit 0 falls back to origin/master"
                     "OPEN ::not a ref::" 0 '() a)
               (list "output from a failing gh is not trusted"
                     "OPEN stack-base" 1 '() a)))])
    (match-define (list name view rc flags base) row)
    (test-case name
      (check-match (lint fx
                         #:view view
                         #:view-rc rc
                         #:args (append local-args flags))
                   (linted 0 _ (== base) (list "pr" "view" _ ...)))))

  (test-case "a detached HEAD finds its pull request by head commit"
    (check-match (lint fx
                       #:view-rc 1
                       #:list (format "~a master\n~a stack-base\n" o f)
                       #:checkout '("--detach" "feature"))
                 (linted 0
                         (regexp #rx"linting changes since origin/stack-base")
                         (== b)
                         (list "pr" "list" _ ...))))

  (test-case "an explicit ancestor SHA resolves to itself without gh"
    (check-match (lint fx
                       #:args (append local-args
                                      (list b "--analyzer-timeout" "30000")))
                 (linted 0 _ (== b) #f)))

  (test-case "a ref with no merge base exits 2 with a message CI cannot mistake"
    (define orphan (lint fx #:args (append local-args (list o))))
    (check-match orphan
                 (linted 2
                         (== (string-append
                              "resyntax: no merge base of HEAD and " o
                              " in . (shallow clone or unrelated history?)\n"))
                         #f
                         #f))
    (check-false (regexp-match? ci-finding (linted-err orphan))))

  (test-case "--local-git-repository without DIR exits 2"
    (for ([args (in-list '(("analyze" "--local-git-repository")
                           ("analyze" "--local-git-repository"
                                      "--analyzer-timeout" "30000")))])
      (check-match (lint fx #:args args)
                   (linted 2
                           "resyntax: --local-git-repository needs DIR [REF]\n"
                           #f
                           #f))))

  (test-case "an inherited GIT_DIR leaves the caller's repository alone"
    (define caller (build-path root "caller"))
    (make-directory caller)
    (git env caller "init" "-q" "-b" "main")
    (git env caller "commit" "-q" "--allow-empty" "-m" "C")
    (define (state)
      (map (lambda (args) (apply git env caller args))
           '(("symbolic-ref" "HEAD") ("rev-parse" "HEAD") ("for-each-ref"))))
    (define before (state))
    (define inherited
      (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! inherited
                                #"GIT_DIR"
                                (path->bytes (build-path caller ".git")))
    (parameterize ([current-environment-variables inherited])
      (with-temporary-directory (nested)
        (lint (make-fixture nested) #:view "OPEN stack-base")))
    (check-equal? (state) before))

  (test-case "a timed-out run kills the whole process group"
    (define start (current-inexact-milliseconds))
    (check-match (run env root "sh" "-c" "sleep 60; :" #:timeout 1)
                 (outcome 'timeout _ _))
    (check-true (< (- (current-inexact-milliseconds) start) 20000))))
