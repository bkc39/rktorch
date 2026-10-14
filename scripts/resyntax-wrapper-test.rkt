#lang racket/base

(module+ test
  (require (only-in racket/file
                    delete-directory/files display-lines-to-file
                    display-to-file file->lines make-temporary-directory)
           (only-in racket/match == match match-define)
           ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in racket/string string-trim)
           (only-in racket/system system*/exit-code)
           (only-in rackunit check-false check-match test-case))

  (define-runtime-path wrapper "resyntax")

  (define ci-finding #px"resyntax: .*\\.rkt:[0-9]+:[0-9]+ \\[")

  (struct fixture (env repo stubs shas))
  (struct result (code err base gh) #:transparent)

  (define (run env dir program . args)
    (define out (open-output-string))
    (define err (open-output-string))
    (define code (box 'timeout))
    (define custodian (make-custodian))
    (define worker
      (parameterize ([current-custodian custodian]
                     [current-subprocess-custodian-mode 'kill]
                     [current-environment-variables env]
                     [current-directory dir]
                     [current-input-port (open-input-string "")]
                     [current-output-port out]
                     [current-error-port err])
        (thread
         (lambda ()
           (set-box! code (apply system*/exit-code
                                 (find-executable-path program)
                                 args))))))
    (sync/timeout 60 worker)
    (custodian-shutdown-all custodian)
    (list (unbox code) (get-output-string out) (get-output-string err)))

  (define (git env repo . args)
    (match (apply run env repo "git" args)
      [(list 0 out _) (string-trim out)]
      [(list code _ err) (error 'git "~a exited ~a: ~a" args code err)]))

  (define (stub-lines name)
    (list "#!/bin/sh"
          (format "printf '%s\\n' \"$@\" > \"$STUB_DIR/~a.args\"" name)
          (format "cat \"$STUB_DIR/~a-$2.out\" 2>/dev/null" name)
          (format "exit \"$(cat \"$STUB_DIR/~a-$2.rc\" 2>/dev/null || echo 0)\""
                  name)))

  (define (environment stubs)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([(name value)
           (in-hash (hash "PATH" (format "~a:~a" stubs (getenv "PATH"))
                          "STUB_DIR" (path->string stubs)
                          "GIT_CONFIG_GLOBAL" "/dev/null"
                          "GIT_CONFIG_NOSYSTEM" "1"
                          "GIT_AUTHOR_NAME" "test"
                          "GIT_AUTHOR_EMAIL" "test@example.com"
                          "GIT_COMMITTER_NAME" "test"
                          "GIT_COMMITTER_EMAIL" "test@example.com"))])
      (environment-variables-set! env
                                  (string->bytes/utf-8 name)
                                  (string->bytes/utf-8 value)))
    env)

  (define (make-fixture root)
    (define repo (build-path root "repo"))
    (define stubs (build-path root "stubs"))
    (make-directory repo)
    (make-directory stubs)
    (for ([name (in-list '("gh" "racket" "resyntax"))])
      (define path (build-path stubs name))
      (display-lines-to-file (stub-lines name) path)
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

  (define (lines-of path)
    (and (file-exists? path) (file->lines path)))

  (define (lint fx
                #:view [view ""]
                #:view-rc [view-rc 0]
                #:list [listing ""]
                #:detached? [detached? #f]
                . args)
    (match-define (fixture env repo stubs _) fx)
    (git env repo "checkout" "-q" "feature")
    (when detached?
      (git env repo "checkout" "-q" "--detach"))
    (for ([name (in-list '("gh.args" "resyntax.args"))])
      (define path (build-path stubs name))
      (when (file-exists? path)
        (delete-file path)))
    (for ([name (in-list '("gh-view.out" "gh-view.rc" "gh-list.out"))]
          [content (in-list (list view (number->string view-rc) listing))])
      (display-to-file content (build-path stubs name) #:exists 'replace))
    (match-define (list code _ err)
      (apply run env repo "bash" (path->string wrapper) args))
    (define base
      (match (lines-of (build-path stubs "resyntax.args"))
        [(list _ ... "--local-git-repository" _ base _ ...) base]
        [_ #f]))
    (result code err base (lines-of (build-path stubs "gh.args"))))

  (define root (make-temporary-directory))
  (define fx (make-fixture root))
  (match-define (fixture _ _ _ (hash-table ('A a) ('B b) ('F f) ('O o))) fx)

  (test-case "an open pull request's base is the default ref"
    (check-match (lint fx #:view "OPEN stack-base"
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== b) (list "pr" "view" _ ...))))

  (test-case "a base origin does not have falls back to origin/master"
    (check-match (lint fx #:view "OPEN pruned-base"
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== a) _)))

  (test-case "a closed pull request's base is ignored"
    (check-match (lint fx #:view "MERGED stack-base"
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== a) _)))

  (test-case "junk from gh with exit 0 falls back to origin/master"
    (check-match (lint fx #:view "OPEN ::not a ref::"
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== a) _)))

  (test-case "output from a failing gh is not trusted"
    (check-match (lint fx #:view "OPEN stack-base" #:view-rc 1
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== a) _)))

  (test-case "a detached HEAD finds its pull request by head commit"
    (check-match (lint fx #:view-rc 1
                       #:list (format "~a master\n~a stack-base\n" o f)
                       #:detached? #t
                       "analyze" "--local-git-repository" ".")
                 (result 0 _ (== b) (list "pr" "list" _ ...))))

  (test-case "an explicit ancestor SHA resolves to itself without gh"
    (check-match (lint fx "analyze" "--local-git-repository" "." b
                       "--analyzer-timeout" "30000")
                 (result 0 _ (== b) #f)))

  (test-case "a ref with no merge base exits 2 with a message CI cannot mistake"
    (define orphan (lint fx "analyze" "--local-git-repository" "." o))
    (check-match
     orphan
     (result 2
             (== (format "resyntax: no merge base of HEAD and ~a in . ~a\n"
                         o
                         "(shallow clone or unrelated history?)"))
             #f
             #f))
    (check-false (regexp-match? ci-finding (result-err orphan))))

  (test-case "--local-git-repository without DIR exits 2"
    (for ([args (in-list '(("analyze" "--local-git-repository")
                           ("analyze" "--local-git-repository"
                                      "--analyzer-timeout" "30000")))])
      (check-match (apply lint fx args)
                   (result 2
                           "resyntax: --local-git-repository needs DIR [REF]\n"
                           #f
                           #f))))

  (delete-directory/files root))
