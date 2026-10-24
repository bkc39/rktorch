#lang racket/base

(require (only-in racket/dict in-dict)
         (only-in racket/file delete-directory/files make-temporary-directory)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/system system*)
         (only-in rackunit check-equal? check-exn check-true test-case)
         (only-in "check-paths.rkt"
                  check-paths mention stale-mentions tracked-files))

(define-runtime-path scripts ".")

(define doc "check-paths-fixture/doc.txt")
(define note "check-paths-fixture/inner/note.txt")
(define files (list doc note))
(define tracked (list* "check-paths.rkt" "check-paths-test.rkt" files))

(define stale
  (list (mention doc 3 "gone.rkt")
        (mention doc 4 "inner/gone.txt")
        (mention doc 7 "span/gone.py")
        (mention doc 9 "fenced/gone.rkt")
        (mention doc 13 "wrapped/gone.rkt")
        (mention doc 15 "flag/gone.rkt")
        (mention note 2 "test/gone.py")))

(test-case "a path resolves from the root or from its file's directory"
  (check-equal? (stale-mentions scripts tracked files) stale))

(test-case "a file on disk that git does not track does not resolve"
  (check-equal? (stale-mentions scripts (remove "check-paths.rkt" tracked) files)
                (list (mention doc 1 "check-paths.rkt")
                      (mention doc 3 "gone.rkt")
                      (mention doc 4 "inner/gone.txt")
                      (mention doc 7 "span/gone.py")
                      (mention doc 9 "fenced/gone.rkt")
                      (mention doc 13 "wrapped/gone.rkt")
                      (mention doc 15 "../check-paths.rkt")
                      (mention doc 15 "flag/gone.rkt")
                      (mention note 2 "test/gone.py"))))

(test-case "a path resolved through its file's directory was extracted, not missed"
  (check-equal? (stale-mentions scripts (remove "check-paths-test.rkt" tracked) files)
                (list (mention doc 3 "gone.rkt")
                      (mention doc 3 "check-paths-test.rkt")
                      (mention doc 4 "inner/gone.txt")
                      (mention doc 7 "span/gone.py")
                      (mention doc 7 "../check-paths-test.rkt")
                      (mention doc 9 "fenced/gone.rkt")
                      (mention doc 9 "../check-paths-test.rkt")
                      (mention doc 13 "wrapped/gone.rkt")
                      (mention doc 15 "flag/gone.rkt")
                      (mention note 1 "../../check-paths-test.rkt")
                      (mention note 2 "test/gone.py"))))

(test-case "an allowance hides its mention, and one that hides nothing is reported"
  (define-values (left unused)
    (check-paths scripts tracked files
                 (list (list doc "gone.rkt") (list doc "inner/note.txt"))))
  (check-equal? left (cdr stale))
  (check-equal? unused (list (list doc "inner/note.txt"))))

(define scratch (make-temporary-directory))

(define (call-with-environment names+values thunk)
  (define env (environment-variables-copy (current-environment-variables)))
  (for ([(name value) (in-dict names+values)])
    (environment-variables-set! env name value))
  (parameterize ([current-environment-variables env])
    (thunk)))

(define (fresh-directory name)
  (define dir (build-path scratch name))
  (make-directory dir)
  dir)

(test-case "outside a git repository, git ls-files failing is the error"
  (define plain (fresh-directory "plain"))
  (call-with-environment
   (list (cons #"GIT_CEILING_DIRECTORIES" (path->bytes scratch)))
   (lambda ()
     (check-exn #rx"check-paths: git ls-files failed in .*not a git repository"
                (lambda () (tracked-files plain))))))

(test-case "a repository with no tracked file is an error, not an empty pass"
  (define empty (fresh-directory "empty"))
  (check-true (parameterize ([current-directory empty])
                (system* (find-executable-path "git") "init" "-q")))
  (check-exn #rx"check-paths: git ls-files lists no file in"
             (lambda () (tracked-files empty))))

(test-case "git missing from PATH is the error"
  (call-with-environment
   (list (cons #"PATH" (path->bytes (build-path scratch "no-bin"))))
   (lambda ()
     (check-exn #rx"check-paths: git is not on PATH"
                (lambda () (tracked-files scripts))))))

(delete-directory/files scratch)
