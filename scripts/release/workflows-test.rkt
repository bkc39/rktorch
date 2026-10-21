#lang racket/base
(require (only-in racket/file file->lines file->string)
         (only-in racket/list append-map remove-duplicates)
         (only-in racket/string string-prefix?)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in rackunit check-equal? check-false check-true test-case)
         (only-in "checks.rkt" read-release-data release-data-inputs)
         (only-in "fixture.rkt" refused)
         (only-in "workflows.rkt" called-workflows job job-writes? unguarded-jobs unpinned-actions
                  workflow-jobs))

(define-runtime-path root "../..")
(define-runtime-path fixtures "fixtures/workflows")

(define (fixture-jobs name)
  (workflow-jobs (file->lines (build-path fixtures name))))

(define (repository-workflows)
  (define dir (build-path root ".github" "workflows"))
  (for/list ([file (in-list (directory-list dir))]
             #:when (regexp-match? #px"\\.ya?ml$" (path->string file)))
    (build-path dir file)))

(test-case "the jobs block reads needs, if, uses, permissions and steps' actions"
  (check-equal?
   (fixture-jobs "guarded.yml")
   (list (job "plan" '() #f #f #f '("actions/checkout@v5"))
         (job "build" '("plan") "needs.plan.outputs.build == 'true'" "./.github/workflows/build.yml"
              #f '())
         (job "carried" '("plan") "needs.plan.outputs.build == 'false'" #f #f '())
         (job "check" '("plan") "github.event_name == 'push'" #f #f '())
         (job "assemble" '("plan" "build" "carried")
              (string-append "!cancelled() && (needs.build.result == 'success'"
                             " || needs.carried.result == 'success')")
              #f #f '())
         (job "draft" '("plan" "assemble")
              "${{ always() && needs.assemble.result == 'success' }}" #f #t
              '("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
                "./.github/actions/draft"))
         (job "publish" '("plan") "success() && github.ref_type == 'tag'" #f #f '())
         (job "skipped-with-build" '("build") #f #f #f '())
         (job "report" '("assemble" "draft") "failure()" "./.github/workflows/report.yaml" #f '()))))

(test-case "a job that needs a skippable job states its status function"
  (check-equal? (unguarded-jobs (fixture-jobs "guarded.yml")) '())
  (check-equal? (unguarded-jobs (fixture-jobs "unguarded.yml")) '("assemble" "draft" "publish")
                (string-append "a skip spreads through a job without an if, a step's text is not the"
                               " job's, and an explicit success() is the implicit one")))

(test-case "a job without an if that needs jobs skipped on different conditions is flagged"
  (check-equal? (unguarded-jobs (fixture-jobs "no-if-assemble.yml")) '("assemble")))

(test-case "a jobs block the reader does not understand fails rather than passing empty"
  (refused #rx"expected a job key indented by two spaces under jobs:, got \"    plan:\""
           (λ () (fixture-jobs "four-spaces.yml")))
  (refused #rx"no top-level jobs: key" (λ () (workflow-jobs '("name: none" "on: push"))))
  (refused #rx"expected a job's key: indented by four spaces, got \"    - plan\""
           (λ () (fixture-jobs "indentless-needs.yml")))
  (refused #rx"expected needs: \\[\\.\\.\\.\\] closed on its line, got \"\\[plan,\""
           (λ () (fixture-jobs "unclosed-needs.yml"))))

(define (writing jobs)
  (filter job-writes? jobs))

(test-case "unpinned-actions lists every step's action not pinned by commit"
  (check-equal? (unpinned-actions (writing (fixture-jobs "guarded.yml"))) '())
  (check-equal? (unpinned-actions (fixture-jobs "guarded.yml")) '(("plan" . "actions/checkout@v5")))
  (check-equal? (unpinned-actions (writing (fixture-jobs "unguarded.yml")))
                '(("draft" . "actions/attest-build-provenance@v3"))))

(define (workflow name)
  (workflow-jobs (file->lines (build-path root ".github" "workflows" name))))

(define release-workflows '("pkg-build.yml" "release-canary.yml" "release.yml"))

(test-case "every job of release.yml, and the other release workflows' writing jobs, pin by commit"
  (check-equal? (unpinned-actions (workflow "release.yml")) '())
  (for ([name (in-list release-workflows)])
    (check-equal? (unpinned-actions (writing (workflow name))) '() name)))

(test-case "no release workflow reads a command through < <( ), which hides the command's failure"
  (for ([name (in-list release-workflows)])
    (check-false (regexp-match #px"<\\s+<\\(" (file->string (build-path root ".github" "workflows" name)))
                 name)))

(test-case "local reusable workflows are the jobs' uses: under .github/workflows"
  (check-equal? (called-workflows (fixture-jobs "guarded.yml"))
                '(".github/workflows/build.yml" ".github/workflows/report.yaml")))

(test-case "no workflow in the repository skips a job behind an implicit success()"
  (define workflows (repository-workflows))
  (check-true (pair? workflows))
  (for ([file (in-list workflows)])
    (define jobs (workflow-jobs (file->lines file)))
    (check-true (pair? jobs) (format "~a has no jobs the reader found" file))
    (check-equal? (unguarded-jobs jobs) '() (format "~a" file))))

(define (called-closure file)
  (define called (called-workflows (workflow-jobs (file->lines (build-path root file)))))
  (append called (append-map called-closure called)))

(test-case "the natives' workflow inputs are exactly the build workflows the release calls"
  (define builds
    (for/list ([file (in-list (remove-duplicates (called-closure ".github/workflows/release.yml")))]
               #:when (regexp-match? #px"scripts/platform/assemble\\.sh"
                                     (file->string (build-path root file))))
      file))
  (check-equal? (sort builds string<?)
                '(".github/workflows/platform-linux-build.yml" ".github/workflows/platform-macos-build.yml"))
  (check-equal? (sort (filter (λ (input) (string-prefix? input ".github/"))
                              (release-data-inputs (read-release-data root)))
                      string<?)
                (sort builds string<?)))

(test-case "a pull request checks the natives' inputs, so a stale exclude fails before the tag"
  (check-true
   (for/or ([file (in-list (repository-workflows))])
     (define text (file->string file))
     (and (regexp-match? #px"(?m:^  pull_request:)" text)
          (regexp-match? #px"racket scripts/release/release\\.rkt natives-inputs" text)))))
