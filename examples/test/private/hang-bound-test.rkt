#lang racket/base

(require (only-in racket/file file->string)
         (only-in racket/match match match-define)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in rackunit check check-true fail test-case)
         (only-in setup/getinfo get-info/full))

(define-runtime-path cell "../../../nix/matrix/racket-cell.sh")
(define-runtime-path runners "..")

(define hang-bound 1200)

(test-case "the cell bounds every example runner by the hang bound"
  (match (regexp-match #px"raco test --make --timeout ([0-9]+) examples/test/" (file->string cell))
    [(list _ seconds) (check >= (string->number seconds) hang-bound)]
    [#f (fail "racket-cell.sh runs no examples/test/ line")]))

(test-case "no runner's own timeout undercuts the hang bound"
  (define info (get-info/full (simplify-path runners)))
  (for ([entry (in-list (if info (info 'test-timeouts (lambda () '())) '()))])
    (match-define (list file seconds) entry)
    (check-true (>= seconds hang-bound) file)))
