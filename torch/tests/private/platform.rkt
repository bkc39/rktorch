#lang racket/base

(require (only-in racket/match match))

(provide host-platform
         os*->platform)

(define (os*->platform os*)
  (match os*
    [(or 'macosx 'darwin) 'macosx]
    [_ os*]))

(define host-platform (os*->platform (system-type 'os*)))

(module+ test
  (require rackunit)

  (test-case "a nix-built darwin Racket's os* name reads as macOS"
    (check-equal? (map os*->platform '(darwin macosx linux windows))
                  '(macosx macosx linux windows))))
