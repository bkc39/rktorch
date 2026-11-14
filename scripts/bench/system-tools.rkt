#lang racket/base

(require (only-in racket/match match match-define)
         (only-in racket/port open-output-nowhere port->string)
         (only-in racket/string string-split string-trim)
         (only-in racket/system process*/ports))

(provide system-tool-directories
         command-output
         affinity-count)

;; A Nix shell's PATH leaves out /usr/sbin, where macOS keeps sysctl.
(define system-tool-directories
  (make-parameter (map string->path '("/usr/sbin" "/sbin" "/usr/bin" "/bin"))))

(define (executable? path)
  (and (file-exists? path)
       (memq 'execute (file-or-directory-permissions path))
       #t))

(define (find-system-tool program)
  (or (find-executable-path program)
      (for*/first ([dir (in-list (system-tool-directories))]
                   [path (in-value (build-path dir program))]
                   #:when (executable? path))
        path)))

(define (command-output program . args)
  (define path (find-system-tool program))
  (cond
    [(not path) #f]
    [else
     (match-define (list out in _pid _err control)
       (apply process*/ports #f #f (open-output-nowhere) path args))
     (close-output-port in)
     (define text (port->string out))
     (close-input-port out)
     (control 'wait)
     (and (zero? (control 'exit-code)) (string-trim text))]))

(define (range-size part)
  (match (regexp-match #px"^\\s*(\\d+)(?:-(\\d+))?\\s*$" part)
    [(list _ _ #f) 1]
    [(list _ lo hi) (add1 (- (string->number hi) (string->number lo)))]
    [#f #f]))

(define (affinity-count text)
  (define ranges (map range-size (string-split (or text "") ",")))
  (and (pair? ranges) (andmap exact-positive-integer? ranges) (apply + ranges)))

(module+ test
  (require rackunit)

  (test-case "the affinity count adds up a CPU list's ranges"
    (check-equal? (affinity-count "0-15") 16)
    (check-equal? (affinity-count "0,2,4,6") 4)
    (check-equal? (affinity-count "0-3, 8-11,14") 9)
    (check-false (affinity-count "3-1"))
    (check-false (affinity-count "a-b"))
    (check-false (affinity-count ""))
    (check-false (affinity-count #f))))
