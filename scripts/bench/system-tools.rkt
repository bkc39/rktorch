#lang racket/base

(require (only-in racket/match match-define)
         (only-in racket/port open-output-nowhere port->string)
         (only-in racket/string string-trim)
         (only-in racket/system process*/ports))

(provide system-tool-directories
         command-output)

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
