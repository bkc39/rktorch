#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file make-directory* make-temporary-directory)
           (only-in racket/match match-define)
           (only-in racket/runtime-path define-runtime-path)
           rackunit
           (only-in "harness.rkt" with-temporary-path)
           (only-in "system-tools.rkt" command-output system-tool-directories))

  (define-runtime-path fake-tool "fixtures/fake-tool.sh")

  ;; Each case lays out bin/ (the whole PATH) and sbin/ (the only system directory),
  ;; naming the fake tools each holds, and gives what `sysctl -n vm.loadavg` prints.
  (define cases
    '(("a tool off PATH is found in a system directory"
       (("bin") ("sbin" "sysctl"))
       "sbin -n vm.loadavg")
      ("PATH wins over the system directories"
       (("bin" "sysctl") ("sbin" "sysctl"))
       "bin -n vm.loadavg")
      ("a tool in neither gives no output"
       (("bin") ("sbin"))
       #f)))

  (define (lay-out! root layout)
    (for ([entry (in-list layout)])
      (match-define (cons dir tools) entry)
      (make-directory* (build-path root dir))
      (for ([tool (in-list tools)])
        (define path (build-path root dir tool))
        (copy-file fake-tool path)
        (file-or-directory-permissions path #o755))))

  (for ([c (in-list cases)])
    (match-define (list name layout expected) c)
    (test-case name
      (with-temporary-path (root (make-temporary-directory))
        (lay-out! root layout)
        (parameterize ([current-environment-variables
                        (make-environment-variables #"PATH" (path->bytes (build-path root "bin")))]
                       [system-tool-directories (list (build-path root "sbin"))])
          (check-equal? (command-output "sysctl" "-n" "vm.loadavg") expected))))))
