#lang racket/base

(require (only-in ffi/unsafe [-> ffi->] _fun _int get-ffi-obj)
         (only-in racket/contract/base -> ->* >/c listof list/c or/c)
         (only-in racket/file find-files)
         (only-in racket/list first second)
         (only-in racket/match match)
         (only-in racket/path file-name-from-path path-only)
         (only-in racket/port open-output-nowhere)
         (only-in racket/string string-suffix?)
         (only-in setup/getinfo get-info/full)
         (only-in "../torch/private/contract.rkt" define/contract-out))

(provide (struct-out overran)
         cover-slowdown
         dumping-px
         file-seconds
         running-px)

(define file-seconds (make-parameter 300))

(define cover-slowdown 3)

(struct overran (file previous seconds) #:transparent)

(define/contract-out (cover-outcome result)
  (-> (or/c exact-integer? overran?) (or/c 'passed 'failed string?))
  (match result
    [0 'passed]
    [1 'failed]
    [142 "raco cover was ended by SIGALRM: a test's process deadline fired (its notice is above)"]
    [(overran file #f seconds)
     (format "~a ran past its ~a s bound under raco cover; the run and its children were ended"
             file seconds)]
    [(overran file previous seconds)
     (format (string-append "~a ran past its ~a s bound under raco cover, or ~a before it did:"
                            " cover logs each start late; the run and its children were ended")
             file seconds previous)]
    [code
     #:when (> code 128)
     (format "raco cover was ended by signal ~a" (- code 128))]
    [code (format "raco cover exited with status ~a" code)]))

(define timeouts/c (listof (list/c string? exact-positive-integer?)))

(define (info-file? p)
  (equal? (file-name-from-path p) (string->path "info.rkt")))

(define (timeouts-in dir)
  (match (get-info/full dir)
    [#f '()]
    [info
     (for/list ([entry (in-list (info 'test-timeouts (lambda () '())))])
       (list (path->string (build-path dir (first entry))) (second entry)))]))

(define/contract-out (test-timeouts [root "torch"])
  (->* () (path-string?) timeouts/c)
  (match (for*/list ([info (in-list (find-files info-file? root))]
                     [entry (in-list (timeouts-in (path-only info)))])
           entry)
    ['() (error 'coverage "no info.rkt under ~a declares test-timeouts" root)]
    [entries entries]))

(define/contract-out (bound-for timeouts)
  (-> timeouts/c (-> string? (>/c 0)))
  (lambda (file)
    (* cover-slowdown
       (or (for/first ([entry (in-list timeouts)]
                       #:when (string-suffix? file (string-append "/" (first entry))))
             (second entry))
           (file-seconds)))))

(define running-px #px"^cover: running file: (.*)$")
(define dumping-px #px"^cover: dumping coverage info")

(struct running (file previous started))

(define (now) (current-inexact-monotonic-milliseconds))

(define (note-line! state line)
  (match line
    [(pregexp running-px (list _ file))
     (set-box! state (running file (match (unbox state) [(running f _ _) f] [_ #f]) (now)))]
    [(pregexp dumping-px) (set-box! state #f)]
    [_ (void)]))

(define (echo out line)
  (with-handlers ([exn:fail? (lambda (_) (open-output-nowhere))])
    (displayln line out)
    (flush-output out)
    out))

(define (pump-lines in out state)
  (let pump ([out out])
    (match (read-line in 'any)
      [(? eof-object?) (void)]
      [line
       (note-line! state line)
       (pump (echo out line))])))

(define (watch sp state bound)
  (let poll ()
    (define ended? (sync/timeout 1 sp))
    (match (unbox state)
      [_ #:when ended? (subprocess-status sp)]
      [(running file previous started)
       #:when (> (- (now) started) (* 1000 (bound file)))
       (overran file previous (bound file))]
      [_ (poll)])))

(define c-kill (get-ffi-obj "kill" #f (_fun _int _int ffi-> _int)))
(define sigkill 9)

(define (end-group! sp)
  (when (eq? 'running (subprocess-status sp))
    (subprocess-kill sp #t))
  (c-kill (- (subprocess-pid sp)) sigkill)
  (sync/timeout 10 sp))

(define (call-with-group-ended sp thunk after)
  (dynamic-wind void
                thunk
                (lambda ()
                  (parameterize-break #f
                    (end-group! sp)
                    (after)))))

(define/contract-out (run-watched command #:bound bound)
  (-> (listof path-string?) #:bound (-> string? (>/c 0)) (or/c exact-integer? overran?))
  (define-values (sp out in _err) (apply subprocess #f #f 'stdout 'new command))
  (close-output-port in)
  (define state (box #f))
  (define pump (thread (lambda () (pump-lines out (current-output-port) state))))
  (call-with-group-ended sp
                         (lambda () (watch sp state bound))
                         (lambda ()
                           (sync/timeout 10 pump)
                           (close-input-port out))))

(module+ test
  (require (only-in racket/file delete-directory/files make-temporary-directory
                    make-temporary-file)
           (only-in racket/logging with-logging-to-port)
           (only-in racket/port with-output-to-string)
           (only-in racket/system system*)
           (only-in rackunit check-equal? check-false check-pred check-regexp-match check-true
                    fail test-case)
           (only-in threading ~>))

  (define sh (find-executable-path "sh"))

  (define (alive? pid)
    (parameterize ([current-error-port (open-output-nowhere)])
      (system* sh "-c" (format "kill -0 ~a" pid))))

  (define (gone-soon? pid)
    (for/or ([_ (in-range 50)])
      (or (not (alive? pid)) (begin (sleep 0.1) #f))))

  (define (watched script #:bound [seconds 1] #:out [out (open-output-string)])
    (define start (now))
    (define result (box 'unfinished))
    (define t
      (parameterize ([current-output-port out])
        (thread (lambda ()
                  (set-box! result
                            (run-watched (list sh "-c" script)
                                         #:bound (lambda (_file) seconds)))))))
    (unless (sync/timeout 30 t)
      (break-thread t)
      (sync/timeout 15 t)
      (fail "run-watched did not return within 30 s"))
    (values (unbox result)
            (- (now) start)
            (if (string-port? out) (get-output-string out) "")))

  (define (pid-script pid-file . rest)
    (apply string-append
           "echo 'cover: running file: /r/a.rkt'; "
           (format "sleep 60 & echo $! > ~a; " pid-file)
           rest))

  (define (pid-in pid-file)
    (match (with-handlers ([exn:fail? (lambda (_) #f)]) (call-with-input-file pid-file read))
      [(? exact-integer? pid) pid]
      [_ #f]))

  (define (read-pid pid-file)
    (for/or ([_ (in-range 100)])
      (or (pid-in pid-file) (begin (sleep 0.05) #f))))

  (test-case "a cover run ended by a signal is an error, not a report"
    (check-equal? (map cover-outcome '(0 1)) '(passed failed))
    (check-regexp-match #rx"SIGALRM: a test's process deadline fired" (cover-outcome 142))
    (check-regexp-match #rx"ended by signal 9" (cover-outcome 137))
    (check-regexp-match #rx"exited with status 2" (cover-outcome 2)))

  (test-case "a file past its bound ends the run and is named"
    (define-values (result ms _out)
      (watched "echo 'cover: running file: /r/torch/tests/slow-test.rkt'; sleep 60"))
    (check-equal? result (overran "/r/torch/tests/slow-test.rkt" #f 1))
    (check-true (< ms 15000.0) (format "the run ended after ~a ms" ms))
    (check-regexp-match #rx"^/r/torch/tests/slow-test.rkt ran past its 1 s bound"
                        (cover-outcome result)))

  (test-case "an overrun names the file before it too, since cover logs each start late"
    (define-values (result _ms _out)
      (watched (string-append "echo 'cover: running file: /r/a.rkt'; "
                              "echo 'cover: running file: /r/b.rkt'; sleep 60")))
    (check-equal? result (overran "/r/b.rkt" "/r/a.rkt" 1))
    (check-regexp-match #rx"^/r/b.rkt ran past its 1 s bound under raco cover, or /r/a.rkt"
                        (cover-outcome result)))

  (test-case "the children a test started die with an overrun"
    (define pid-file (make-temporary-file "coverage-pid~a"))
    (define-values (result _ms _out) (watched (pid-script pid-file "wait")))
    (define pid (read-pid pid-file))
    (delete-file pid-file)
    (check-pred overran? result)
    (check-true (gone-soon? pid) (format "child ~a outlived the run" pid)))

  (test-case "a run that exits leaves no child holding its output open"
    (define pid-file (make-temporary-file "coverage-pid~a"))
    (define-values (result ms out)
      (watched (pid-script pid-file "echo last; exit 0") #:bound 30))
    (define pid (read-pid pid-file))
    (delete-file pid-file)
    (check-equal? result 0)
    (check-true (< ms 5000.0) (format "the run returned after ~a ms" ms))
    (check-regexp-match #rx"last\n$" out)
    (check-true (gone-soon? pid) (format "child ~a outlived the run" pid)))

  (test-case "a break ends the run's whole process group"
    (define pid-file (make-temporary-file "coverage-pid~a"))
    (define t
      (parameterize ([current-output-port (open-output-nowhere)])
        (thread (lambda ()
                  (with-handlers ([exn:break? void])
                    (run-watched (list sh "-c" (pid-script pid-file "wait"))
                                 #:bound (lambda (_file) 600)))))))
    (define pid (read-pid pid-file))
    (delete-file pid-file)
    (check-true (alive? pid))
    (break-thread t)
    (check-true (and (sync/timeout 15 t) #t) "the broken run did not return")
    (check-true (gone-soon? pid) (format "child ~a outlived the break" pid)))

  (test-case "writing the report stops the last file's clock"
    (define-values (result _ms _out)
      (watched (string-append "echo 'cover: running file: /r/a.rkt'; "
                              "echo 'cover: dumping coverage info into \"coverage\"'; "
                              "sleep 3; exit 0")))
    (check-equal? result 0))

  (test-case "a dead output port does not stop the pump draining cover"
    (define broken
      (make-output-port 'broken always-evt
                        (lambda (_bs _start _end _non-block? _breakable?)
                          (raise (exn:fail "error writing to stream port"
                                           (current-continuation-marks))))
                        void))
    (define-values (result _ms _out)
      (watched (string-append "echo 'cover: running file: /r/a.rkt'; "
                              "i=0; while [ $i -lt 4000 ]; do "
                              "echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx; "
                              "i=$((i+1)); done; exit 0")
               #:bound 20 #:out broken))
    (check-equal? result 0))

  (test-case "a run inside its bound returns cover's status, its stderr in order"
    (define-values (failed _ms1 out)
      (watched (string-append "echo 'cover: running file: /r/a.rkt'; echo notice >&2; "
                              "echo 'cover: running file: /r/b.rkt'; exit 1")
               #:bound 30))
    (check-equal? failed 1)
    (check-equal? out "cover: running file: /r/a.rkt\nnotice\ncover: running file: /r/b.rkt\n")
    (define-values (quiet _ms2 _out2) (watched "sleep 2; exit 0"))
    (check-equal? quiet 0 "no start line, no bound"))

  (define-logger cover)

  (test-case "the patterns match what cover's logger prints"
    (define out
      (with-output-to-string
        (lambda ()
          (with-logging-to-port (current-output-port)
            #:logger (current-logger)
            (lambda ()
              (log-cover-info "running file: ~a" "/r/torch/tests/a-test.rkt")
              (log-cover-info "dumping coverage info into ~s" "coverage"))
            'info))))
    (match (regexp-split #rx"\n" out)
      [(list running dumping "")
       (check-equal? (regexp-match running-px running)
                     (list running "/r/torch/tests/a-test.rkt"))
       (check-true (regexp-match? dumping-px dumping))]
      [lines (fail (format "cover's logger printed ~s" lines))]))

  (test-case "every info.rkt under torch/ gives its test-timeouts"
    (define repo
      (~> (#%variable-reference) variable-reference->module-source path-only (build-path 'up)))
    (define bound (bound-for (parameterize ([current-directory repo]) (test-timeouts))))
    (check-equal? (bound "/w/torch/tests/python-cross-test.rkt") 3600)
    (check-equal? (bound "/w/torch/pkg-build/smoke.rkt") 900)
    (check-equal? (bound "/w/torch/tests/no-entry-test.rkt") (* cover-slowdown 300)))

  (test-case "no test-timeouts anywhere is an error, not the default for every file"
    (define dir (make-temporary-directory "coverage-test~a"))
    (check-regexp-match #rx"no info.rkt under .* declares test-timeouts"
                        (with-handlers ([exn:fail? exn-message]) (test-timeouts dir)))
    (delete-directory/files dir))

  (test-case "a file's bound is its test-timeouts entry, else the default, times the slowdown"
    (define bound (bound-for '(("torch/tests/loader-workers-test.rkt" 600))))
    (check-equal? (bound "/w/torch/tests/loader-workers-test.rkt") (* cover-slowdown 600))
    (check-equal? (bound "/w/torch/tests/race-stress-test.rkt") (* cover-slowdown 300))
    (check-equal? (parameterize ([file-seconds 10]) (bound "/w/torch/tests/a.rkt"))
                  (* cover-slowdown 10))))
