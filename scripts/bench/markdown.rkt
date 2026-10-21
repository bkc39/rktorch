#lang racket/base

(require (only-in racket/format ~a ~r)
         (only-in racket/list remove-duplicates)
         (only-in racket/string string-join string-trim))

(provide ref
         num
         load-text
         commit
         meta-text
         host
         execution
         invocation
         one-line
         cell
         table
         median
         iqr
         select
         by-context
         settings-text
         impl-of
         rktorch?
         (struct-out section)
         make-section)

(define (ref r . keys)
  (for/fold ([v r]) ([k (in-list keys)])
    (and (hash? v) (hash-ref v k #f))))

(define (num x [digits 1])
  (if (real? x) (~r x #:precision (list '= digits)) "-"))

(define (load-text r)
  (define l (ref r 'load))
  (if (pair? l) (num (car l)) "-"))

(define (commit r)
  (define sha (ref r 'meta 'git_sha))
  (format "`~a`~a"
          (if (string? sha) (substring sha 0 (min 7 (string-length sha))) "?")
          (if (ref r 'meta 'git_dirty) " (dirty)" "")))

(define (meta-text r key)
  (or (ref r 'meta key) "?"))

(define (host r)
  (format "~a (~a, ~a, cpus ~a)" (meta-text r 'host) (meta-text r 'cpu) (meta-text r 'os)
          (meta-text r 'affinity)))

(define (execution r)
  (define m (ref r 'meta))
  (format "~a, ~a threads (~a)~a"
          (ref m 'device) (ref m 'threads) (ref m 'threads_source)
          (if (string? (ref m 'gpu)) (format ", ~a" (ref m 'gpu)) "")))

(define (invocation r)
  (format "~a, ~a, ~a, run ~a" (commit r) (host r) (execution r) (ref r 'run_id)))

(define (one-line text)
  (regexp-replace* #rx"[ \t]*[\r\n]+[ \t]*" (string-trim (~a text)) " / "))

(define (cell text)
  (regexp-replace* #rx"\\|" (one-line text) "\\\\|"))

(define (table header rows)
  (string-join
   (append (list (string-join header " | " #:before-first "| " #:after-last " |")
                 (string-join (map (lambda (_) "---") header)
                              "|"
                              #:before-first "|"
                              #:after-last "|"))
           (for/list ([row (in-list rows)])
             (string-join (map cell row) " | " #:before-first "| " #:after-last " |")))
   "\n"))

(define (median r) (ref r 'stats 'median))
(define (iqr r) (ref r 'stats 'iqr))

(define (select records suite group)
  (for/list ([r (in-list records)]
             #:when (and (equal? (ref r 'suite) suite) (or (not group) (equal? (ref r 'group) group))))
    r))

(define (by-context records context)
  (for/list ([c (in-list (remove-duplicates (map context records)))])
    (cons c (filter (lambda (r) (equal? (context r) c)) records))))

(define (settings-text settings)
  (if (hash? settings)
      (string-join (for/list ([k (in-list (sort (hash-keys settings) symbol<?))]
                              #:unless (regexp-match? #rx"^RKTORCH_.*_DIR$" (symbol->string k)))
                     (format "~a=~a" k (hash-ref settings k)))
                   " ")
      "-"))

(define (impl-of r)
  (or (ref r 'impl) (ref r 'meta 'impl) "rktorch"))

(define (rktorch? r) (equal? (impl-of r) "rktorch"))

(struct section (suite group title render context keep))

(define (make-section suite group title render context #:keep [keep (lambda (_r) #t)])
  (section suite group title render context keep))
