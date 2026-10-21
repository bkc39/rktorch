#lang racket/base
(require (only-in file/unzip call-with-unzip-entry)
         (only-in racket/contract/base ->)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->string)
         (only-in racket/match match-define)
         (only-in racket/path path-only)
         (only-in racket/string string-trim)
         (only-in setup/getinfo get-info/full))

(provide catalog-entry)

(define (zip-dependencies zip)
  (call-with-unzip-entry
   zip "info.rkt"
   (λ (info-file)
     (define info (get-info/full (path-only info-file)))
     (append (info 'deps (λ () '())) (info 'build-deps (λ () '()))))))

(define/contract (catalog-entry name source zip)
  (-> string? string? path-string? hash?)
  (hash 'name name
        'source source
        'checksum (string-trim (file->string (format "~a.CHECKSUM" (path->complete-path zip))))
        'author ""
        'description ""
        'tags '()
        'dependencies (zip-dependencies zip)
        'modules '()))

(module+ main
  (match-define (vector name source zip) (current-command-line-arguments))
  (write (catalog-entry name source zip))
  (newline))
