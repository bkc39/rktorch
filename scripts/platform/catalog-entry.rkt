#lang racket/base
(require (only-in file/unzip call-with-unzip-entry)
         (only-in racket/file file->string)
         (only-in racket/match match-define)
         (only-in racket/path path-only)
         (only-in racket/string string-trim)
         (only-in setup/getinfo get-info/full))

(match-define (vector name source zip) (current-command-line-arguments))

(define dependencies
  (call-with-unzip-entry
   zip "info.rkt"
   (λ (info-file)
     (define info (get-info/full (path-only info-file)))
     (append (info 'deps (λ () '())) (info 'build-deps (λ () '()))))))

(write (hash 'name name
             'source source
             'checksum (string-trim (file->string (string-append zip ".CHECKSUM")))
             'author ""
             'description ""
             'tags '()
             'dependencies dependencies
             'modules '()))
(newline)
