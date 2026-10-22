#lang racket/base
(require (only-in file/unzip call-with-unzip-entry)
         (only-in racket/contract/base ->)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->string)
         (only-in racket/match match-define)
         (only-in racket/path path-only)
         (only-in racket/string string-trim)
         (only-in setup/getinfo get-info/full)
         (only-in "../../nix/racket-deps/racket-deps.rkt" info-dependencies))

(provide catalog-entry
         checksum-path
         zip-catalog-entry)

(define/contract (checksum-path zip)
  (-> path-string? path?)
  (string->path (format "~a.CHECKSUM" zip)))

(define (zip-dependencies zip)
  (call-with-unzip-entry
   zip "info.rkt"
   (λ (info-file) (info-dependencies (get-info/full (path-only info-file))))))

(define/contract (catalog-entry name source #:checksum checksum #:dependencies dependencies)
  (-> string? string? #:checksum string? #:dependencies list? hash?)
  (hash 'name name
        'source source
        'checksum checksum
        'author ""
        'description ""
        'tags '()
        'dependencies dependencies
        'modules '()))

(define/contract (zip-catalog-entry name source zip)
  (-> string? string? path-string? hash?)
  (catalog-entry name source
                 #:checksum (string-trim (file->string (checksum-path zip)))
                 #:dependencies (zip-dependencies zip)))

(module+ main
  (match-define (vector name source zip) (current-command-line-arguments))
  (write (zip-catalog-entry name source zip))
  (newline))
