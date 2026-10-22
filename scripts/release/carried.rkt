#lang racket/base
(require (only-in file/sha1 bytes->hex-string)
         (only-in json jsexpr?)
         (only-in racket/contract/base -> listof)
         (only-in racket/contract/region define/contract)
         (only-in racket/file file->string)
         (only-in racket/match match)
         (only-in racket/string string-trim)
         (only-in "../platform/catalog-entry.rkt" checksum-path)
         (only-in "checks.rkt" refuse)
         (only-in "github.rkt" download-url))

(provide carried-source-urls
         check-carried)

(define (released-package manifest name)
  (match manifest
    [(hash-table ['platform-packages (? list? packages)] [_ _] ...)
     (or (findf (λ (package) (equal? (hash-ref package 'name #f) name)) packages)
         (refuse "the previous manifest.json has no platform package ~a" name))]
    [_ (refuse "the previous manifest.json has no platform-packages list")]))

(define/contract (check-carried manifest catalog name zip)
  (-> jsexpr? hash? string? path-string? void?)
  (define released-sha256 (hash-ref (released-package manifest name) 'sha256 #f))
  (define released-checksum (hash-ref (hash-ref catalog name (hash)) 'checksum #f))
  (define checksum-file (checksum-path zip))
  (define sha256 (bytes->hex-string (call-with-input-file zip sha256-bytes)))
  (define checksum (string-trim (file->string checksum-file)))
  (cond
    [(not released-sha256)
     (refuse "the previous manifest.json's ~a entry has no sha256" name)]
    [(not released-checksum)
     (refuse "the previous catalog-entry.rktd has no checksum for ~a" name)]
    [(not (string=? sha256 released-sha256))
     (refuse "~a has sha256 ~a, but the previous manifest.json released ~a" zip sha256 released-sha256)]
    [(not (string=? checksum released-checksum))
     (refuse "~a holds ~a, but the previous catalog-entry.rktd's ~a checksum is ~a"
             checksum-file checksum name released-checksum)]
    [else (void)]))

(define/contract (carried-source-urls manifest repository)
  (-> jsexpr? string? (listof string?))
  (match manifest
    [(hash-table ['natives-tag (? string? natives-tag)] ['sources (? list? sources)] [_ _] ...)
     (for/list ([source (in-list sources)])
       (match source
         [(hash-table ['file (? string? file)] [_ _] ...) (download-url repository natives-tag file)]
         [_ (refuse "a source in the previous manifest.json has no file: ~s" source)]))]
    [_ (refuse "the previous manifest.json needs a natives-tag string and a sources list")]))
