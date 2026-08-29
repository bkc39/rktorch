#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in racket/contract/base -> any cons/c listof or/c)
         (only-in file/gunzip gunzip-through-ports)
         (only-in file/untar untar)
         (only-in racket/list first)
         (only-in racket/string string-split string-trim)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/util.rkt" with-temporary-directory)
         (only-in "data.rkt" download-audio-cached load-audio))

(provide (struct-out utterance))

(struct utterance (id path transcript) #:transparent)

(define librispeech-base-url "https://www.openslr.org/resources/12/")

(define known-splits '("dev-clean" "test-clean"))

(define split/c (apply or/c known-splits))

;; SPEAKER-CHAPTER-UTT TRANSCRIPT IN CAPS
(define/contract-out (parse-trans-line line)
  (-> string? (or/c #f (cons/c string? string?)))
  (define trimmed (string-trim line))
  (cond
    [(string=? trimmed "") #f]
    [else
     (define parts (string-split trimmed " " #:trim? #t))
     (when (< (length parts) 2)
       (error 'parse-trans-line "no transcript after the utterance id: ~e"
              line))
     (cons (first parts)
           (string-trim (substring trimmed (string-length (first parts)))))]))

(define (gzip-magic? path)
  (call-with-input-file path
    (lambda (in)
      (define bs (read-bytes 2 in))
      (and (bytes? bs) (= (bytes-length bs) 2) (equal? bs #"\037\213")))))

(define (split-archive split)
  (download-audio-cached
   (format "librispeech/~a.tar.gz" split)
   (string-append librispeech-base-url split ".tar.gz")
   #:valid? gzip-magic?))

(define (evict! archive e)
  (with-handlers ([exn:fail? void])
    (delete-file archive))
  (raise e))

(define (gunzip-untar! archive dest)
  (define-values (pipe-in pipe-out) (make-pipe (* 1024 1024)))
  (define outcome (box #f))
  (define untarrer
    (thread
     (lambda ()
       (dynamic-wind
        void
        (lambda ()
          (with-handlers ([(lambda (_) #t)
                           (lambda (e) (set-box! outcome (vector e)))])
            (untar pipe-in #:dest dest #:permissive? #f)
            (set-box! outcome 'ok)))
        ;; unblocks a writer stuck on a full pipe if untar died early
        (lambda () (close-input-port pipe-in))))))
  (define gunzip-outcome
    (dynamic-wind
     void
     (lambda ()
       (with-handlers ([(lambda (_) #t) vector])
         (call-with-input-file archive
           (lambda (in) (gunzip-through-ports in pipe-out)))
         #f))
     (lambda ()
       (close-output-port pipe-out)
       (thread-wait untarrer))))
  (define r (unbox outcome))
  (cond
    [(vector? r) (raise (vector-ref r 0))]
    [(vector? gunzip-outcome) (raise (vector-ref gunzip-outcome 0))]
    [(eq? r 'ok) (void)]
    [else (error 'librispeech "extraction did not complete for ~a"
                 archive)]))

;; extract into a private staging directory and publish by rename, so a
;; failed or concurrent extraction can never surface a partial corpus
(define (extracted-root split)
  (define archive (split-archive split))
  (define-values (parent _name _dir?) (split-path archive))
  (define dest (build-path parent split))
  (unless (directory-exists? dest)
    (with-temporary-directory (staging #:template "librispeech-extract-~a"
                                       #:directory parent)
      (with-handlers ([(lambda (_) #t) (lambda (e) (evict! archive e))])
        (gunzip-untar! archive staging)
        (unless (directory-exists?
                 (build-path staging "LibriSpeech" split))
          (error 'librispeech "~a did not contain LibriSpeech/~a"
                 archive split)))
      (with-handlers ([exn:fail:filesystem?
                       (lambda (e)
                         (unless (directory-exists? dest) (raise e)))])
        (rename-file-or-directory staging dest))))
  (build-path dest "LibriSpeech" split))

(define/contract-out (librispeech-utterances split) ;; noqa
  (-> split/c (listof utterance?))
  (define root (extracted-root split))
  (sort
   (for*/list ([speaker (in-list (directory-list root))]
               #:when (directory-exists? (build-path root speaker))
               [chapter (in-list (directory-list (build-path root speaker)))]
               #:when (directory-exists? (build-path root speaker chapter))
               [chapter-dir (in-value (build-path root speaker chapter))]
               [trans (in-value
                       (build-path chapter-dir
                                   (format "~a-~a.trans.txt"
                                           speaker chapter)))]
               [entry (in-list
                       (with-input-from-file trans
                         (lambda ()
                           (for/list ([line (in-lines)]
                                      #:do [(define parsed
                                              (parse-trans-line line))]
                                      #:when parsed)
                             parsed))))])
     (utterance (car entry)
                (build-path chapter-dir (format "~a.flac" (car entry)))
                (cdr entry)))
   string<? #:key utterance-id))

(define/contract-out (load-utterance u) ;; noqa
  (-> utterance? any)
  (load-audio (utterance-path u)))

(define-runtime-path fixture-flac "fixtures/librispeech-1272-128104-0000.flac")
(define-runtime-path fixture-trans
  "fixtures/librispeech-1272-128104-0000.txt")

(define/contract-out (load-librispeech-fixture) ;; noqa
  (-> any)
  (define-values (samples rate) (load-audio fixture-flac))
  (values samples rate
          (cdr (parse-trans-line
                (with-input-from-file fixture-trans
                  (lambda () (read-line (current-input-port) 'any)))))))
