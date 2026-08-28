#lang racket/base

;; whole-module on purpose: the expansion needs bindings only-in would strip
(require racket/runtime-path
         (only-in file/gunzip gunzip-through-ports)
         (only-in file/untar untar)
         (only-in racket/file
                  delete-directory/files make-temporary-file)
         (only-in racket/list first)
         (only-in racket/string string-split string-trim)
         (only-in "data.rkt" download-audio-cached load-audio))

(provide (struct-out utterance)
         librispeech-utterances
         load-librispeech-fixture
         load-utterance
         parse-trans-line)

(struct utterance (id path transcript) #:transparent)

(define librispeech-base-url "https://www.openslr.org/resources/12/")

(define known-splits '("dev-clean" "test-clean"))

;; SPEAKER-CHAPTER-UTT TRANSCRIPT IN CAPS
(define (parse-trans-line line)
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
  (unless (member split known-splits)
    (error 'librispeech "unknown split ~e (expected one of ~a)"
           split known-splits))
  (download-audio-cached
   (format "librispeech/~a.tar.gz" split)
   (string-append librispeech-base-url split ".tar.gz")
   #:valid? gzip-magic?))

(define (gunzip-untar! archive dest)
  (define-values (pipe-in pipe-out) (make-pipe (* 1024 1024)))
  (define outcome (box #f))
  (define untarrer
    (thread
     (lambda ()
       (with-handlers ([exn:fail? (lambda (e) (set-box! outcome e))])
         (untar pipe-in #:dest dest)
         (set-box! outcome 'ok))
       ;; unblocks a writer stuck on a full pipe if untar died early
       (close-input-port pipe-in))))
  (define gunzip-exn
    (with-handlers ([exn:fail? values])
      (call-with-input-file archive
        (lambda (in) (gunzip-through-ports in pipe-out)))
      #f))
  (close-output-port pipe-out)
  (thread-wait untarrer)
  (define r (unbox outcome))
  (cond
    [(exn? r) (raise r)]
    [gunzip-exn (raise gunzip-exn)]
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
    (define staging
      (make-temporary-file "librispeech-extract-~a" 'directory parent))
    (with-handlers ([exn:fail?
                     (lambda (e)
                       (delete-directory/files staging #:must-exist? #f)
                       (raise e))])
      (gunzip-untar! archive staging)
      (with-handlers ([exn:fail:filesystem?
                       (lambda (e)
                         (delete-directory/files staging #:must-exist? #f)
                         (unless (directory-exists? dest) (raise e)))])
        (rename-file-or-directory staging dest))))
  (build-path dest "LibriSpeech" split))

(define (librispeech-utterances split)
  (define root (extracted-root split))
  (sort
   (for*/list ([speaker (in-list (directory-list root))]
               [chapter (in-list (directory-list (build-path root speaker)))]
               #:when #t
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

(define (load-utterance u)
  (load-audio (utterance-path u)))

(define-runtime-path fixture-flac "fixtures/librispeech-1272-128104-0000.flac")
(define-runtime-path fixture-trans
  "fixtures/librispeech-1272-128104-0000.txt")

(define (load-librispeech-fixture)
  (define-values (samples rate) (load-audio fixture-flac))
  (values samples rate
          (cdr (parse-trans-line
                (with-input-from-file fixture-trans
                  (lambda () (read-line (current-input-port) 'any)))))))
