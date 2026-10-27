#lang racket/base

(require (only-in file/gunzip gunzip-through-ports)
         (only-in file/untar untar)
         (only-in racket/contract/base
                  -> ->* any cons/c flat-named-contract listof or/c)
         (only-in racket/list drop-right first last)
         (only-in racket/match match-define)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-split string-trim)
         (only-in "../data/dataset.rkt"
                  define-dataset default-collate? indices->list)
         (only-in "../foreign.rkt" cat device dtype shape stack tensor? zeros)
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
(define/contract-out (librispeech-root split) ;; noqa
  (-> split/c path?)
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
  dest)

(define/contract-out (librispeech-utterances split) ;; noqa
  (-> split/c (listof utterance?))
  (define root (build-path (librispeech-root split) "LibriSpeech" split))
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

(define (fixture-transcript)
  (cdr (parse-trans-line
        (with-input-from-file fixture-trans
          (lambda () (read-line (current-input-port) 'any))))))

(define/contract-out (load-librispeech-fixture) ;; noqa
  (-> any)
  (define-values (samples rate) (load-audio fixture-flac))
  (values samples rate (fixture-transcript)))

(define/contract-out (librispeech-fixture-utterance) ;; noqa
  (-> utterance?)
  (utterance "1272-128104-0000" fixture-flac (fixture-transcript)))

(define features/c
  (flat-named-contract 'tensor-with-a-time-dimension
                       (lambda (v) (and (tensor? v) (pair? (shape v))))))

(define (leading-dimensions features)
  (drop-right (shape features) 1))

(define (placement features)
  (list (leading-dimensions features) (dtype features) (device features)))

(define (padded-batch features transcripts)
  (match-define (cons head tail) features)
  (define head-placement (placement head))
  (match-define (list lead head-dtype head-device) head-placement)
  (for ([f (in-list tail)]
        #:unless (equal? (placement f) head-placement))
    (raise-arguments-error 'librispeech "items differ outside the last dimension"
                           "first item" (list (shape head) head-dtype head-device)
                           "differing item" (list (shape f) (dtype f) (device f))))
  (define lengths (map (lambda (f) (last (shape f))) features))
  (define t-max (apply max lengths))
  (define padded
    (for/list ([f (in-list features)] [t (in-list lengths)])
      (if (= t t-max)
          f
          (cat (list f (apply zeros #:dtype head-dtype #:device head-device
                              (append lead (list (- t-max t)))))
               (length lead)))))
  (values (stack padded) lengths transcripts))

(define audio-transform/c
  (-> tensor? exact-positive-integer? features/c))

(define (decoded u transform)
  (define-values (samples rate) (load-audio (utterance-path u)))
  (values (transform samples rate) (utterance-transcript u)))

(define-dataset librispeech (utterances transform) ;; noqa
  #:contract (->* [(listof utterance?)]
                  [#:transform audio-transform/c]
                  librispeech?)
  #:init (utts #:transform [transform (lambda (samples _rate) samples)])
  (set! utterances (vector->immutable-vector (list->vector utts)))
  #:length (vector-length utterances)
  #:ref (i)
  (decoded (vector-ref utterances i) transform)
  #:batch (indices collate)
  (define items
    (for/list ([i (in-list (indices->list indices))])
      (call-with-values (lambda () (decoded (vector-ref utterances i) transform))
                        list)))
  (if (default-collate? collate)
      (padded-batch (map car items) (map cadr items))
      (collate items)))
