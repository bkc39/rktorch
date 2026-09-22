#lang racket/base

(require (only-in file/unzip
                  call-with-unzip-entry read-zip-directory
                  zip-directory-contains?)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/contract/base
                  -> ->* and/c any/c cons/c contract-out listof or/c vectorof)
         (only-in racket/file file->string make-directory*)
         (only-in racket/list append-map index-of remove-duplicates)
         (only-in racket/port copy-port)
         ;; whole-module on purpose: the expansion needs bindings only-in
         ;; would strip
         racket/runtime-path
         (only-in racket/string string-prefix? string-split string-trim)
         (only-in "../main.rkt" tensor tensor? tensor->list to-dtype)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/util.rkt" with-temporary-file))

(provide (contract-out [word-vocab? (-> any/c boolean?)]
                       [word-vocab-words (-> word-vocab? (vectorof string?))]))

(define pair/c (cons/c string? string?))
(define language/c (or/c 'eng 'fra))

(define special-words '("<pad>" "<sos>" "<eos>"))

;; the specials head every vocabulary in this order, so an id is a position
(define/contract-out pad-id exact-nonnegative-integer? ;; noqa
  (index-of special-words "<pad>"))
(define/contract-out sos-id exact-nonnegative-integer? ;; noqa
  (index-of special-words "<sos>"))
(define/contract-out eos-id exact-nonnegative-integer? ;; noqa
  (index-of special-words "<eos>"))

(struct word-vocab (words ids))

(define/contract-out (strip-accents s) ;; noqa
  (-> string? string?)
  (list->string
   (for/list ([c (in-string (string-normalize-nfd s))]
              #:unless (eq? (char-general-category c) 'mn))
     c)))

(define/contract-out (normalize-sentence s) ;; noqa
  (-> string? string?)
  (define ascii (strip-accents (string-downcase (string-trim s))))
  (define spaced (regexp-replace* #rx"[.!?]" ascii " \\0"))
  (string-trim (regexp-replace* #rx"[^a-zA-Z!?]+" spaced " ")))

(define/contract-out tutorial-prefixes (listof string?) ;; noqa
  '("i am " "i m " "he is" "he s " "she is" "she s "
    "you are" "you re " "we are" "we re " "they are" "they re "))

(define (word-count s)
  (length (string-split s " " #:trim? #f)))

(define/contract-out (parse-pairs text ;; noqa
                                  #:source [source 'fra]
                                  #:max-length [max-length 10]
                                  #:prefixes [prefixes tutorial-prefixes])
  (->* [string?]
       [#:source language/c
        #:max-length exact-positive-integer?
        #:prefixes (or/c (listof string?) #f)]
       (listof pair/c))
  (for*/list ([line (in-list (string-split text "\n"))]
              [fields (in-value (string-split line "\t" #:trim? #f))]
              #:when (>= (length fields) 2)
              [eng (in-value (normalize-sentence (car fields)))]
              [fra (in-value (normalize-sentence (cadr fields)))]
              #:when (and (< (word-count eng) max-length)
                          (< (word-count fra) max-length)
                          (or (not prefixes)
                              (for/or ([p (in-list prefixes)])
                                (string-prefix? eng p)))))
    (if (eq? source 'fra) (cons fra eng) (cons eng fra))))

(define (sentence-words s)
  (map string->immutable-string (string-split s " ")))

(define/contract-out (words->vocab sentences) ;; noqa
  (-> (listof string?) word-vocab?)
  (define words
    (vector->immutable-vector
     (list->vector
      (remove-duplicates
       (append special-words (append-map sentence-words sentences))))))
  (word-vocab words
              (for/hash ([w (in-vector words)] [i (in-naturals)])
                (values w i))))

(define/contract-out (pairs->vocabs pairs) ;; noqa
  (-> (listof pair/c) (values word-vocab? word-vocab?))
  (values (words->vocab (map car pairs))
          (words->vocab (map cdr pairs))))

(define/contract-out (vocab-size v) ;; noqa
  (-> word-vocab? exact-positive-integer?)
  (vector-length (word-vocab-words v)))

(define/contract-out (encode-sentence v sentence) ;; noqa
  (-> word-vocab? string? (listof exact-nonnegative-integer?))
  (append
   (for/list ([w (in-list (sentence-words sentence))])
     (hash-ref (word-vocab-ids v) w
               (lambda ()
                 (raise-arguments-error 'encode-sentence
                                        "word not in the vocabulary"
                                        "word" w))))
   (list eos-id)))

(define (token-list ids)
  (if (tensor? ids) (map inexact->exact (tensor->list ids)) ids))

(define/contract-out (decode-tokens v ids) ;; noqa
  (-> word-vocab? (or/c tensor? (listof exact-nonnegative-integer?)) string?)
  (define words
    (for/list ([i (in-list (token-list ids))]
               #:break (= i eos-id)
               #:unless (or (= i pad-id) (= i sos-id)))
      (vector-ref (word-vocab-words v) i)))
  (apply string-append
         (if (null? words)
             '()
             (cons (car words)
                   (for/list ([w (in-list (cdr words))])
                     (string-append " " w))))))

(define (pad-to ids width)
  (append ids (for/list ([_ (in-range (- width (length ids)))]) pad-id)))

(define/contract-out (sentences->tensor v sentences #:width [width #f]) ;; noqa
  (->* [word-vocab? (and/c (listof string?) pair?)]
       [#:width (or/c exact-positive-integer? #f)]
       tensor?)
  (define rows (for/list ([s (in-list sentences)]) (encode-sentence v s)))
  (define longest (apply max (map length rows)))
  (when (and width (< width longest))
    (raise-arguments-error 'sentences->tensor
                           "a sentence is longer than the width"
                           "width" width "longest, with its <eos>" longest))
  (to-dtype (tensor (for/list ([r (in-list rows)])
                      (pad-to r (or width longest))))
            'int64))

(define/contract-out (pairs->tensors pairs source-vocab target-vocab ;; noqa
                                     #:width [width #f])
  (->* [(and/c (listof pair/c) pair?) word-vocab? word-vocab?]
       [#:width (or/c exact-positive-integer? #f)]
       (values tensor? tensor?))
  (values (sentences->tensor source-vocab (map car pairs) #:width width)
          (sentences->tensor target-vocab (map cdr pairs) #:width width)))

(define-runtime-path pairs-fixture "fixtures/eng-fra-excerpt.txt")

(define/contract-out (load-translation-fixture #:source [source 'fra]) ;; noqa
  (->* [] [#:source language/c] (listof pair/c))
  (parse-pairs (file->string pairs-fixture) #:source source))

(define archive-url "https://download.pytorch.org/tutorial/data.zip")
(define archive-name "pytorch-tutorial-data.zip")
(define archive-entry "data/eng-fra.txt")

(define (translation-cache-dir)
  (define override (getenv "RKTORCH_TRANSLATION_DIR"))
  (if (and override (not (string=? override "")))
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "translation")))

;; A zip's directory sits at its end, so a truncated download fails to list
;; the entry; inflating it and reading the first pair catches a body that
;; was damaged in the middle with the directory intact.
(define/contract-out (translation-archive? path) ;; noqa
  (-> path-string? boolean?)
  (with-handlers ([exn:fail? (lambda (_e) #f)])
    (and (zip-directory-contains? (read-zip-directory path) archive-entry)
         (call-with-unzip-entry
          path archive-entry
          (lambda (entry)
            (define first-line (call-with-input-file entry read-line))
            (and (string? first-line) (regexp-match? #rx"\t" first-line)))))))

(define (fetch-archive!)
  (define dest (build-path (translation-cache-dir) archive-name))
  (unless (file-exists? dest)
    (make-directory* (translation-cache-dir))
    (with-temporary-file (tmp #:template "pairs-~a.part"
                              #:directory (translation-cache-dir))
      (call/input-url (string->url archive-url)
                      (lambda (u) (get-pure-port u #:redirections 3))
                      (lambda (in)
                        (call-with-output-file tmp #:exists 'truncate
                          (lambda (out) (copy-port in out)))))
      (unless (translation-archive? tmp)
        (raise (exn:fail:network
                (format "load-translation-pairs: ~a did not answer a complete archive holding ~a; not caching"
                        archive-url archive-entry)
                (current-continuation-marks))))
      (rename-file-or-directory tmp dest #t)))
  dest)

(define/contract-out (load-translation-pairs ;; noqa
                      #:source [source 'fra]
                      #:max-length [max-length 10]
                      #:prefixes [prefixes tutorial-prefixes])
  (->* []
       [#:source language/c
        #:max-length exact-positive-integer?
        #:prefixes (or/c (listof string?) #f)]
       (listof pair/c))
  (parse-pairs (call-with-unzip-entry (fetch-archive!) archive-entry file->string)
               #:source source #:max-length max-length #:prefixes prefixes))
