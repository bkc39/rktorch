#lang racket/base

(require (only-in racket/contract/base -> ->* cons/c listof or/c)
         (only-in racket/string string-suffix?)
         (only-in "../data/dataset.rkt" define-dataset)
         (only-in "../foreign.rkt" device/c tensor tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "image.rkt" read-image))

(define image-extensions '(".jpg" ".jpeg" ".png"))

(define (image-file? path extensions)
  (define-values (_dir name _must-be-dir?) (split-path path))
  (define lowered (string-downcase (path->string name)))
  (for/or ([ext (in-list extensions)]) (string-suffix? lowered ext)))

(define/contract-out (image-folder-classes root) ;; noqa
  (-> path-string? (listof string?))
  (for/list ([entry (in-list (directory-list root))]
             #:when (directory-exists? (build-path root entry)))
    (path->string entry)))

(define (files-under dir)
  (define entries (directory-list dir #:build? #t))
  (append (filter file-exists? entries)
          (apply append (map files-under (filter directory-exists? entries)))))

(define/contract-out (image-folder-samples root ;; noqa
                                           #:extensions
                                           [extensions image-extensions])
  (->* [path-string?] [#:extensions (listof string?)]
       (listof (cons/c path? exact-nonnegative-integer?)))
  (for*/list ([class+label (in-list (for/list ([class (image-folder-classes
                                                       root)]
                                               [label (in-naturals)])
                                      (cons class label)))]
              [file (in-list (files-under (build-path root
                                                      (car class+label))))]
              #:when (image-file? file extensions))
    (cons file (cdr class+label))))

(define-dataset image-folder (samples transform device) ;; noqa
  #:contract (->* [path-string?]
                  [#:transform (-> tensor? tensor?)
                   #:extensions (listof string?)
                   #:device (or/c #f device/c)]
                  image-folder?)
  #:init (root #:transform [transform values]
               #:extensions [extensions image-extensions]
               #:device [device #f])
  (set! samples
        (list->vector (image-folder-samples root #:extensions extensions)))
  #:length (vector-length samples)
  #:ref (i)
  (define sample (vector-ref samples i))
  (values (transform (read-image (car sample) #:mode 'rgb #:device device))
          (tensor (cdr sample) #:device device)))
