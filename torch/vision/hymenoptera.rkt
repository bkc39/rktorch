#lang racket/base

(require (only-in file/unzip make-filesystem-entry-reader unzip)
         (only-in racket/contract/base -> ->* or/c)
         (only-in "../foreign.rkt" device/c tensor?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/download.rkt" call-with-verified-download)
         (only-in "../private/util.rkt"
                  cache-dir env-setting with-temporary-directory)
         (only-in "image-folder.rkt" image-folder image-folder?))

(define archive-url
  "https://download.pytorch.org/tutorial/hymenoptera_data.zip")
(define archive-bytes 47286322)
(define archive-sha256
  "fbc41b31d544714d18dd1230b1e2b455e1557766e13e67f9f5a7a23af7c02209")

(define (hymenoptera-dir)
  (cache-dir "RKTORCH_HYMENOPTERA_DIR" "hymenoptera"))

(define (data-root) (build-path (hymenoptera-dir) "hymenoptera_data"))

(define/contract-out (hymenoptera-cached?) ;; noqa
  (-> boolean?)
  (directory-exists? (data-root)))

;; unpacked beside the cache and renamed into place, so a failed unpack
;; leaves no tree
(define (install! zip)
  (with-temporary-directory (unpacked #:template "hymenoptera-~a"
                                      #:directory (hymenoptera-dir))
    (call-with-input-file zip
      (lambda (in) (unzip in (make-filesystem-entry-reader #:dest unpacked))))
    (rename-file-or-directory (build-path unpacked "hymenoptera_data")
                              (data-root))))

(define/contract-out (hymenoptera-root) ;; noqa
  (-> path?)
  (unless (hymenoptera-cached?)
    (call-with-verified-download 'hymenoptera-root
                                 (or (env-setting "RKTORCH_HYMENOPTERA_URL")
                                     archive-url)
                                 (hymenoptera-dir) archive-bytes archive-sha256
                                 install!))
  (data-root))

(define/contract-out (hymenoptera-dataset split ;; noqa
                                          #:transform [transform values]
                                          #:device [device #f])
  (->* [(or/c 'train 'val)]
       [#:transform (-> tensor? tensor?) #:device (or/c #f device/c)]
       image-folder?)
  (image-folder (build-path (hymenoptera-root) (symbol->string split))
                #:transform transform
                #:device device))
