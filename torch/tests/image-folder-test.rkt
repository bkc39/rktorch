#lang racket/base

(module+ test
  (require (only-in net/url path->url url->string)
           (only-in racket/file copy-directory/files make-directory*)
           ;; whole-module: define-runtime-path needs phase-1 bindings
           ;; only-in strips
           racket/runtime-path
           (only-in rackunit
                    check-equal? check-exn check-false check-true test-case)
           (only-in "../data/dataset.rkt" dataset-ref)
           (only-in "../main.rkt"
                    item length tensor-dtype tensor-shape to-dtype)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../vision/hymenoptera.rkt"
                    hymenoptera-cached? hymenoptera-dataset hymenoptera-root)
           (only-in "../vision/image-folder.rkt"
                    image-folder image-folder-classes image-folder-samples))

  (define-runtime-path fixture "../vision/fixtures/hymenoptera")
  (define-runtime-path images "../vision/fixtures/images")

  (define (names samples)
    (for/list ([s (in-list samples)])
      (define-values (_dir name _must-be-dir?) (split-path (car s)))
      (list (path->string name) (cdr s))))

  (define (with-env settings thunk)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([kv (in-list settings)])
      (environment-variables-set! env (string->bytes/utf-8 (car kv))
                                  (string->bytes/utf-8 (cdr kv))))
    (parameterize ([current-environment-variables env]) (thunk)))

  (test-case "one class per directory, labelled in name order"
    (check-equal? (image-folder-classes fixture) '("ants" "bees"))
    (check-equal? (names (image-folder-samples fixture))
                  '(("11381045_b352a47d8c.jpg" 0) ("8398478_50ef10c47a.jpg" 0)
                    ("10870992_eebeeb3a12.jpg" 1) ("26589803_5ba7000313.jpg" 1))))

  (test-case "an item is the RGB image and its label, transformed on request"
    (define folder (image-folder fixture))
    (check-equal? (length folder) 4)
    (define-values (image label) (dataset-ref folder 2))
    (check-equal? (tensor-shape image) '(3 464 500))
    (check-equal? (tensor-dtype image) 'uint8)
    (check-equal? (item label) 1)
    (check-equal? (tensor-dtype label) 'int64)
    (define-values (floats _)
      (dataset-ref (image-folder fixture
                                 #:transform (lambda (x) (to-dtype x 'float32)))
                   0))
    (check-equal? (tensor-dtype floats) 'float32))

  (test-case "only image files count, nested directories included"
    (with-temporary-directory (root)
      (make-directory* (build-path root "cats" "more"))
      (make-directory* (build-path root "dogs"))
      (for ([copy (in-list '(("gradient.png" "cats" "b.PNG")
                             ("smooth.jpg" "cats" "a.jpeg")
                             ("smooth.jpg" "cats" "more" "c.jpg")
                             ("gradient.png" "cats" "notes")
                             ("gradient.png" "cats" "anim.gif")
                             ("gradient-gray.png" "dogs" "d.png")))])
        (copy-file (build-path images (car copy))
                   (apply build-path root (cdr copy))))
      (call-with-output-file (build-path root "stray.jpg")
        (lambda (out) (write-bytes #"not a class" out)))
      (check-equal? (image-folder-classes root) '("cats" "dogs"))
      (check-equal? (names (image-folder-samples root))
                    '(("a.jpeg" 0) ("b.PNG" 0) ("c.jpg" 0) ("d.png" 1)))
      (check-equal? (names (image-folder-samples root #:extensions '(".png")))
                    '(("b.PNG" 0) ("d.png" 1)))
      (check-equal? (names (image-folder-samples root #:extensions '(".PNG")))
                    '(("b.PNG" 0) ("d.png" 1))
                    "case is ignored on both sides")
      (define-values (gray _) (dataset-ref (image-folder root) 3))
      (check-equal? (car (tensor-shape gray)) 3 "decoded as RGB")))

  (test-case "the ants-and-bees tree is read from the cache as it lies"
    (with-temporary-directory (cache)
      (for ([split (in-list '("train" "val"))])
        (make-directory* (build-path cache "hymenoptera_data"))
        (copy-directory/files fixture
                              (build-path cache "hymenoptera_data" split)))
      (with-env (list (cons "RKTORCH_HYMENOPTERA_DIR" (path->string cache))
                      (cons "RKTORCH_HYMENOPTERA_URL" "file:///nonexistent/"))
        (lambda ()
          (check-true (hymenoptera-cached?))
          (check-equal? (hymenoptera-root) (build-path cache "hymenoptera_data"))
          (check-equal? (length (hymenoptera-dataset 'val)) 4)
          (check-exn exn:fail:contract?
                     (lambda () (hymenoptera-dataset 'test)))))))

  (test-case "without an override the tree lives in the user's cache"
    (define env (environment-variables-copy (current-environment-variables)))
    (environment-variables-set! env #"RKTORCH_HYMENOPTERA_DIR" #f)
    (parameterize ([current-environment-variables env])
      (check-true (boolean? (hymenoptera-cached?)))))

  (test-case "an archive that is not the tutorial's never reaches the cache"
    (with-temporary-directory (release)
      (with-temporary-directory (cache)
        (call-with-output-file (build-path release "hymenoptera_data.zip")
          (lambda (out) (write-bytes #"<html>moved</html>" out)))
        (with-env
         (list (cons "RKTORCH_HYMENOPTERA_DIR" (path->string cache))
               (cons "RKTORCH_HYMENOPTERA_URL"
                     (url->string
                      (path->url (build-path release "hymenoptera_data.zip")))))
         (lambda ()
           (check-false (hymenoptera-cached?))
           (check-exn #rx"hymenoptera-root: download does not match"
                      (lambda () (hymenoptera-root)))
           (check-false (hymenoptera-cached?))
           (check-equal? (directory-list cache) '())))))))
