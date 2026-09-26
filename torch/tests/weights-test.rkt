#lang racket/base

(module+ test
  (require (only-in net/url path->url url->string)
           (only-in racket/file make-directory*)
           (only-in rackunit check-equal? check-exn check-false check-not-false
                    check-true test-case)
           (only-in "../private/util.rkt" with-temporary-directory)
           (only-in "../vision/weights.rkt"
                    pretrained-weights pretrained-weights-cached?
                    pretrained-weights-names))

  (define (with-weights-env settings thunk)
    (define env (environment-variables-copy (current-environment-variables)))
    (for ([kv (in-list settings)])
      (environment-variables-set! env (string->bytes/utf-8 (car kv))
                                  (string->bytes/utf-8 (cdr kv))))
    (parameterize ([current-environment-variables env]) (thunk)))

  (define (directory-url dir)
    (url->string (path->url (path->directory-path dir))))

  (define name 'resnet18-imagenet1k-v1)
  (define file "resnet18-imagenet1k-v1.safetensors")
  (define hub-path
    (string-append "timm/resnet18.tv_in1k/resolve/"
                   "bbd144b3e5565108aad885f145491d11bc6ce807/model.safetensors"))

  ;; a mirror has Hugging Face's layout, as HF_ENDPOINT mirrors do
  (define (mirror-file mirror)
    (define path (build-path mirror hub-path))
    (define-values (dir _name _dir?) (split-path path))
    (make-directory* dir)
    path)

  (test-case "the published checkpoints"
    (check-equal? pretrained-weights-names
                  '(resnet18-imagenet1k-v1 resnet34-imagenet1k-v1
                                           resnet50-imagenet1k-v1)))

  (test-case "an unknown name is an error that lists the known ones"
    (check-exn #rx"no such checkpoint.*resnet18-imagenet1k-v1"
               (lambda () (pretrained-weights 'vgg99)))
    (check-exn #rx"no such checkpoint"
               (lambda () (pretrained-weights-cached? 'vgg99))))

  (test-case "a cached file is answered as it is, with no fetch"
    (with-temporary-directory (cache)
      (with-weights-env
       (list (cons "RKTORCH_WEIGHTS_DIR" (path->string cache))
             (cons "RKTORCH_WEIGHTS_URL" "file:///nonexistent/"))
       (lambda ()
         (check-false (pretrained-weights-cached? name))
         (call-with-output-file (build-path cache file)
           (lambda (out) (write-bytes #"stand-in" out)))
         (check-true (pretrained-weights-cached? name))
         (check-equal? (pretrained-weights name) (build-path cache file))))))

  (test-case "a download that is not the checkpoint never reaches the cache"
    (with-temporary-directory (mirror)
      (with-temporary-directory (cache)
        (call-with-output-file (mirror-file mirror)
          (lambda (out) (write-bytes #"a redirect page, say" out)))
        (with-weights-env
         (list (cons "RKTORCH_WEIGHTS_DIR" (path->string cache))
               (cons "RKTORCH_WEIGHTS_URL" (directory-url mirror)))
         (lambda ()
           (check-exn #rx"does not match the published file.*bytes: 20"
                      (lambda () (pretrained-weights name)))
           (check-false (pretrained-weights-cached? name))
           (check-equal? (directory-list cache) '()))))))

  (test-case "a mirror's URL may leave off its trailing slash"
    (with-temporary-directory (mirror)
      (with-temporary-directory (cache)
        (call-with-output-file (mirror-file mirror)
          (lambda (out) (write-bytes #"not the checkpoint" out)))
        (define bare
          (regexp-replace #rx"/$" (directory-url mirror) ""))
        (with-weights-env
         (list (cons "RKTORCH_WEIGHTS_DIR" (path->string cache))
               (cons "RKTORCH_WEIGHTS_URL" bare))
         (lambda ()
           (check-exn (regexp (string-append
                               "url: \""
                               (regexp-quote bare)
                               "/" (regexp-quote hub-path) "\""))
                      (lambda () (pretrained-weights name))
                      "the file was found and read, so the slash was added"))))))

  (test-case "the right size is not enough: the checksum decides"
    (with-temporary-directory (mirror)
      (with-temporary-directory (cache)
        (call-with-output-file (mirror-file mirror)
          (lambda (out) (file-truncate out 46807446)))
        (with-weights-env
         (list (cons "RKTORCH_WEIGHTS_DIR" (path->string cache))
               (cons "RKTORCH_WEIGHTS_URL" (directory-url mirror)))
         (lambda ()
           (check-exn #rx"does not match the published file.*sha256: \"[0-9a-f]+\""
                      (lambda () (pretrained-weights name)))
           (check-equal? (directory-list cache) '())))))))
