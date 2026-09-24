#lang racket/base

(require (only-in file/sha1 bytes->hex-string)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/contract/base -> listof)
         (only-in racket/file make-directory*)
         (only-in racket/port copy-port)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/util.rkt" with-temporary-file))

(struct checkpoint (file size sha256 source weights))

(define checkpoints
  (list
   (cons 'resnet18-imagenet1k-v1
         (checkpoint "resnet18-imagenet1k-v1.safetensors" 46807920
                     "db77c3aa4b2b89016936bb46f01dc292eedd0d14fb30e1833919693d656f5162"
                     "https://download.pytorch.org/models/resnet18-f37072fd.pth"
                     "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"))
   (cons 'resnet34-imagenet1k-v1
         (checkpoint "resnet34-imagenet1k-v1.safetensors" 87279112
                     "4db058525703b0bdcdfb19c5b4d5e498cc8f46f05117af21d9e7aa81d415b6a8"
                     "https://download.pytorch.org/models/resnet34-b627a593.pth"
                     "torchvision.models.ResNet34_Weights.IMAGENET1K_V1"))
   (cons 'resnet50-imagenet1k-v1
         (checkpoint "resnet50-imagenet1k-v1.safetensors" 102470400
                     "86904c337f79cbc83fe044bea66de10137568ac616fe5203e3b8f10b2872aa97"
                     "https://download.pytorch.org/models/resnet50-0676ba61.pth"
                     "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"))))

(define release "https://github.com/bkc39/rktorch/releases/download/weights-v1/")

(define licence
  (string-append
   "BSD-3-Clause, the torchvision licence, Copyright (c) Soumith Chintala"
   " 2016: https://github.com/pytorch/vision/blob/main/LICENSE\n"
   "The weights were trained on ImageNet-1K; its terms of access bind"
   " whoever uses them.\n"))

(define (setting name default)
  (define v (getenv name))
  (if (and v (not (string=? v ""))) v default))

(define (weights-dir)
  (define override (setting "RKTORCH_WEIGHTS_DIR" #f))
  (if override
      (string->path override)
      (build-path (find-system-path 'cache-dir) "rktorch" "weights")))

(define (entry-of who name)
  (define e (assq name checkpoints))
  (unless e
    (raise-arguments-error who "no such checkpoint"
                           "name" name
                           "known" (map car checkpoints)))
  (cdr e))

(define/contract-out pretrained-weights-names (listof symbol?) ;; noqa
  (map car checkpoints))

(define/contract-out (pretrained-weights-cached? name) ;; noqa
  (-> symbol? boolean?)
  (file-exists? (build-path (weights-dir)
                            (checkpoint-file (entry-of 'pretrained-weights-cached?
                                                       name)))))

(define (sha256-hex path)
  (call-with-input-file path (lambda (in) (bytes->hex-string (sha256-bytes in)))))

(define (notice c)
  (string-append (checkpoint-weights c) "\n"
                 "exported from " (checkpoint-source c) "\n"
                 "sha256 " (checkpoint-sha256 c) "\n"
                 licence))

(define (fetch! c dest)
  (define url (string-append (setting "RKTORCH_WEIGHTS_URL" release)
                             (checkpoint-file c)))
  (make-directory* (weights-dir))
  ;; temp file, checked in full, then an atomic rename: a redirect page or
  ;; a transfer cut short must not reach the cache
  (with-temporary-file (tmp #:template "weights-~a.part"
                            #:directory (weights-dir))
    (call/input-url (string->url url)
                    (lambda (u) (get-pure-port u #:redirections 5))
                    (lambda (in)
                      (call-with-output-file tmp #:exists 'truncate
                        (lambda (out) (copy-port in out)))))
    (define size (file-size tmp))
    (define digest (sha256-hex tmp))
    (unless (and (= size (checkpoint-size c))
                 (string=? digest (checkpoint-sha256 c)))
      (raise-arguments-error 'pretrained-weights
                             "download does not match the published checkpoint"
                             "url" url
                             "bytes" size
                             "expected bytes" (checkpoint-size c)
                             "sha256" digest
                             "expected sha256" (checkpoint-sha256 c)))
    (call-with-output-file (path-add-extension dest #".txt") #:exists 'truncate
      (lambda (out) (write-string (notice c) out)))
    (rename-file-or-directory tmp dest #t)))

(define/contract-out (pretrained-weights name) ;; noqa
  (-> symbol? path?)
  (define c (entry-of 'pretrained-weights name))
  (define dest (build-path (weights-dir) (checkpoint-file c)))
  (unless (file-exists? dest)
    (fetch! c dest))
  dest)
