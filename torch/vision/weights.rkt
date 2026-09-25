#lang racket/base

(require (only-in file/sha1 bytes->hex-string)
         (only-in net/url call/input-url get-pure-port string->url)
         (only-in racket/contract/base -> listof)
         (only-in racket/file make-directory*)
         (only-in racket/port copy-port)
         (only-in racket/string string-suffix?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/util.rkt"
                  cache-dir env-setting with-temporary-file))

(struct checkpoint (file repo revision size sha256 weights))

;; timm's copies of torchvision's IMAGENET1K_V1 weights, pinned to a commit
(define checkpoints
  (list
   (cons 'resnet18-imagenet1k-v1
         (checkpoint "resnet18-imagenet1k-v1.safetensors"
                     "timm/resnet18.tv_in1k"
                     "bbd144b3e5565108aad885f145491d11bc6ce807"
                     46807446
                     "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81"
                     "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"))
   (cons 'resnet34-imagenet1k-v1
         (checkpoint "resnet34-imagenet1k-v1.safetensors"
                     "timm/resnet34.tv_in1k"
                     "1b7b21cca82ff974d341713f777bf740c2db38c4"
                     87278522
                     "0bb82595a564991a9d424708b33f5d843f5aaed7f6c0886ff10849ff97022235"
                     "torchvision.models.ResNet34_Weights.IMAGENET1K_V1"))
   (cons 'resnet50-imagenet1k-v1
         (checkpoint "resnet50-imagenet1k-v1.safetensors"
                     "timm/resnet50.tv_in1k"
                     "78f3ecfdb38e06d9b8397f662e7ab8fee96026fa"
                     102469840
                     "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888"
                     "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"))))

(define hugging-face "https://huggingface.co/")

(define licence
  (string-append
   "BSD-3-Clause, the torchvision licence, Copyright (c) Soumith Chintala"
   " 2016: https://github.com/pytorch/vision/blob/main/LICENSE\n"
   "The weights were trained on ImageNet-1K; its terms of access bind"
   " whoever uses them.\n"))

(define (weights-dir)
  (cache-dir "RKTORCH_WEIGHTS_DIR" "weights"))

(define (checkpoint-url c)
  (define base (or (env-setting "RKTORCH_WEIGHTS_URL") hugging-face))
  (string-append (if (string-suffix? base "/") base (string-append base "/"))
                 (checkpoint-repo c) "/resolve/" (checkpoint-revision c)
                 "/model.safetensors"))

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

(define (notice c url)
  (string-append (checkpoint-weights c) "\n"
                 "fetched from " url "\n"
                 "sha256 " (checkpoint-sha256 c) "\n"
                 licence))

(define (fetch! c dest)
  (define url (checkpoint-url c))
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
    (call-with-output-file (path-replace-extension dest #".txt") #:exists 'truncate
      (lambda (out) (write-string (notice c url) out)))
    (rename-file-or-directory tmp dest #t)))

(define/contract-out (pretrained-weights name) ;; noqa
  (-> symbol? path?)
  (define c (entry-of 'pretrained-weights name))
  (define dest (build-path (weights-dir) (checkpoint-file c)))
  (unless (file-exists? dest)
    (fetch! c dest))
  dest)
