#lang racket/base

(require (only-in racket/contract/base -> listof)
         (only-in racket/file file->bytes)
         (only-in racket/string string-suffix?)
         (only-in "../private/contract.rkt" define/contract-out)
         (only-in "../private/download.rkt" call-with-verified-download)
         (only-in "../private/safetensors.rkt" safetensors-subset)
         (only-in "../private/util.rkt" cache-dir env-setting))

;; with `keys`, a prefix, only the file's first `size` bytes are fetched:
;; its header and the entries with that prefix, which `sha256` covers
(struct checkpoint (file repo revision size sha256 weights keys))

;; timm's copies of torchvision's IMAGENET1K_V1 weights, pinned to a commit
(define checkpoints
  (list
   (cons 'resnet18-imagenet1k-v1
         (checkpoint "resnet18-imagenet1k-v1.safetensors"
                     "timm/resnet18.tv_in1k"
                     "bbd144b3e5565108aad885f145491d11bc6ce807"
                     46807446
                     "694f673df6520a3158624e8a89af086f59923ee4cd7436fe5bc3bc71d295ad81"
                     "torchvision.models.ResNet18_Weights.IMAGENET1K_V1"
                     #f))
   (cons 'resnet34-imagenet1k-v1
         (checkpoint "resnet34-imagenet1k-v1.safetensors"
                     "timm/resnet34.tv_in1k"
                     "1b7b21cca82ff974d341713f777bf740c2db38c4"
                     87278522
                     "0bb82595a564991a9d424708b33f5d843f5aaed7f6c0886ff10849ff97022235"
                     "torchvision.models.ResNet34_Weights.IMAGENET1K_V1"
                     #f))
   (cons 'resnet50-imagenet1k-v1
         (checkpoint "resnet50-imagenet1k-v1.safetensors"
                     "timm/resnet50.tv_in1k"
                     "78f3ecfdb38e06d9b8397f662e7ab8fee96026fa"
                     102469840
                     "5d061a3c593d795bfe682d9b152bafbcf550579873492def3515b46db1189888"
                     "torchvision.models.ResNet50_Weights.IMAGENET1K_V1"
                     #f))
   (cons 'vgg16-features-imagenet1k-v1
         (checkpoint "vgg16-features-imagenet1k-v1.safetensors"
                     "timm/vgg16.tv_in1k"
                     "b8d8aa2dd860af9233c8c67385a8097fd6c35d3f"
                     58861562
                     "d7d175e1efcf3c245e9f8f2dd2643071784a0867e18e7ace320ddc2836fae3ff"
                     "torchvision.models.VGG16_Weights.IMAGENET1K_V1, features"
                     "features."))))

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

(define (notice c url)
  (string-append (checkpoint-weights c) "\n"
                 "fetched from " url
                 (if (checkpoint-keys c)
                     (format ", its first ~a bytes, keeping the entries named ~a*"
                             (checkpoint-size c) (checkpoint-keys c))
                     "")
                 "\n"
                 "sha256 " (checkpoint-sha256 c) "\n"
                 licence))

(define (fetch! c dest)
  (define url (checkpoint-url c))
  (define keys (checkpoint-keys c))
  (call-with-verified-download
   'pretrained-weights url
   (weights-dir) (checkpoint-size c) (checkpoint-sha256 c)
   #:headers (if keys
                 (list (format "Range: bytes=0-~a" (sub1 (checkpoint-size c))))
                 '())
   (lambda (tmp)
     (when keys
       (define subset
         (safetensors-subset 'pretrained-weights (file->bytes tmp) keys))
       (call-with-output-file tmp #:exists 'truncate
         (lambda (out) (write-bytes subset out))))
     (call-with-output-file (path-replace-extension dest #".txt")
       #:exists 'truncate
       (lambda (out) (write-string (notice c url) out)))
     (rename-file-or-directory tmp dest #t))))

(define/contract-out (pretrained-weights name) ;; noqa
  (-> symbol? path?)
  (define c (entry-of 'pretrained-weights name))
  (define dest (build-path (weights-dir) (checkpoint-file c)))
  (unless (file-exists? dest)
    (fetch! c dest))
  dest)
