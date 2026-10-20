#lang racket/base
(require (only-in net/url path->url url->string))

(module test racket/base)

(define memory-mb 1024)
(define public-catalog "https://pkgs.racket-lang.org/")

(define (setting name default)
  (define v (getenv name))
  (if (and v (not (string=? v ""))) v default))

(define (catalog-url where)
  (if (regexp-match? #rx"^[a-z]+://" where)
      where
      (url->string (path->url (path->complete-path where)))))

(define (pkg-build name)
  (dynamic-require 'pkg-build name))

(define (make-vm name)
  (define docker-vm (pkg-build 'docker-vm))
  (docker-vm #:name name
             #:from-image "racket/pkg-build:deps-x86_64"
             #:shell '("/usr/bin/xvfb-run" "--auto-servernum" "/bin/sh" "-c")
             #:memory-mb memory-mb
             #:minimal-variant (docker-vm #:name (string-append name "-min")
                                          #:from-image "racket/pkg-build:deps-min-x86_64"
                                          #:memory-mb memory-mb)))

(module+ main
  (require (only-in racket/format ~a)
           (only-in racket/string string-split))

  (define racket-version (setting "PKG_BUILD_RACKET" "9.3"))
  (define work-dir (path->complete-path (setting "PKG_BUILD_WORK_DIR" ".scratch/pkg-build/out/work")))
  (define jobs (string->number (setting "PKG_BUILD_JOBS" "")))
  (define catalogs
    (append (map catalog-url (string-split (setting "PKG_BUILD_CATALOG" "")))
            (list public-catalog)))
  (printf "pkg-build: racket ~a, jobs ~a, work dir ~a\ncatalogs: ~a\n"
          racket-version (or jobs "default") work-dir catalogs)
  ((pkg-build 'build-pkgs)
   #:work-dir work-dir
   #:snapshot-url (~a "https://mirror.racket-lang.org/releases/" racket-version "/")
   #:installer-name (~a "racket-" racket-version "-x86_64-linux-natipkg-pkg-build.sh")
   #:compile-any? #t
   #:pkgs-for-version racket-version
   #:pkg-catalogs catalogs
   #:only-packages '("torch")
   #:timeout 2400
   #:jobs jobs
   #:vms (list (make-vm "torch-pkg-build"))
   #:steps ((pkg-build 'steps-in) 'download 'summary)))
