#lang racket/base
(require (only-in file/sha1 bytes->hex-string)
         (only-in racket/file display-lines-to-file make-directory*)
         (only-in racket/path path-only)
         (only-in rackunit check-exn))

(provide make-root
         packages
         refused
         stamp-a
         stamp-b
         tarball-bytes
         tarball-sha256
         write-forms)

(define stamp-a (make-string 64 #\a))
(define stamp-b (make-string 64 #\b))
(define packages '("torch-aarch64-macosx" "torch-x86_64-linux"))
(define tarball-bytes #"pretend LGPL source tarball\n")
(define tarball-sha256 (bytes->hex-string (sha256-bytes tarball-bytes)))

(define (write-forms file lang forms)
  (make-directory* (path-only file))
  (with-output-to-file file #:exists 'truncate
    (λ ()
      (when lang (displayln lang))
      (for-each writeln forms))))

(define (platform-deps names version)
  (for/list ([name (in-list names)])
    `(,name #:platform ,(substring name 6) #:version ,version)))

(define (make-root dir
                   #:version [version "0.0.1"]
                   #:native-version [native-version "0.1"]
                   #:flake-version [flake-version version]
                   #:skeleton-version [skeleton-version native-version]
                   #:declared [declared packages]
                   #:listed [listed packages]
                   #:lgpl [lgpl '("lame-3.100.tar.gz")]
                   #:excluded [excluded '("pkgs/torch-x86_64-linux/NOTES")]
                   #:stamp [stamp stamp-a])
  (write-forms (build-path dir "torch" "info.rkt") "#lang info"
               `((define collection "torch")
                 (define version ,version)
                 (define native-version ,native-version)
                 (define deps '(("base" #:version "9.1") ,@(platform-deps declared native-version)))))
  (for ([name (in-list declared)])
    (write-forms (build-path dir "pkgs" name "info.rkt") "#lang info"
                 `((define collection "torch-native")
                   (define version ,skeleton-version)
                   (define deps '("base")))))
  (display-lines-to-file (list "{" (format "      version = ~s;" flake-version) "}")
                         (build-path dir "flake.nix")
                         #:exists 'truncate)
  (write-forms (build-path dir "torch" "foreign" "raw" "abi-stamp.rktd") #f (list stamp))
  (make-directory* (build-path dir "scripts" "platform"))
  (display-lines-to-file
   (list "# sha256  file  url"
         (format "~a  lame-3.100.tar.gz  https://example.org/lame-3.100.tar.gz" tarball-sha256))
   (build-path dir "scripts" "platform" "sources.txt")
   #:exists 'truncate)
  (write-forms (build-path dir "scripts" "release" "release.rktd") #f
               `((platform-zips
                  ,@(for/list ([name (in-list listed)])
                      (format "~a/pkg/~a.zip" name name)))
                 (lgpl-sources ,@lgpl)
                 (natives-inputs "pkgs")
                 (natives-inputs-exclude ,@excluded)))
  dir)

(define (refused rx thunk)
  (check-exn (λ (e) (and (exn:fail:user? e) (regexp-match? rx (exn-message e)))) thunk))
