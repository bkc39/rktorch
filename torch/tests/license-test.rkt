#lang racket/base

(module+ test
  (require (only-in racket/file file->bytes file->value)
           (only-in racket/match match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in setup/getinfo get-info/full)
           rackunit)

  (define-runtime-path spdx-ids "spdx-ids.rktd")
  (define-runtime-path torch-dir "..")
  (define-runtime-path repo-dir "../..")

  (match-define (list (list 'list-version _)
                      (list 'licenses licenses ...)
                      (list 'exceptions exceptions ...))
    (file->value spdx-ids))

  (define (license-ref? id)
    (regexp-match? #rx"^LicenseRef-[A-Za-z0-9.-]+$" (symbol->string id)))

  (define (id? v)
    (and (symbol? v) (not (memq v '(AND OR WITH)))))

  (define (well-formed? expr)
    (match expr
      [(list (? id?) 'WITH (? id?)) #t]
      [(list l (or 'AND 'OR) r) (and (well-formed? l) (well-formed? r))]
      [v (id? v)]))

  (define (unknown-ids expr)
    (match expr
      [(list licence 'WITH exception)
       (append (unknown-ids licence)
               (if (memq exception exceptions) '() (list exception)))]
      [(list l (or 'AND 'OR) r)
       (append (unknown-ids l) (unknown-ids r))]
      [id
       (if (or (memq id licenses) (license-ref? id)) '() (list id))]))

  (define package-dirs
    (cons torch-dir
          (for/list ([name (in-list '("torch-x86_64-linux" "torch-aarch64-macosx"))]
                     #:when (directory-exists? (build-path repo-dir "pkgs" name)))
            (build-path repo-dir "pkgs" name))))

  (test-case "every licence expression is in Racket's binary grammar"
    (for ([dir (in-list package-dirs)])
      (check-true (well-formed? ((get-info/full dir) 'license)) (format "~a" dir))))

  (test-case "every licence id is on the pinned SPDX list or a LicenseRef-"
    (for ([dir (in-list package-dirs)])
      (define expr ((get-info/full dir) 'license))
      (check-equal? (unknown-ids expr) '() (format "~a" dir))))

  (test-case "a malformed expression is rejected"
    (for ([expr (in-list '(()
                           (MIT Apache-2.0)
                           (AND)
                           (MIT AND)
                           (MIT AND Apache-2.0 OR Zlib)
                           ((MIT AND Zlib) WITH LLVM-exception)))])
      (check-false (well-formed? expr) (format "~s" expr))))

  (test-case "a nested binary expression is accepted"
    (check-true (well-formed? '(Apache-2.0 AND (CC-BY-4.0 AND LicenseRef-CC-BY-2.0-FR))))
    (check-true (well-formed? '((Apache-2.0 WITH LLVM-exception) OR MIT))))

  (test-case "an id the SPDX list lacks is reported"
    (check-equal? (unknown-ids '(Apache-2.0 AND CC-BY-2.0-FR)) '(CC-BY-2.0-FR))
    (check-equal? (unknown-ids '(MIT AND (Apache-2.0 WITH No-exception)))
                  '(No-exception))
    (check-equal? (unknown-ids '(MIT OR LicenseRef-Anything)) '()))

  (test-case "the package's LICENSE is the repository's"
    (define root-license (build-path repo-dir "LICENSE"))
    (when (file-exists? root-license)
      (check-equal? (file->bytes (build-path torch-dir "LICENSE"))
                    (file->bytes root-license)))))
