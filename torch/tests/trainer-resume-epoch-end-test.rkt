#lang racket/base

(module+ test
  (require (only-in racket/match match-define)
           (only-in racket/port open-output-nowhere)
           rackunit
           (only-in "../data/loader.rkt" dataloader tensor-dataset)
           (only-in "../main.rkt" make-generator manual-seed! randn)
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/trainer-kit.rkt" call-with-logged-warnings))

  (define (epoch-path dir r [prefix "epoch"])
    (build-path dir (format "~a=~a.ckpt" prefix (run-epoch r))))

  (define-trainer Plain (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y)))

  (define-trainer SelfSaving (net dir)
    (define-init (d) (set! net (Linear 2 1)) (set! dir d))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-train-epoch-end)
      (log! 'tag (exact->inexact (run-epoch run)))
      (save-checkpoint! run (epoch-path dir run))))

  (define-callback checkpoint-saver (dir prefix)
    #:kind checkpoint
    (define-init (d p) (set! dir d) (set! prefix p))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run prefix))))

  (define-callback checkpoint-tagger ()
    #:kind checkpoint
    (define-hook (on-train-epoch-end)
      (log! 'tag (exact->inexact (run-epoch run)))))

  (define-callback monitor-saver (dir)
    #:kind monitor
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run))))

  (define-callback monitor-tagger ()
    #:kind monitor
    (define-hook (on-train-epoch-end)
      (log! 'tag (exact->inexact (run-epoch run)))))

  (define-callback logging-saver (dir)
    #:kind checkpoint
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (log! 'tag (exact->inexact (run-epoch run)))
      (save-checkpoint! run (epoch-path dir run "late"))))

  (define-callback save-log-saver (dir)
    #:kind checkpoint
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run))
      (log! 'tag (exact->inexact (run-epoch run)))
      (save-checkpoint! run (epoch-path dir run "late"))))

  (define-callback weights-saver (dir)
    #:kind checkpoint
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run "weights") #:weights-only? #t)))

  (define-callback save-then-log (dir)
    #:kind checkpoint
    (define-init (d) (set! dir d))
    (define-hook (on-train-epoch-end)
      (save-checkpoint! run (epoch-path dir run))
      (log! 'tag (exact->inexact (run-epoch run)))))

  (struct legs (straight split saving resuming))

  (define (go dir trainer-of callbacks-of max-epochs ckpt)
    (manual-seed! 0)
    (define tags (box '()))
    (define g (make-generator 1))
    (define data (tensor-dataset (randn 4 2 #:generator g) (randn 4 1 #:generator g)))
    (define reader
      (lambda-callback #:on-train-epoch-start
                       (lambda (r) (set-box! tags (cons (run-metric r 'tag) (unbox tags))))))
    (define-values (_r warnings)
      (call-with-logged-warnings
       (lambda ()
         (parameterize ([current-error-port (open-output-nowhere)])
           (fit (trainer-of dir) (dataloader data #:batch-size 2) #:max-epochs max-epochs
                #:logger #f #:ckpt-path ckpt #:callbacks (cons reader (callbacks-of dir)))))))
    (list (reverse (unbox tags)) warnings))

  (define (resumed trainer-of callbacks-of)
    (with-temporary-directory (root #:template "epoch-end-~a")
      (define (leg-dir name) (define d (build-path root name)) (make-directory d) d)
      (define split (leg-dir "split"))
      (match-define (list straight _) (go (leg-dir "straight") trainer-of callbacks-of 4 #f))
      (match-define (list first saving) (go split trainer-of callbacks-of 2 #f))
      (match-define (list later resuming)
        (go split trainer-of callbacks-of 4 (build-path split "epoch=1.ckpt")))
      (legs straight (append first later) saving resuming)))

  (define (plain _dir) (Plain))

  (define exact-tags '(#f 0.0 1.0 2.0))

  (test-case "a save followed in on-train-epoch-end by a callback that logs warns while saving"
    (define kinds
      (list (cons 'checkpoint (lambda (d) (list (checkpoint-saver d "epoch") (checkpoint-tagger))))
            (cons 'monitor (lambda (d) (list (monitor-saver d) (monitor-tagger))))))
    (for ([kind-callbacks (in-list kinds)])
      (match-define (cons kind callbacks-of) kind-callbacks)
      (match-define (legs straight split saving resuming) (resumed plain callbacks-of))
      (check-equal? straight exact-tags)
      (check-equal? split '(#f 0.0 0.0 2.0) (format "~a: the resumed epoch reads a stale tag" kind))
      (check-equal? (length saving) 1 (format "~a: one warning in the saving run" kind))
      (check-regexp-match
       #rx"-tagger[0-9]* callback logged in on-train-epoch-end after a checkpoint was saved"
       (car saving))
      (check-regexp-match #rx"not exact" (car saving))
      (check-equal? resuming saving
                    (format "~a: the resumed run warns for its saves, not for the resume" kind))))

  (test-case "a later save in the same on-train-epoch-end does not hide the first save's warning"
    (for ([callbacks-of
           (in-list (list (lambda (d) (list (checkpoint-saver d "epoch") (logging-saver d)))
                          (lambda (d) (list (save-log-saver d)))))])
      (match-define (legs straight split saving resuming) (resumed plain callbacks-of))
      (check-equal? straight exact-tags)
      (check-equal? split '(#f 0.0 0.0 2.0) "the resumed epoch reads a stale tag")
      (check-equal? (length saving) 1 "one warning in the saving run")
      (check-regexp-match #rx"saver[0-9]* callback logged in on-train-epoch-end after"
                          (car saving))))

  (test-case "a weights-only save followed by a callback that logs does not warn"
    (with-temporary-directory (dir #:template "weights-~a")
      (match-define (list _ warnings)
        (go dir plain (lambda (d) (list (weights-saver d) (checkpoint-tagger))) 2 #f))
      (check-equal? warnings '())))

  (test-case "a callback that logs after its own save is told to log before saving"
    (match-define (legs _ split saving _) (resumed plain (lambda (d) (list (save-then-log d)))))
    (check-equal? split '(#f 0.0 0.0 2.0) "the resumed epoch reads a stale tag")
    (check-equal? (length saving) 1 "one warning in the saving run")
    (check-regexp-match #rx"save-then-log[0-9]* callback logged in on-train-epoch-end after"
                        (car saving))
    (check-regexp-match #rx"log before saving" (car saving))
    (check-false (regexp-match? #rx"list the saving callback last" (car saving))))

  (test-case "a saver listed after every callback that logs resumes exactly, without a warning"
    (match-define (legs straight split saving resuming)
      (resumed plain (lambda (d) (list (checkpoint-tagger) (checkpoint-saver d "epoch")))))
    (check-equal? split straight)
    (check-equal? (append saving resuming) '()))

  (test-case "two savers that log nothing do not warn"
    (match-define (legs straight split saving resuming)
      (resumed plain (lambda (d) (list (checkpoint-saver d "other") (checkpoint-saver d "epoch")))))
    (check-equal? split straight)
    (check-equal? (append saving resuming) '()))

  (test-case "a save from the trainer's own on-train-epoch-end after its logging resumes exactly"
    (match-define (legs straight split saving resuming)
      (resumed (lambda (d) (SelfSaving d)) (lambda (_d) '())))
    (check-equal? straight exact-tags)
    (check-equal? split straight)
    (check-equal? saving '())
    (check-equal? (length resuming) 1)
    (check-regexp-match #rx"saved before or during the trainer's on-train-epoch-end" (car resuming))
    (check-regexp-match #rx"values logged there after the save are not in the file"
                        (car resuming))))
