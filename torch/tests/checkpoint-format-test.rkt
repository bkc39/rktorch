#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in racket/file file->bytes)
           (only-in racket/list filter-map)
           (only-in racket/match match-lambda)
           (only-in racket/runtime-path define-runtime-path)
           (only-in racket/string string-prefix?)
           rackunit
           (only-in "../nn.rkt" Linear load-state! mse-loss parameters save-state! sgd state-dict)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt"
                    call-with-live-run fit-draws saving-callback Small stored)
           (only-in "private/comparable.rkt" comparable layer-bytes))

  (define-runtime-path format-module "../train/private/format.rkt")

  (test-case "T3-L6: saving every epoch leaves the parameters and every random stream as they were"
    (with-temporary-directory (dir #:template "pure-~a")
      (define plain (fit-draws '()))
      (define (epoch-path r) (build-path dir (format "e~a.ckpt" (run-epoch r))))
      (define saved (fit-draws (list (saving-callback epoch-path))))
      (check-not-equal? (hash-ref plain 'cpu-rng) (hash-ref plain 'seeded-rng)
                        "the fit draws from the default stream")
      (check-equal? saved plain)
      (check-equal? (length (directory-list dir)) 3)))

  (test-case "T3-L6: two consecutive saves give byte-identical files"
    (with-temporary-directory (dir #:template "twice-~a")
      (define-values (a b) (values (build-path dir "a.ckpt") (build-path dir "b.ckpt")))
      (void (call-with-live-run (lambda (r) (save-checkpoint! r a) (save-checkpoint! r b))))
      (check-equal? (file->bytes a) (file->bytes b))))

  (define-trainer Watched (net lr seen)
    (define-init (s) (set! net (Linear 2 1)) (set! lr 0.1) (set! seen s))
    (define-hparams lr)
    (define-optimizers (sgd (parameters net) #:lr lr))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-hook (on-save-checkpoint ckpt) (set-box! seen (comparable ckpt #:hash-kinds? #t))))

  (test-case "T3-L7: load-state! reads the model, read-checkpoint the tree the hooks saw"
    (with-temporary-directory (dir #:template "format-~a")
      (define path (build-path dir "w.ckpt"))
      (define seen (box #f))
      (define tr (Watched seen))
      (void (call-with-live-run #:trainer tr (lambda (r) (save-checkpoint! r path))))
      (define trained
        (comparable (filter-map (match-lambda
                                  [(cons (regexp #rx"^net[.](.*)$" (list _ name)) t) (cons name t)]
                                  [_ #f])
                                (state-dict tr))))
      (define fresh (Linear 2 1))
      (check-not-equal? (layer-bytes fresh) trained "the fresh layer starts elsewhere")
      (load-state! fresh path #:rename (lambda (name)
                                         (and (string-prefix? name "state_dict.net.")
                                              (substring name (string-length "state_dict.net.")))))
      (check-equal? (layer-bytes fresh) trained "the checkpoint holds the trainer's parameters")
      (check-equal? (comparable (read-checkpoint path) #:hash-kinds? #t) (unbox seen))
      (for ([key '("epoch" "global_step" "state_dict" "loops" "callbacks" "optimizer_states"
                   "lr_schedulers" "trainer_hparams" "hyper_parameters" "init_formals"
                   "init_dropped" "callback_metrics" "rng")])
        (check-true (hash-has-key? (read-checkpoint path) key) key))))

  (test-case "T3-L9: a failed save leaves the earlier file and adds none"
    (with-temporary-directory (dir #:template "atomic-~a")
      (define path (build-path dir "a.ckpt"))
      (define current (box 1))
      (define current-write-fault (dynamic-require format-module 'current-write-fault))
      (void
       (call-with-live-run
        #:callbacks (list (stored current))
        (lambda (r)
          (save-checkpoint! r path)
          (define before (file->bytes path))
          (define files (directory-list dir))
          (set-box! current (list 1 (hash "f" (lambda () 1))))
          (check-exn (lambda (e) (and (exn:fail:contract? e)
                                      (regexp-match? #rx"probe" (exn-message e))
                                      (regexp-match? #rx"\"f\"" (exn-message e))))
                     (lambda () (save-checkpoint! r path))
                     "a procedure in the tree names its key path")
          (check-equal? (file->bytes path) before)
          (check-equal? (directory-list dir) files)
          (set-box! current 2)
          (check-exn #rx"planned write fault"
                     (lambda ()
                       (parameterize ([current-write-fault
                                       (lambda () (error 'write "planned write fault"))])
                         (save-checkpoint! r path))))
          (check-equal? (file->bytes path) before "after a write fault")
          (check-equal? (directory-list dir) files "after a write fault"))))))

  (define (version-2 bs)
    (regexp-replace #rx#"(\"version\" *: *)\"1\"" bs #"\\1\"2\""))

  (test-case "T3-L20: what is not a version-1 checkpoint raises before any trainer is built"
    (with-temporary-directory (dir #:template "refused-~a")
      (define good (build-path dir "good.ckpt"))
      (void (call-with-live-run (lambda (r) (save-checkpoint! r good))))
      (define raw (file->bytes good))
      (define (written name bs)
        (define p (build-path dir name))
        (call-with-output-file p (lambda (o) (write-bytes bs o)))
        p)
      (define state-file (build-path dir "state.safetensors"))
      (save-state! (Linear 2 1) state-file)
      (define cases
        (list (list "a save-state! file" state-file #rx"not a.*checkpoint")
              (list "version 2" (written "v2.ckpt" (version-2 raw))
                    #rx"version.*\"2\".*\"1\"|version.*2.*1")
              (list "a truncated file"
                    (written "cut.ckpt" (subbytes raw 0 (quotient (bytes-length raw) 2)))
                    #rx".")
              (list "a missing file" (build-path dir "absent.ckpt") #rx"absent.ckpt")))
      (for ([c (in-list cases)])
        (define-values (label path rx) (apply values c))
        (check-exn rx
                   (lambda () (read-checkpoint path))
                   label)
        (define built (box 0))
        (check-exn exn:fail?
                   (lambda () (load-from-checkpoint (lambda () (set-box! built 1) (Small)) path))
                   label)
        (check-equal? (unbox built) 0 (format "~a: make-trainer was not called" label))))))
