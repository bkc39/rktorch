#lang racket/base

(module+ test
  (require (for-syntax racket/base)
           (only-in json jsexpr->string)
           (only-in racket/file file->bytes)
           (only-in racket/list filter-map make-list)
           (only-in racket/match match-lambda)
           (only-in racket/runtime-path define-runtime-path)
           (only-in racket/string string-prefix?)
           rackunit
           (only-in "../nn.rkt" Linear load-state! mse-loss parameters save-state! sgd state-dict)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "private/checkpoint-kit.rkt"
                    call-with-live-run check-exn/contract fit-draws saving-callback Small stored)
           (only-in "private/comparable.rkt" comparable layer-bytes)
           (only-in "private/safetensors-reader.rkt" safetensors-metadata))

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
    (define-validation-step (x y) (mse-loss (net x) y))
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
          (check-exn/contract #rx"probe.*\"f\""
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

  (define (header-length raw) (integer-bytes->integer raw #f #f 0 8))

  (test-case "T3-L20: what is not a version-1 checkpoint raises before any trainer is built"
    (with-temporary-directory (dir #:template "refused-~a")
      (define good (build-path dir "good.ckpt"))
      (void (call-with-live-run (lambda (r) (save-checkpoint! r good))))
      (define raw (file->bytes good))
      (define (written name bs)
        (define p (build-path dir name))
        (call-with-output-file p (lambda (o) (write-bytes bs o)))
        p)
      (define (unreadable label path)
        (define name (let-values ([(_dir name _dir?) (split-path path)]) (path->string name)))
        (list label path (pregexp (string-append "not a rktorch checkpoint.*path: .*"
                                                 (regexp-quote name)))))
      (define state-file (build-path dir "state.safetensors"))
      (save-state! (Linear 2 1) state-file)
      (define cases
        (list (list "a save-state! file" state-file #rx"not a.*checkpoint")
              (list "version 2" (written "v2.ckpt" (version-2 raw))
                    #rx"version.*\"2\".*\"1\"|version.*2.*1")
              (unreadable "an empty file" (written "empty.ckpt" #""))
              (unreadable "a cut length prefix" (written "prefix.ckpt" (subbytes raw 0 4)))
              (unreadable "a cut header"
                          (written "header.ckpt"
                                   (subbytes raw 0 (+ 8 (quotient (header-length raw) 2)))))
              (unreadable "a cut payload"
                          (written "payload.ckpt" (subbytes raw 0 (sub1 (bytes-length raw)))))
              (unreadable "a Lightning zip checkpoint"
                          (written "lightning.ckpt" (bytes-append #"PK\3\4" (make-bytes 60 0))))
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
        (check-equal? (unbox built) 0 (format "~a: make-trainer was not called" label)))))

  (define current-header-limit (dynamic-require format-module 'current-header-limit))

  (define (header-bytes path) (header-length (file->bytes path)))

  (test-case "the 100 MB limit is on the safetensors header, checked before any file is made"
    (with-temporary-directory (dir #:template "limit-~a")
      (define-values (fits over) (values (build-path dir "fits.ckpt") (build-path dir "over.ckpt")))
      (void
       (call-with-live-run
        #:callbacks (list (stored (box (make-string 1000 #\"))))
        (lambda (r)
          (save-checkpoint! r fits)
          (define size (header-bytes fits))
          (define text (string-utf-8-length (hash-ref (safetensors-metadata fits) 'checkpoint)))
          (check-true (< text (sub1 size)) "the escaped header is longer than the text")
          (check-exn #rx"save-checkpoint!:.*100 MB header limit"
                     (lambda ()
                       (parameterize ([current-header-limit (sub1 size)])
                         (save-checkpoint! r over))))
          (check-equal? (directory-list dir) (list (string->path "fits.ckpt")))
          (parameterize ([current-header-limit size])
            (save-checkpoint! r over))
          (check-equal? (header-bytes over) size))))))

  (test-case "a shared string or a bignum is counted at its length before any text is built"
    (with-temporary-directory (dir #:template "estimate-~a")
      (define path (build-path dir "big.ckpt"))
      (define value (box #f))
      (define bigs
        (hash "a shared string" (make-list 1000 (make-string 10000 #\a))
              "a bignum" (expt 10 1000000)))
      (define allocated (make-hash))
      (void
       (call-with-live-run
        #:callbacks (list (stored value))
        (lambda (r)
          (for ([(label big) (in-hash bigs)])
            (set-box! value big)
            (collect-garbage)
            (define before (current-memory-use 'cumulative))
            (check-exn #rx"100 MB header limit"
                       (lambda ()
                         (parameterize ([current-header-limit 100000])
                           (save-checkpoint! r path)))
                       label)
            (hash-set! allocated label (- (current-memory-use 'cumulative) before))))))
      (check-equal? (for/list ([(label n) (in-hash allocated)] #:unless (< n 4000000)) label) '()
                    "refused without writing the text out")
      (check-equal? (directory-list dir) '())))

  (define (hand-built dir name text #:entries [entries (hasheq)] #:payload [payload #""])
    (define meta (hasheq 'format "rktorch-checkpoint" 'version "1" 'checkpoint text))
    (define header (string->bytes/utf-8 (jsexpr->string (hash-set entries '__metadata__ meta))))
    (written-file (build-path dir name)
                  (bytes-append (integer->integer-bytes (bytes-length header) 8 #f #f)
                                header payload)))

  (define (written-file path bs)
    (call-with-output-file path (lambda (o) (write-bytes bs o)))
    path)

  (define (f32-entry shape end)
    (hasheq '$0 (hasheq 'dtype "F32" 'shape shape 'data_offsets (list 0 end))))

  (test-case "every malformed file is not a rktorch checkpoint, named by the caller and the path"
    (with-temporary-directory (dir #:template "malformed-~a")
      (define one-float (f32-entry '(1) 4))
      (define cases
        (list (list "metadata not JSON" "{not json" one-float)
              (list "a $tensor naming a number" "{\"x\":{\"$tensor\":5}}" one-float)
              (list "a $symbol of a number" "{\"x\":{\"$symbol\":5}}" one-float)
              (list "entries not a list" "{\"x\":{\"$hash\":\"equal\",\"entries\":5}}" one-float)
              (list "an unsupported dtype tag" "{\"x\":{\"$tensor\":\"$0\"}}"
                    (hasheq '$0 (hasheq 'dtype "I32" 'shape '(1) 'data_offsets '(0 4))))
              (list "a shape the bytes do not fill" "{\"x\":{\"$tensor\":\"$0\"}}"
                    (f32-entry '(2) 4))
              (list "an eq? hash with a string key"
                    "{\"x\":{\"$hash\":\"eq\",\"entries\":[[\"s\",1]]}}" one-float)
              (list "a tensor key"
                    "{\"x\":{\"$hash\":\"equal\",\"entries\":[[{\"$tensor\":\"$0\"},1]]}}"
                    one-float)
              (list "a top-level key that is not a string"
                    "{\"$hash\":\"equal\",\"entries\":[[1,2]]}" one-float)
              (list "a JSON null" "{\"x\":null}" one-float)))
      (for ([c (in-list cases)] [i (in-naturals)])
        (define-values (label text entries) (apply values c))
        (define path
          (hand-built dir (format "m~a.ckpt" i) text #:entries entries #:payload (make-bytes 4 0)))
        (for ([who (in-list '(read-checkpoint load-from-checkpoint))])
          (define rx (pregexp (format "^~a: not a rktorch checkpoint.*path: .*m~a[.]ckpt" who i)))
          (check-exn/contract rx
                              (lambda () (if (eq? who 'read-checkpoint)
                                             (read-checkpoint path)
                                             (load-from-checkpoint Small path)))
                              (format "~a, ~a" label who)))))))

