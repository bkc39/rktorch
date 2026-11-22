#lang racket/base

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/path file-name-from-path)
           (only-in racket/port open-output-nowhere)
           (only-in racket/match match-define)
           (only-in racket/runtime-path #%datum define-runtime-path)
           rackunit
           (only-in "../nn.rkt" Linear mse-loss parameters sgd)
           (only-in "../private/util.rkt" with-temporary-directory)
           "../train.rkt"
           (only-in "../train/private/types.rkt" callback-hook callback-state-key)
           (only-in "private/checkpoint-kit.rkt" fit-draws Small)
           (only-in "private/trainer-kit.rkt" apply-keywords call-with-logged-warnings toy-loader))

  (define-runtime-path fixture-path "lightning-checkpoint.rktd")

  (define scores '(0.5 0.3 0.4 +nan.0 0.2 0.6))

  (define-trainer Scripted (net)
    (define-init () (set! net (Linear 2 1)))
    (define-optimizers (sgd (parameters net) #:lr 0.1))
    (define-training-step (x y) (mse-loss (net x) y))
    (define-validation-step (_x _y) (log! 'val-loss (list-ref scores (run-epoch run)))))

  (define (name-of p) (and p (not (equal? p "")) (path->string (file-name-from-path p))))

  (define (listing dir)
    (for/hash ([f (in-list (directory-list dir))])
      (values (path->string f) (file-or-directory-identity (build-path dir f)))))

  (define (changes before after)
    (define saved (sort (for/list ([(f id) (in-hash after)]
                                   #:unless (equal? (hash-ref before f #f) id))
                          f)
                        string<?))
    (append (for/list ([f (in-list saved)] #:unless (equal? f "last.ckpt")) (list 'save f))
            (for/list ([f (in-list (sort (hash-keys before) string<?))]
                       #:unless (hash-has-key? after f))
              (list 'delete f))
            (if (member "last.ckpt" saved) '((save "last.ckpt")) '())))

  (define (watching dir events)
    (define seen (box #f))
    (define (look! _r)
      (define now (listing dir))
      (when (unbox seen) (set-box! events (append (unbox events) (changes (unbox seen) now))))
      (set-box! seen now))
    (lambda-callback #:on-train-start look! #:on-train-epoch-start look! #:on-train-end look!))

  (define (replay root name kws #:existing [existing '()] #:accumulate [accumulate 1])
    (define dir (build-path root name))
    (make-directory dir)
    (for ([f (in-list existing)]) (call-with-output-file (build-path dir f) void))
    (define mc (apply-keywords model-checkpoint (cons (cons '#:dirpath dir) kws)))
    (define events (box '()))
    (void (fit (Scripted) (toy-loader 6) #:val (toy-loader 2) #:max-epochs (length scores)
               #:num-sanity-val-steps 0 #:accumulate-grad-batches accumulate #:logger #f
               #:callbacks (list (watching dir events) mc)))
    (define state ((callback-hook mc 'state-dict) mc #f))
    (append (unbox events)
            (list (cons 'best-k (sort (for/list ([(p v) (in-hash (model-checkpoint-best-k-models mc))])
                                        (cons (name-of p) v))
                                      string<? #:key car))
                  (list 'best (name-of (model-checkpoint-best-model-path mc))
                        (model-checkpoint-best-model-score mc))
                  (list 'kth (name-of (hash-ref state "kth_best_model_path" #f))
                        (hash-ref state "kth_value"))
                  (cons 'last (name-of (model-checkpoint-last-model-path mc)))
                  (cons 'state-key (callback-state-key mc)))))

  (define cases
    `((no-monitor () ())
      (min-top-2 ((#:monitor . val-loss) (#:save-top-k . 2) (#:filename . "{epoch}-{val-loss:.2f}"))
                 ())
      (max-top-all ((#:monitor . val-loss) (#:save-top-k . -1) (#:mode . max)) ())
      (save-last ((#:monitor . val-loss) (#:save-last . #t)) ())
      (pre-existing () ((#:existing . ("epoch=0-step=3.ckpt"))))
      (unformatted ((#:monitor . val-loss) (#:filename . "{epoch}-{val-loss}")) ())
      (every-2-steps ((#:every-n-train-steps . 2) (#:save-top-k . -1)) ((#:accumulate . 2)))))

  (test-case "T3-L25: model-checkpoint replays Lightning's ModelCheckpoint on a scripted val-loss"
    (define fixture (hash-ref (file->value fixture-path) 'cases))
    (with-temporary-directory (root #:template "replay-~a")
      (for ([c (in-list cases)])
        (match-define (list name kws replay-kws) c)
        (check-equal? (apply-keywords replay replay-kws root (symbol->string name) kws)
                      (hash-ref fixture name)
                      (symbol->string name)))))

  (define (fit-small tr mc #:epochs [epochs 2] #:ckpt-path [path #f] #:val? [val? #t]
                     #:more [more '()])
    (apply-keywords fit
                    (append (list (cons '#:max-epochs epochs) (cons '#:logger #f)
                                  (cons '#:ckpt-path path) (list '#:callbacks mc))
                            (if val? (list (cons '#:val (toy-loader 2))) '())
                            more)
                    tr (toy-loader 4)))

  (define (monitoring dir)
    (model-checkpoint #:dirpath dir #:monitor 'val-loss #:save-top-k 2 #:save-last #t))

  (define (restorable mc)
    (define state ((callback-hook mc 'state-dict) mc #f))
    (list (model-checkpoint-best-model-path mc) (model-checkpoint-best-model-score mc)
          (model-checkpoint-best-k-models mc) (model-checkpoint-last-model-path mc)
          (hash-ref state "kth_best_model_path" #f) (hash-ref state "kth_value" #f)))

  (define (warnings-of thunk)
    (define-values (_result warnings)
      (call-with-logged-warnings
       (lambda () (parameterize ([current-error-port (open-output-nowhere)]) (thunk)))))
    warnings)

  (test-case "T3-L26: the same dirpath restores fully, another restores only the best path"
    (with-temporary-directory (root #:template "mc-restore-~a")
      (define dir (build-path root "a"))
      (define mc (monitoring dir))
      (void (fit-small (Small) mc))
      (define path (model-checkpoint-last-model-path mc))
      (define same (monitoring dir))
      (void (fit-small (Small) same #:ckpt-path path))
      (check-equal? (restorable same) (restorable mc))
      (define other (monitoring (build-path root "b")))
      (define warnings (warnings-of (lambda () (fit-small (Small) other #:ckpt-path path))))
      (check-equal? (model-checkpoint-best-model-path other) (model-checkpoint-best-model-path mc))
      (check-equal? (model-checkpoint-best-k-models other) (hash))
      (check-true (for/or ([w (in-list warnings)]) (regexp-match? #rx"dirpath" w)))))

  (define (saved-at dir hook-epochs #:val? [val? #t] #:more [more '()])
    (define seen (box '()))
    (define ((note! where) r)
      (set-box! seen (cons (list where (run-epoch r) (length (directory-list dir))) (unbox seen))))
    (define mc (model-checkpoint #:dirpath dir #:save-top-k -1))
    (make-directory dir)
    (apply-keywords fit
                    (append (list (cons '#:max-epochs hook-epochs) (cons '#:logger #f)
                                  (list '#:callbacks
                                        (lambda-callback #:on-train-epoch-end (note! 'epoch-end)
                                                         #:on-train-epoch-start (note! 'epoch-start))
                                        mc))
                            (if val? (list (cons '#:val (toy-loader 2))) '())
                            more)
                    (Small) (toy-loader 4))
    (reverse (unbox seen)))

  (test-case "T3-L27: where model-checkpoint saves"
    (with-temporary-directory (root #:template "mc-points-~a")
      (check-equal? (saved-at (build-path root "every") 2)
                    '((epoch-start 0 0) (epoch-end 0 0) (epoch-start 1 1) (epoch-end 1 1))
                    "validation every epoch: at on-train-epoch-end")
      (check-equal? (saved-at (build-path root "second") 2
                              #:more '((#:check-val-every-n-epoch . 2)))
                    '((epoch-start 0 0) (epoch-end 0 0) (epoch-start 1 0) (epoch-end 1 1))
                    "validation every second epoch: at on-validation-end")
      (check-equal? (saved-at (build-path root "none") 2 #:val? #f)
                    '((epoch-start 0 0) (epoch-end 0 0) (epoch-start 1 1) (epoch-end 1 1))
                    "no validation: at the epoch end")))

  (test-case "T3-L27: one save per global step, none under #:fast-dev-run"
    (with-temporary-directory (root #:template "mc-once-~a")
      (define dir (build-path root "once"))
      (define mc (model-checkpoint #:dirpath dir #:save-top-k -1 #:save-on-train-epoch-end #f))
      (void (fit (Small) (toy-loader 8) #:val (toy-loader 2) #:max-epochs 1 #:logger #f
                 #:val-check-interval 1 #:accumulate-grad-batches 2 #:callbacks (list mc)))
      (check-equal? (length (directory-list dir)) 2 "4 validations, 2 global steps")
      (check-false (for/or ([f (in-list (directory-list dir))])
                     (regexp-match? #rx"-v[0-9]" (path->string f))))
      (define fast (build-path root "fast"))
      (void (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 2 #:logger #f
                 #:fast-dev-run 2 #:callbacks (list (model-checkpoint #:dirpath fast))))
      (check-true (or (not (directory-exists? fast)) (null? (directory-list fast))))))

  (test-case "T3-L6: model-checkpoint leaves the parameters and every random stream as they were"
    (with-temporary-directory (root #:template "mc-pure-~a")
      (define plain (fit-draws '()))
      (define saved (fit-draws (list (model-checkpoint #:dirpath root #:save-top-k -1))))
      (check-not-equal? (hash-ref plain 'cpu-rng) (hash-ref plain 'seeded-rng)
                        "the fit draws from the default stream")
      (check-equal? saved plain)
      (check-equal? (length (filter (lambda (f) (regexp-match? #rx"[.]ckpt$" (path->string f)))
                                    (directory-list root)))
                    3)))

  (test-case "model-checkpoint: a missing #:monitor key warns while no validation has run, then raises"
    (with-temporary-directory (root #:template "mc-absent-~a")
      (define (absent-monitor name) (model-checkpoint #:dirpath (build-path root name) #:monitor 'absent))
      (define-values (_run warnings)
        (call-with-logged-warnings
         (lambda ()
           (parameterize ([current-error-port (open-output-nowhere)])
             (fit (Small) (toy-loader 4) #:max-epochs 2 #:logger #f
                  #:callbacks (list (absent-monitor "no-val")))))))
      (check-true (for/or ([w (in-list warnings)]) (regexp-match? #rx"absent" w))
                  "without a validation loop it warns")
      (check-exn (lambda (e) (and (exn:fail? e) (regexp-match? #rx"absent" (exn-message e))))
                 (lambda ()
                   (fit (Small) (toy-loader 4) #:val (toy-loader 2) #:max-epochs 2 #:logger #f
                        #:callbacks (list (absent-monitor "val"))))
                 "after a validation loop it raises, naming the key"))))
