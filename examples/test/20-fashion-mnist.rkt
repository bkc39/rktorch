#lang racket/base

(module+ main
  (require (only-in torch accelerator-if-available)
           (only-in "private/env.rkt" env-number)
           (only-in "private/fashion-mnist-kit.rkt" print-epoch-rows run-headline))
  (define device (accelerator-if-available))
  (printf "device: ~a\n" device)
  (define-values (rows _run _total) (run-headline #:epochs (env-number "EPOCHS" 5) #:device device))
  (print-epoch-rows rows))

(module+ test
  (require (only-in racket/file file->value)
           (only-in racket/list append* append-map count fifth filter-map filter-not first fourth
                    last second third)
           (only-in racket/match match match-define match-lambda)
           (only-in racket/runtime-path #%datum define-runtime-path)
           (only-in racket/system system*)
           rackunit
           torch
           torch/data/loader
           torch/nn
           torch/scribblings/examples/20-fashion-mnist
           (only-in torch/private/util with-temporary-directory)
           (only-in torch/tests/private/python-env
                    call-with-python-env python python-cuda-available? python-module-available?
                    python-torch-available? tol)
           (only-in torch/train run-global-step run-grad-scaler run-optimizers)
           (only-in torch/vision/fashion-mnist load-fashion-mnist-fixture)
           (only-in (submod torch/vision/fashion-mnist private) fashion-mnist-size)
           (only-in "private/fashion-mnist-kit.rkt"
                    epoch-recorder fixture-batch-size full-batch-size print-epoch-rows
                    run-example twin-config twin-json)
           (only-in "private/lightning-trace.rkt"
                    compare-golden compare-golden-file compare-trace flat-state golden-gaps
                    lightning-name state-summary trace-callback trace-gaps trace-golden trace-labels
                    trace-logger trace-losses trace-lrs trace-norms trace-recorder trace-rows)
           (only-in "../../scripts/fashion-mnist-results.rkt" fashion-mnist-record)
           (only-in racket/port with-output-to-string))

  (define-runtime-path bundle-path "../../torch/scribblings/results/fashion-mnist.rktd")
  (define-runtime-path golden-path "20-fashion-mnist-lightning.rktd")
  (define-runtime-path twin-path "../python/20_fashion_mnist.py")

  (match-define (hash-table ('seed seed) ('lr lr) ('weight-decay weight-decay)
                            ('accumulate accumulate) ('clip clip-val))
    fashion-recipe)

  (define epochs 2)
  (define total-steps 16)
  (define warmup 4)

  (define (named-state classifier)
    (append (named-parameters classifier) (named-buffers classifier)))

  (define (state-of classifier)
    (flat-state (named-state classifier)))

  (define (traced device precision)
    (define tr (trace-recorder))
    (define-values (recorder rows) (epoch-recorder))
    (define-values (r classifier stepping)
      (run-example #:device device #:precision precision #:epochs epochs
                   #:callbacks (list (trace-callback tr) recorder)
                   #:logger (trace-logger tr)))
    (values tr r classifier stepping (rows)))

  (define (hand-loop precision)
    (manual-seed! seed)
    (define net (fashion-net))
    (define-values (xs ys) (load-fashion-mnist-fixture 'train))
    (define-values (vx vy) (load-fashion-mnist-fixture 'test))
    (define loader
      (dataloader (tensor-dataset xs ys) #:batch-size fixture-batch-size
                  #:shuffle? #t #:generator (make-generator seed)))
    (define val (dataloader (tensor-dataset vx vy) #:batch-size 64))
    (define opt (adamw (weight-decay-param-groups net #:weight-decay weight-decay) #:lr lr))
    (define schedule
      (sequential-lr opt
                     (list (linear-lr opt #:start-factor 0.1 #:total-iters warmup)
                           (cosine-annealing-lr opt #:t-max (- total-steps warmup)))
                     #:milestones (list warmup)))
    (define params (append-map param-group-params (optimizer-param-groups opt)))
    (define (loss-of xb yb)
      (match precision
        ['bf16-mixed
         (with-autocast #:device 'cpu #:dtype 'bfloat16 #:cache-enabled? #f
           (cross-entropy (net xb) yb))]
        [_ (cross-entropy (net xb) yb)]))
    (define (validate! limit)
      (eval! net)
      (with-no-grad
        (for ([_ (in-range limit)] [(xb yb) (in-dataloader val)])
          (loss-of xb yb)))
      (train! net))
    (define (step-batch! i xb yb)
      (define loss (/ (loss-of xb yb) accumulate))
      (when (even? i) (zero-grads! opt))
      (backward! loss)
      (define rates
        (and (odd? i) (list (learning-rate opt #:group 0) (learning-rate opt #:group 1))))
      (when rates
        (clip-grad-norm! params clip-val)
        (step! opt)
        (step! schedule))
      (cons (item loss) rates))
    (define (train-epoch!)
      (begin0
        (for/list ([i (in-range (dataloader-length loader))] [(xb yb) (in-dataloader loader)])
          (step-batch! i xb yb))
        (validate! 64)))
    (validate! 2)
    (define steps (append* (for/list ([_ (in-range epochs)]) (train-epoch!))))
    (values (flat-state (append (named-parameters net) (named-buffers net)))
            (map car steps) (filter-map cdr steps)))

  (define-values (trace-a run-a classifier-a stepping-a rows-a) (traced (cpu-device) '32-true))
  (define-values (trace-b run-b classifier-b _stepping-b _rows-b)
    (traced (cpu-device) 'bf16-mixed))

  (define (without-prefix state)
    (map (match-lambda [(cons name v) (cons (regexp-replace #rx"^net[.]" name "") v)]) state))

  (test-case "G4-L7: fit runs the loop a hand-written one runs, in (a) and (b)"
    (for ([config (in-list (list (list '32-true trace-a classifier-a)
                                 (list 'bf16-mixed trace-b classifier-b)))])
      (match-define (list precision tr classifier) config)
      (define-values (state losses lrs) (hand-loop precision))
      (check-equal? (without-prefix (state-of classifier)) state (format "~a: state" precision))
      (check-equal? (trace-losses tr) losses (format "~a: losses" precision))
      (check-equal? (trace-lrs tr) lrs (format "~a: lrs" precision))))

  (test-case "G4-L8: the network's parameters and its two groups"
    (define net (fashion-net))
    (check-equal? (map (match-lambda [(cons name t) (cons name (shape t))]) (named-parameters net))
                  '(("0.conv.weight" 16 1 3 3) ("0.conv.bias" 16)
                    ("0.bn.weight" 16) ("0.bn.bias" 16)
                    ("1.conv.weight" 32 16 3 3) ("1.conv.bias" 32)
                    ("1.bn.weight" 32) ("1.bn.bias" 32)
                    ("3.weight" 128 1568) ("3.bias" 128)
                    ("5.weight" 10 128) ("5.bias" 10)))
    (check-equal? (for/sum ([p (in-list (parameters net))]) (numel p)) 207018)
    (match-define (list decay no-decay) (weight-decay-param-groups net #:weight-decay weight-decay))
    (define (sizes g)
      (define ts (param-group-params g))
      (list (length ts) (for/sum ([t (in-list ts)]) (numel t))))
    (check-equal? (sizes decay) '(4 206736))
    (check-equal? (sizes no-decay) '(8 282))
    (check-equal? (hash-ref (param-group-hyperparameters no-decay) 'weight-decay) 0.0))

  (test-case "G4-L9: sixteen optimizer steps, four of warmup, over 32 batches"
    (check-equal? stepping-a total-steps)
    (check-equal? (warmup-steps stepping-a) warmup)
    (check-equal? (run-global-step run-a) total-steps)
    (check-equal? (length (trace-labels trace-a)) 32)
    (define lrs (trace-lrs trace-a))
    (check-equal? (length lrs) total-steps)
    (for ([pair (in-list lrs)])
      (match-define (list decayed undecayed) pair)
      (check-equal? decayed undecayed "the groups' rates"))
    (check-equal? (first (list-ref lrs 0)) (* lr 0.1))
    (check-equal? (first (list-ref lrs warmup)) lr))

  (test-case "G4-L10: the clip is live in (a)"
    (check-true (> (apply max (trace-norms trace-a)) clip-val)))

  (test-case "the runner's test acc line is the fraction docs/epoch-check.md reads"
    (define out (with-output-to-string (lambda () (print-epoch-rows '((1 0.5 0.4 0.8993 1.0))))))
    (check-equal? (cadr (regexp-match #rx"test acc ([0-9.]+)" out)) "0.8993"))

  (test-case "the epoch rows are the logged epoch reductions"
    (define (logged key)
      (for*/list ([entry (in-list (trace-rows trace-a))]
                  [v (in-value (hash-ref (second entry) key #f))]
                  #:when v)
        v))
    (check-equal? (map first rows-a) '(1 2))
    (check-equal? (map second rows-a) (logged 'train/loss-epoch))
    (check-equal? (map third rows-a) (logged 'val/loss))
    (check-equal? (map fourth rows-a) (logged 'val/acc))
    (check-true (andmap (lambda (row) (positive? (fifth row))) rows-a)))

  (test-case "G4-L20: only (c) has a scaler"
    (check-false (run-grad-scaler run-a))
    (check-false (run-grad-scaler run-b)))

  (define synthetic-rows '((1 0.9 0.8 0.7 10.0) (2 0.5 0.45 0.84 9.5)))

  (define synthetic-versions
    (hasheq 'python "3.12.0" 'torch "2.14.0" 'lightning "2.6.5" 'cuda "none"))

  (define (atoms? v)
    (match v
      [(? hash?) (for/and ([x (in-hash-values v)]) (atoms? x))]
      [_ (or (string? v) (real? v))]))

  (test-case "G4-L21: the record is strings and reals around the two sides' rows"
    (for ([dev (in-list (list (cpu-device) (cuda-device)))])
      (define record
        (fashion-mnist-record synthetic-rows (reverse synthetic-rows) dev
                              #:config (twin-config #:batch full-batch-size #:epochs 2
                                                    #:total 470 #:warmup 118)
                              #:versions synthetic-versions
                              #:commit "0123abcd" #:seconds '(20.0 30.0)))
      (for ([(key value) (in-hash record)] #:unless (memq key '(racket lightning)))
        (check-true (atoms? value) (format "~a: ~s" key value)))
      (for ([side (in-list '(racket lightning))])
        (define s (hash-ref record side))
        (check-equal? (sort (hash-keys s) symbol<?) '(accuracy epochs seconds))
        (check-equal? (hash-ref s 'accuracy) (fourth (last (hash-ref s 'epochs)))))
      (check-equal? (hash-ref (hash-ref record 'racket) 'accuracy) 0.84)
      (check-equal? (hash-ref (hash-ref record 'lightning) 'seconds) 30.0)
      (check-equal? (hash-ref record 'seed) seed)
      (check-equal? (hash-ref record 'python-torch) "2.14.0")
      (match-define (hash-table ('total total) ('warmup w)) (hash-ref record 'config))
      (check-equal? (list total w) '(470 118))))

  (test-case "G4-L22: the committed headline, both sides within 0.01"
    (define bundle (file->value bundle-path))
    (define ours (hash-ref bundle 'racket))
    (define theirs (hash-ref bundle 'lightning))
    (define n (length (hash-ref ours 'epochs)))
    (define total
      (~> (fashion-mnist-size 'train) (/ full-batch-size) ceiling (/ accumulate) ceiling (* n)))
    (check-equal? (length (hash-ref theirs 'epochs)) n)
    (check-equal? (hash-ref bundle 'seed) seed)
    (check-equal? (hash-ref bundle 'precision) "16-mixed")
    (check-equal? (hash-ref bundle 'config)
                  (twin-config #:batch full-batch-size #:epochs n #:total total
                               #:warmup (warmup-steps total)))
    (check-true (string? (hash-ref bundle 'python-torch)))
    (check-true (<= (abs (- (hash-ref ours 'accuracy) (hash-ref theirs 'accuracy))) 0.01))
    (check-true (>= (hash-ref ours 'accuracy) 0.88)))

  (define (initial-named)
    (manual-seed! seed)
    (named-state (FashionClassifier)))

  (define (initial-state)
    (flat-state (initial-named)))

  (define (check-names r classifier json)
    (define names (hash-ref json 'names))
    (define (named-of xs) (map (match-lambda [(cons name _) (lightning-name name)]) xs))
    (check-equal? (named-of (named-parameters classifier)) (hash-ref names 'parameters))
    (check-equal? (named-of (named-buffers classifier)) (hash-ref names 'buffers))
    (define by-tensor
      (for/hasheq ([np (in-list (named-parameters classifier))])
        (match-define (cons name t) np)
        (values t name)))
    (match-define (list decay no-decay)
      (for/list ([g (in-list (~> r run-optimizers car optimizer-param-groups))])
        (map (lambda (t) (hash-ref by-tensor t)) (param-group-params g))))
    (check-equal? (map lightning-name decay) (hash-ref names 'decay))
    (check-equal? (map lightning-name no-decay) (hash-ref names 'no_decay)))

  (define (check-twin label device precision tr r classifier tolerance acc-tolerance)
    (define json (twin-json device precision 'fixture epochs))
    (test-case (format "G4-L11: ~a: the twin's config is the chapter's" label)
      (check-equal? (hash-ref json 'config)
                    (twin-config #:batch fixture-batch-size #:epochs epochs
                                 #:total total-steps #:warmup warmup)))
    (test-case (format "G4-L12: ~a: names and groups" label)
      (check-names r classifier json))
    (test-case (format "G4-L13: ~a: the initial state is the twin's, bit for bit" label)
      (for ([entry (in-list (initial-state))])
        (match-define (cons name vals) entry)
        (check-equal? vals (hash-ref (hash-ref json 'init) (string->symbol name)) name)))
    (printf "[20-fashion-mnist] ~a largest gaps: ~s\n" label
            (trace-gaps tr json (state-of classifier)))
    (test-case (format "G4-L14..L19: ~a: the trace is the twin's" label)
      (compare-trace label tr json (state-of classifier) tolerance
                     #:acc-tolerance acc-tolerance))
    (test-case (format "G4-L20: ~a: scales" label)
      (check-equal? (count values (hash-ref json 'scales))
                    (if (run-grad-scaler r) total-steps 0))))

  (define golden (file->value golden-path))
  (define initial-summary (state-summary (initial-named)))
  (define adam-drift (* 2 total-steps lr))

  (define golden-evidence
    `((x86_64
       "i9-9900K, OMP_NUM_THREADS 1 4 8, ATen default, oneDNN SSE41, MKL SSE4_2 and AVX"
       (32-true (losses 2.4e-7 ,tol) (norms 1.4e-6 ,tol) (rows 7.9e-4 3e-3) (acc 0 1/128)
                (values 1.0e-2 ,adam-drift) (sum 6.3e-3 0.03) (sum-sq 2.4e-2 0.1))
       (bf16-mixed (losses 7.9e-3 0.04) (norms 0.12 0.5) (rows 1.4e-2 0.06) (acc 1/128 1/32)
                   (values 1.6e-2 ,adam-drift) (sum 3.6e-2 0.15) (sum-sq 6.1e-2 0.25)))
      (cross-arch
       "aarch64-darwin, macos-latest, Nix checks run 43994518225"
       (32-true (losses 5.17e-4 2e-3) (norms 8.78e-3 0.035) (rows 9.47e-4 4e-3) (acc 0 1/128)
                (values 1.09e-2 ,adam-drift) (sum 6.2e-3 0.03) (sum-sq 2.1e-2 0.1))
       (bf16-mixed (losses 1.01e-2 0.04) (norms 0.103 0.5) (rows 1.31e-2 0.06)
                   (acc 0.0078 1/32) (values 1.25e-2 ,adam-drift) (sum 1.16e-2 0.15)
                   (sum-sq 2.0e-2 0.25)))))

  (define golden-profile
    (match-lambda
      ['x86_64 'x86_64]
      [_ 'cross-arch]))

  (define ((profile-column pick) profile)
    (match-define (list _ _ configs ...) (assq profile golden-evidence))
    (for/hasheq ([config (in-list configs)])
      (match-define (cons precision entries) config)
      (values precision
              (for/hasheq ([entry (in-list entries)])
                (match-define (list quantity gap bound) entry)
                (values quantity (pick gap bound))))))

  (define profile-gaps (profile-column (lambda (gap _) gap)))
  (define profile-tolerances (profile-column (lambda (_ bound) bound)))

  (define active-profile (golden-profile (system-type 'arch)))
  (define golden-tolerances (profile-tolerances active-profile))

  (define (golden-config precision)
    (hash-ref (hash-ref golden 'configs) precision))

  (define (check-golden label precision tr r classifier)
    (define g (golden-config precision))
    (test-case (format "G4-L23: ~a: the committed Lightning trace's config, names and groups" label)
      (check-equal? (hash-ref g 'config)
                    (twin-config #:batch fixture-batch-size #:epochs epochs
                                 #:total total-steps #:warmup warmup))
      (check-names r classifier g))
    (define ours (trace-golden tr initial-summary (state-summary (named-state classifier))))
    (printf "[20-fashion-mnist] ~a golden largest gaps [~a]: ~s\n" label active-profile
            (golden-gaps ours g))
    (test-case (format "G4-L24: ~a: the fit is the committed Lightning trace" label)
      (compare-golden label ours g (hash-ref golden-tolerances precision))))

  (check-golden "(a)" '32-true trace-a run-a classifier-a)
  (check-golden "(b)" 'bf16-mixed trace-b run-b classifier-b)

  (define (failures thunk)
    (define n 0)
    (parameterize ([current-check-around
                    (lambda (check)
                      (with-handlers ([exn:test:check? (lambda (_) (set! n (add1 n)))])
                        (check)))])
      (thunk))
    n)

  (define (hash-update-in h path f)
    (match path
      [(list k) (hash-update h k f)]
      [(cons k more) (hash-update h k (lambda (v) (hash-update-in v more f)))]))

  (define ours-a (trace-golden trace-a initial-summary (state-summary (named-state classifier-a))))

  (define ((first-of f) xs)
    (match-define (cons x more) xs)
    (cons (f x) more))

  (define ((each-row key f) rows)
    (for/list ([row (in-list rows)])
      (match-define (list step m) row)
      (list step (if (hash-has-key? m key) (hash-update m key f) m))))

  (define ((scaled-tolerance factor tolerances) quantity)
    (* factor (hash-ref tolerances quantity)))

  (define (float-shifts amount base)
    (match-define (hash-table ('shape dims) ('sum-sq sum-sq))
      (~> base (hash-ref 'state) (hash-ref 'net.3.weight)))
    (define ((by quantity [scale 1.0]) x)
      (~> (amount quantity) (* scale) (+ x)))
    `(((losses) ,(first-of (by 'losses)))
      ((norms) ,(first-of (by 'norms)))
      ((rows) ,(each-row 'train/loss_step (by 'rows)))
      ((rows) ,(each-row 'val/acc (by 'acc)))
      ((state net.3.weight head) ,(first-of (by 'values)))
      ((init net.5.bias tail) ,(first-of (by 'values)))
      ((state net.3.weight sum) ,(by 'sum (~> (apply * dims) (* sum-sq) sqrt)))
      ((state net.3.weight sum-sq) ,(by 'sum-sq sum-sq))))

  (define exact-perturbations
    `(((rows) ,(first-of (first-of add1)))
      ((state net.3.weight shape) ,reverse)
      ((state net.0.bn.num_batches_tracked sum) ,add1)
      ((labels) ,(first-of (first-of (lambda (y) (- 9 y)))))
      ((lrs) ,(first-of (first-of (lambda (x) (* x (+ 1 1e-12))))))
      ((scales) ,(lambda (_) '(65536.0)))))

  (define (shifted base ps)
    (for/fold ([g base]) ([p (in-list ps)])
      (match-define (list path f) p)
      (hash-update-in g path f)))

  (define (profile-failures profile g)
    (define tolerances (hash-ref (profile-tolerances profile) '32-true))
    (failures (lambda () (compare-golden "(a)" ours-a g tolerances))))

  (define (check-perturbed profile base)
    (define tolerances (hash-ref (profile-tolerances profile) '32-true))
    (define past (float-shifts (scaled-tolerance 2 tolerances) base))
    (check-equal? (profile-failures profile base) 0 (format "~a: unperturbed" profile))
    (for ([p (in-list (append past exact-perturbations))])
      (match-define (list path f) p)
      (check-true (positive? (profile-failures profile (hash-update-in base path f)))
                  (format "~a: ~a" profile path))))

  (test-case "G4-L25: a value perturbed past the active profile's tolerance fails G4-L24"
    (check-perturbed active-profile (golden-config '32-true)))

  (define aarch64-golden
    (let ([base (golden-config '32-true)]
          [gaps (hash-ref (profile-gaps 'cross-arch) '32-true)])
      (shifted base (float-shifts (lambda (quantity) (hash-ref gaps quantity)) base))))

  (test-case "G4-L28: the architecture picks the profile, and aarch64's gaps pass only its own"
    (check-equal? (map golden-profile '(x86_64 aarch64 arm ppc riscv64))
                  '(x86_64 cross-arch cross-arch cross-arch cross-arch))
    (for* ([profile (in-list '(x86_64 cross-arch))]
           [(precision gaps) (in-hash (profile-gaps profile))]
           [(quantity gap) (in-hash gaps)])
      (define bound (~> (profile-tolerances profile) (hash-ref precision) (hash-ref quantity)))
      (check-true (> bound (* 2 gap)) (format "~a ~a ~a" profile precision quantity)))
    (check-true (positive? (profile-failures 'x86_64 aarch64-golden)))
    (check-perturbed 'cross-arch aarch64-golden))

  (define (check-regenerated regenerated)
    (compare-golden-file regenerated golden golden-tolerances))

  (define (regenerated-failures path f)
    (failures (lambda () (check-regenerated (hash-update-in golden path f)))))

  (define (regenerated-shifts factor)
    (for*/list ([precision (in-list '(32-true bf16-mixed))]
                [tolerances (in-value (hash-ref golden-tolerances precision))]
                [p (in-list (float-shifts (scaled-tolerance factor tolerances)
                                          (golden-config precision)))])
      (match-define (list path f) p)
      (list (list* 'configs precision path) f)))

  (define ((rename-key from to) h)
    (hash-remove (hash-set h to (hash-ref h from)) from))

  (define (without-validation rows)
    (filter-not (match-lambda [(list _ m) (hash-has-key? m 'val/acc)]) rows))

  (define stale
    `(((lightning) ,(lambda (_) "2.6.6"))
      ((configs) ,(lambda (cs) (hash-remove cs 'bf16-mixed)))
      ((configs 32-true config lr) ,(lambda (x) (* 2 x)))
      ((configs 32-true config) ,(lambda (c) (hash-set c 'momentum 0.9)))
      ((configs 32-true names parameters) ,(first-of (lambda (s) (string-append s "_"))))
      ((configs bf16-mixed names decay) ,cdr)
      ((configs 32-true state) ,(rename-key 'net.3.weight 'net.3.weights))
      ((configs bf16-mixed state net.3.weight shape) ,reverse)
      ((configs 32-true state net.0.bn.num_batches_tracked sum) ,add1)
      ((configs 32-true labels) ,(first-of (first-of (lambda (y) (- 9 y)))))
      ((configs bf16-mixed lrs) ,(first-of (first-of (lambda (x) (* x (+ 1 1e-12))))))
      ((configs 32-true scales) ,(lambda (_) '(65536.0)))
      ((configs 32-true rows) ,(first-of (first-of add1)))
      ((configs bf16-mixed rows) ,without-validation)))

  (test-case "G4-L27: G4-L26 holds rounding to G4-L24's tolerances and the rest exactly"
    (check-equal? (regenerated-failures '(torch) (lambda (_) "2.15.0")) 0)
    (for ([p (in-list (regenerated-shifts 1/2))])
      (match-define (list path f) p)
      (check-not-equal? (hash-update-in golden path f) golden (format "~a" path))
      (check-equal? (regenerated-failures path f) 0 (format "~a" path)))
    (for ([p (in-list (append stale (regenerated-shifts 2)))])
      (match-define (list path f) p)
      (check-true (positive? (regenerated-failures path f)) (format "~a" path))))

  (cond
    [(not (and (python-torch-available?) (python-module-available? "lightning")))
     (printf "[20-fashion-mnist] twin parity and fixture freshness [~a] skipped: ~a\n"
             active-profile "python3 torch or lightning not available")]
    [else
     (test-case "G4-L26: the committed Lightning trace is the twin's golden output"
       (with-temporary-directory (root #:template "fashion-golden-~a")
         (define out (build-path root "golden.rktd"))
         (define log (open-output-string))
         (define ok?
           (parameterize ([current-output-port log] [current-error-port log])
             (call-with-python-env #:env '(("RKTORCH_TWIN_MODE" . "golden"))
                                   (lambda () (system* python "-I" twin-path out)))))
         (check-true ok? (get-output-string log))
         (check-regenerated (file->value out))))
     (check-twin "(a)" (cpu-device) '32-true trace-a run-a classifier-a tol tol)
     (check-twin "(b)" (cpu-device) 'bf16-mixed trace-b run-b classifier-b tol tol)])

  (cond
    [(not (and (python-torch-available?) (python-module-available? "lightning")
               (cuda-available?) (python-cuda-available?)))
     (displayln "[20-fashion-mnist] (c) skipped: CUDA, or python3 torch with CUDA, not available")]
    [else
     (define-values (trace-c run-c classifier-c _stepping-c _rows-c)
       (traced (cuda-device) '16-mixed))
     (check-true (grad-scaler? (run-grad-scaler run-c)))
     (check-twin "(c)" (cuda-device) '16-mixed trace-c run-c classifier-c 5e-3 (/ 1.0 128))]))
