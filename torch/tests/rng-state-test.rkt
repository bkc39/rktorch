#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/match match-define)
           (only-in racket/sequence sequence->list)
           (only-in threading ~>)
           (only-in "../main.rkt"
                    bytes->tensor cat cpu-device cuda-available? cuda-device
                    cuda-device-count current-generator device device-type
                    draw-seed dtype generator-state in-flattened-tensor
                    make-generator manual-seed! mps-available? mps-device
                    narrow rand randn randperm reshape rng-state
                    set-generator-state! set-rng-state! shape tensor
                    tensor->bytes to uniform! zero! zeros)
           (only-in "../foreign/generator.rkt" normal!)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/race-harness.rkt" with-race-hook)
           (only-in "private/within.rkt" bounded-test))

  (define devices
    (append (list (cpu-device))
            (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define state-sizes (hash 'cpu 5056 'cuda 16 'mps 44))

  (define (state-size on) (hash-ref state-sizes (device-type on)))

  (define (cpu? on) (eq? 'cpu (device-type on)))

  (define (host t)
    (~> t (to 'cpu) in-flattened-tensor sequence->list))

  (define (filled fill! on)
    (define t (zeros 4 #:device on))
    (fill! t)
    (host t))

  (define (draws on g #:seed? [seed? (or (and g #t) (cpu? on))])
    (parameterize ([current-generator g])
      (append (list (host (rand 5 #:device on))
                    (host (randn 3 2 #:device on))
                    (host (randn 17 #:device on))
                    (filled (lambda (t) (normal! t 0.0 1.0)) on)
                    (filled (lambda (t) (uniform! t -1.0 1.0)) on))
              (if seed? (list (draw-seed)) '())
              (if (cpu? on) (list (host (randperm 6))) '()))))

  (struct stream (on read restore! generator))

  (define (fresh kind on seed)
    (case kind
      [(default)
       (manual-seed! seed)
       (stream on
               (lambda () (rng-state #:device on))
               (lambda (s) (set-rng-state! s #:device on))
               #f)]
      [else
       (define g (make-generator seed #:device on))
       (stream on
               (lambda () (generator-state g))
               (lambda (s) (set-generator-state! g s))
               g)]))

  (define kinds '(default generator))

  (define (read-state st) ((stream-read st)))
  (define (restore! st s) ((stream-restore! st) s))
  (define (draws-of st) (draws (stream-on st) (stream-generator st)))

  (define (warm! st)
    (parameterize ([current-generator (stream-generator st)])
      (randn 3 #:device (stream-on st))))

  (define (label kind on) (format "~a ~a" kind on))

  (define (u8 bs) (bytes->tensor bs 'uint8 (list (bytes-length bs))))

  (define (patched s start end v)
    (define bs (bytes-copy (tensor->bytes s)))
    (integer->integer-bytes v (- end start) #f #f bs start)
    (u8 bs))

  (define (le-uint bs start end) (integer-bytes->integer bs #f #f start end))

  (bounded-test "L1: a state is a uint8 CPU tensor of its device type's size"
    (for* ([on (in-list devices)]
           [s (in-list (list (rng-state #:device on)
                             (generator-state (make-generator 0 #:device on))))])
      (check-equal? (device s) (cpu-device) (format "~a" on))
      (check-equal? (dtype s) 'uint8 (format "~a" on))
      (check-equal? (shape s) (list (state-size on)) (format "~a" on))))

  (bounded-test "L2: a seeded default stream holds the state a seeded generator does"
    (for* ([on (in-list devices)]
           [seed (in-list (list 0 7 (sub1 (expt 2 64))))])
      (manual-seed! seed)
      (define default (tensor->bytes (rng-state #:device on)))
      (define generated
        (~> (make-generator seed #:device on) generator-state tensor->bytes))
      (check-equal? default generated (format "~a ~a" on seed))
      (define seed-at (if (eq? 'mps (device-type on)) 28 0))
      (check-equal? (le-uint default seed-at (+ seed-at 8)) seed
                    (format "~a ~a" on seed))
      (when (eq? 'cuda (device-type on))
        (check-equal? (le-uint default 8 16) 0 (format "~a ~a" on seed)))))

  (bounded-test "L3: reading a state draws nothing"
    (for* ([on (in-list devices)]
           [kind (in-list kinds)])
      (define never-read (draws-of (fresh kind on 5)))
      (define st (fresh kind on 5))
      (check-equal? (tensor->bytes (read-state st))
                    (tensor->bytes (read-state st))
                    (label kind on))
      (check-equal? (draws-of st) never-read (label kind on))))

  (bounded-test "L4: a state is a snapshot, and setting it copies it"
    (for* ([on (in-list devices)]
           [kind (in-list kinds)])
      (define st (fresh kind on 6))
      (warm! st)
      (define s (read-state st))
      (define before (tensor->bytes s))
      (draws-of st)
      (check-equal? (tensor->bytes s) before (label kind on))
      (restore! st s)
      (zero! s)
      (define after-zeroing (draws-of st))
      (restore! st (u8 before))
      (check-equal? (draws-of st) after-zeroing (label kind on))))

  (bounded-test "L5: a generator restored to a state replays its draws"
    (for ([on (in-list devices)])
      (define st (fresh 'generator on 13))
      (warm! st)
      (define s (read-state st))
      (define first-run (draws-of st))
      (restore! st s)
      (check-equal? (draws-of st) first-run (format "~a" on))))

  (bounded-test "L6: a default stream restored to a state replays its draws"
    (for ([on (in-list devices)])
      (define st (fresh 'default on 13))
      (warm! st)
      (define s (read-state st))
      (define first-run (draws-of st))
      (define unrelated (make-generator 21 #:device on))
      (define unrelated-run (draws on (make-generator 21 #:device on)))
      (restore! st s)
      (check-equal? (draws-of st) first-run (format "~a" on))
      (check-equal? (draws on unrelated) unrelated-run (format "~a" on))))

  (bounded-test "L7: a state read from one stream makes another replay it"
    (for ([on (in-list devices)])
      (define g₁ (make-generator 3 #:device on))
      (define g₂ (make-generator 99 #:device on))
      (rand 7 #:device on #:generator g₁)
      (randn 5 #:device on #:generator g₂)
      (set-generator-state! g₂ (generator-state g₁))
      (check-equal? (draws on g₂) (draws on g₁) (format "g₁ -> g₂ ~a" on))
      (warm! (fresh 'default on 8))
      (define g₃ (make-generator 0 #:device on))
      (set-generator-state! g₃ (rng-state #:device on))
      (check-equal? (draws on g₃ #:seed? (cpu? on)) (draws on #f)
                    (format "default -> generator ~a" on))
      (define g₄ (make-generator 12 #:device on))
      (rand 3 #:device on #:generator g₄)
      (set-rng-state! (generator-state g₄) #:device on)
      (check-equal? (draws on #f) (draws on g₄ #:seed? (cpu? on))
                    (format "generator -> default ~a" on))))

  (cond
    [(cuda-available?)
     (bounded-test "L8: setting the CPU stream leaves the CUDA stream alone"
       (manual-seed! 1)
       (define s (rng-state #:device 'cuda))
       (define expected (draws (cuda-device) #f))
       (set-rng-state! s #:device 'cuda)
       (define cpu-state (rng-state))
       (rand 9)
       (set-rng-state! cpu-state #:device 'cpu)
       (check-equal? (draws (cuda-device) #f) expected))]
    [else
     (displayln "[rng-state-test] skipped L8: needs CUDA")])

  (bounded-test "L9: a state seen through a view or another shape replays"
    (for ([kind (in-list kinds)])
      (define st (fresh kind (cpu-device) 4))
      (warm! st)
      (define s (read-state st))
      (define expected (draws-of st))
      (for ([view (in-list
                   (list (narrow (cat (list (zeros 4 #:dtype 'uint8) s)) 0 4 5056)
                         (reshape s 2 2528)))])
        (restore! st view)
        (check-equal? (draws-of st) expected
                      (format "~a ~a" kind (shape view))))))

  (define ((refused expected) e)
    (and ((blames-caller? "rng-state-test") e)
         (regexp-match? expected (exn-message e))
         (not (regexp-match? #rx"FFI call failed" (exn-message e)))))

  (define (check-refusals rows)
    (for ([row (in-list rows)])
      (match-define (list name expected thunk) row)
      (check-exn (refused expected) thunk name)))

  (define cpu-expected #rx"expected: cpu-rng-state")
  (define cuda-expected #rx"expected: cuda-rng-state")

  (bounded-test "L10: a tensor that is not a state of the device's type blames the caller"
    (manual-seed! 2)
    (define valid (rng-state))
    (define cuda-shaped (u8 (make-bytes 16 0)))
    (check-refusals
     (list
      (list "a: float32 pair" cpu-expected
            (lambda () (set-rng-state! (tensor '(1.0 2.0)))))
      (list "b: float32 of 5056" cpu-expected
            (lambda () (set-rng-state! (zeros 5056))))
      (list "c: 5055 bytes" cpu-expected
            (lambda () (set-rng-state! (narrow valid 0 0 5055))))
      (list "d: 5056 zero bytes" cpu-expected
            (lambda () (set-rng-state! (zeros 5056 #:dtype 'uint8))))
      (list "e: next = 625" cpu-expected
            (lambda () (set-rng-state! (patched valid 16 24 625))))
      (list "e: next = 625 + 2^32" cpu-expected
            (lambda ()
              (set-rng-state! (patched valid 16 24 (+ 625 (expt 2 32))))))
      (list "f: seeded = 0" cpu-expected
            (lambda () (set-rng-state! (patched valid 12 16 0))))
      (list "g: left = 625" cpu-expected
            (lambda () (set-rng-state! (patched valid 8 12 625))))
      (list "h: a CUDA-shaped state on the CPU stream" cpu-expected
            (lambda () (set-rng-state! cuda-shaped)))
      (list "i: a CUDA-shaped state on a CPU generator" cpu-expected
            (lambda () (set-generator-state! (make-generator 1) cuda-shaped)))
      (list "j: a CPU state on the CUDA stream" cuda-expected
            (lambda () (set-rng-state! (rng-state) #:device 'cuda)))
      (list "k: a Philox offset of 3" cuda-expected
            (lambda ()
              (set-rng-state! (patched cuda-shaped 8 16 3) #:device 'cuda)))
      (list "l: 24 bytes for CUDA" cuda-expected
            (lambda () (~> (make-bytes 24 0) u8 (set-rng-state! #:device 'cuda))))
      (list "m: a byte string" cpu-expected
            (lambda () (set-rng-state! (make-bytes 5056 0))))
      (list "n: not a generator" #rx"expected: generator\\?"
            (lambda () (set-generator-state! 'g valid)))
      (list "o: not a device" #rx"expected: \\(or/c device\\?"
            (lambda () (rng-state #:device 'tpu))))))

  (cond
    [(cuda-available?)
     (bounded-test "L10 p-r: CUDA states are refused off their type and taken short"
       (check-refusals
        (list
         (list "p: a CPU state moved to CUDA" cpu-expected
               (lambda () (set-rng-state! (to (rng-state) 'cuda))))
         (list "q: a CPU state on a CUDA generator" cuda-expected
               (lambda ()
                 (set-generator-state! (make-generator 1 #:device 'cuda)
                                       (rng-state))))
         (list "q: a CUDA state on a CPU generator" cpu-expected
               (lambda ()
                 (set-generator-state! (make-generator 1)
                                       (rng-state #:device 'cuda))))))
       (manual-seed! 17)
       (rand 5 #:device 'cuda)
       (define seed-only
         (~> (rng-state #:device 'cuda) tensor->bytes (subbytes 0 8) u8))
       (set-rng-state! seed-only #:device 'cuda)
       (check-equal? (draws (cuda-device) #f)
                     (draws (cuda-device)
                            (make-generator 17 #:device 'cuda)
                            #:seed? #f)
                     "r: the seed alone"))]
    [else
     (displayln "[rng-state-test] skipped L10 p-r: needs CUDA")])

  (cond
    [(mps-available?)
     (bounded-test "L10: a CPU state on the MPS stream blames the caller"
       (check-refusals
        (list (list "a CPU state on MPS" #rx"expected: mps-rng-state"
                    (lambda () (set-rng-state! (rng-state) #:device 'mps))))))]
    [else
     (displayln "[rng-state-test] skipped the L10 MPS row: needs MPS")])

  (define (absent? e)
    (and (exn:fail? e) (not (exn:fail:contract? e))))

  (bounded-test "L11: a device that is not present is a failure, not a blame"
    (define missing (cuda-device (cuda-device-count)))
    (check-exn absent? (lambda () (rng-state #:device missing)))
    (check-exn absent?
               (lambda () (set-rng-state! (u8 (make-bytes 16 0)) #:device missing)))
    (unless (mps-available?)
      (check-exn absent? (lambda () (rng-state #:device 'mps)))
      (check-exn absent?
                 (lambda ()
                   (set-rng-state! (u8 (make-bytes 44 0)) #:device 'mps)))))

  (bounded-test "L11: a CUDA seed for a listed absent index passes the contract"
    (check-exn absent?
               (lambda ()
                 (set-rng-state! (u8 (make-bytes 8 0))
                                 #:device (list 'cuda (cuda-device-count))))))

  (bounded-test "the CPU stream is the one read and set without a device"
    (manual-seed! 9)
    (check-equal? (tensor->bytes (rng-state))
                  (tensor->bytes (rng-state #:device 'cpu)))
    (define s (rng-state))
    (define first-run (draws (cpu-device) #f))
    (set-rng-state! s)
    (check-equal? (draws (cpu-device) #f) first-run))

  (bounded-test "the default stream is the one read and set whatever current-generator holds"
    (for ([on (in-list devices)])
      (manual-seed! 9)
      (rand 3 #:device on)
      (define g (make-generator 1 #:device on))
      (define g-state (tensor->bytes (generator-state g)))
      (define s (rng-state #:device on))
      (check-equal? (parameterize ([current-generator g])
                      (tensor->bytes (rng-state #:device on)))
                    (tensor->bytes s)
                    (format "read ~a" on))
      (define expected (draws on #f))
      (parameterize ([current-generator g])
        (set-rng-state! s #:device on))
      (check-equal? (draws on #f) expected (format "set ~a" on))
      (check-equal? (tensor->bytes (generator-state g)) g-state
                    (format "set leaves current-generator ~a" on))))

  (bounded-test "a CPU state's next is read as its low 32 bits, as libtorch reads it"
    (manual-seed! 3)
    (rand 9)
    (define valid (rng-state))
    (define next (le-uint (tensor->bytes valid) 16 24))
    (define expected (draws (cpu-device) #f))
    (set-rng-state! (patched valid 16 24 (+ next (expt 2 32))))
    (check-equal? (draws (cpu-device) #f) expected))

  (define (ledger-entries thunk)
    (define here (current-thread))
    (define added 0)
    (with-race-hook (lambda (name _subject)
                      (when (and (eq? name 'ledger-entry-added)
                                 (eq? (current-thread) here))
                        (set! added (add1 added))))
      (thunk))
    added)

  (bounded-test "L12: a read puts one tensor on the ledger, a set none"
    (for ([on (in-list devices)])
      (define g (make-generator 5 #:device on))
      (define default-state (rng-state #:device on))
      (define generator-state₀ (generator-state g))
      (check-equal? (ledger-entries (lambda () (rng-state #:device on))) 1
                    (format "rng-state ~a" on))
      (check-equal? (ledger-entries (lambda () (generator-state g))) 1
                    (format "generator-state ~a" on))
      (check-equal? (ledger-entries
                     (lambda () (set-rng-state! default-state #:device on)))
                    0
                    (format "set-rng-state! ~a" on))
      (check-equal? (ledger-entries
                     (lambda () (set-generator-state! g generator-state₀)))
                    0
                    (format "set-generator-state! ~a" on)))))
