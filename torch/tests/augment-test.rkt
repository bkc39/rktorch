#lang racket/base

(module+ test
  (require rackunit
           (only-in racket/list make-list)
           (only-in racket/match match-define)
           (only-in "../main.rkt"
                    * + - < cuda-available? cuda-device cuda-memory-stats
                    cuda-reset-peak-stats! device dtype make-generator
                    mps-available? mps-device native-memory-use numel rand
                    reclaim-native-memory! select shape stack tensor
                    tensor->list to to-dtype)
           (only-in "../vision/augment.rkt"
                    adjust-brightness adjust-contrast adjust-hue
                    adjust-saturation crop-view crop-view? multi-crop
                    random-affine-crop random-color-jitter random-grayscale
                    random-solarize resized-crop rgb->grayscale solarize)
           (only-in "private/blame.rkt" blames-caller?)
           (only-in "private/op-calls.rkt" op-calls-of)
           (only-in "private/within.rkt" bounded-test))

  (define devices
    (append (list 'cpu)
            (if (cuda-available?) (list (cuda-device)) '())
            (if (mps-available?) (list (mps-device)) '())))

  (define (values-of t)
    (tensor->list (to t 'cpu)))

  (define (same? a b)
    (and (equal? (shape a) (shape b))
         (equal? (values-of a) (values-of b))))

  (define (close? a b tolerance)
    (and (equal? (shape a) (shape b))
         (for/and ([u (in-list (flat (values-of a)))]
                   [v (in-list (flat (values-of b)))])
           (<= (abs (- u v)) tolerance))))

  (define (flat v)
    (if (list? v) (apply append (map flat v)) (list v)))

  (define (images n on #:side [side 12])
    (to (rand n 3 side side) on))

  (define (bytes-images n on #:side [side 32])
    (to (to-dtype (* (rand n 3 side side) 255) 'uint8) on))

  (define (dino-global #:solarize-p [solarize-p 0.0])
    (crop-view #:size 32 #:scale '(0.4 1.0) #:solarize-p solarize-p
               #:mean '(0.4914 0.4822 0.4465) #:std '(0.2470 0.2435 0.2616)))

  (define (dino-local)
    (crop-view #:size 16 #:scale '(0.05 0.4)
               #:mean '(0.4914 0.4822 0.4465) #:std '(0.2470 0.2435 0.2616)))

  (define (dino-crop x g)
    (multi-crop x
                (list (dino-global) (dino-global #:solarize-p 0.2))
                (make-list 6 (dino-local))
                #:generator g))

  (define (boxes-for x)
    (define n (car (shape x)))
    (to (tensor (make-list n '(1.0 2.0 7.5 6.0))) (device x)))

  (define (factors-for x lo hi)
    (define n (car (shape x)))
    (+ lo (* (- hi lo) (rand n #:device (device x)))))

  (define (flips-for x)
    (define n (car (shape x)))
    (< (rand n #:device (device x)) 0.5))

  (define cases
    (list
     (list "resized-crop" 'float
           (lambda (x g) (resized-crop x (boxes-for x) 8 #:flip (flips-for x))))
     (list "adjust-brightness" 'float
           (lambda (x g) (adjust-brightness x (factors-for x 0.5 1.5))))
     (list "adjust-contrast" 'float
           (lambda (x g) (adjust-contrast x (factors-for x 0.5 1.5))))
     (list "adjust-saturation" 'float
           (lambda (x g) (adjust-saturation x (factors-for x 0.5 1.5))))
     (list "adjust-hue" 'float
           (lambda (x g) (adjust-hue x (factors-for x -0.3 0.3))))
     (list "rgb->grayscale" 'float
           (lambda (x g) (rgb->grayscale x #:channels 3)))
     (list "solarize" 'float
           (lambda (x g) (solarize x (factors-for x 0.3 0.7))))
     (list "random-affine-crop" 'float
           (lambda (x g) (random-affine-crop x 8 #:flip-p 0.5 #:generator g)))
     (list "random-color-jitter" 'float
           (lambda (x g)
             (random-color-jitter x #:brightness 0.4 #:contrast 0.4
                                  #:saturation 0.2 #:hue 0.1 #:p 0.8
                                  #:generator g)))
     (list "random-grayscale" 'float
           (lambda (x g) (random-grayscale x #:p 0.2 #:generator g)))
     (list "random-solarize" 'float
           (lambda (x g) (random-solarize x 0.5 #:p 0.2 #:generator g)))
     (list "crop-view global" 'uint8
           (lambda (x g) ((dino-global #:solarize-p 0.2) x #:generator g)))
     (list "crop-view local" 'uint8
           (lambda (x g) ((dino-local) x #:generator g)))
     (list "multi-crop 2+6" 'uint8
           (lambda (x g)
             (define-values (big small) (dino-crop x g))
             (list big small)))))

  (define golden-op-calls
    '(("resized-crop" . 0)
      ("adjust-brightness" . 0)
      ("adjust-contrast" . 0)
      ("adjust-saturation" . 0)
      ("adjust-hue" . 0)
      ("rgb->grayscale" . 0)
      ("solarize" . 0)
      ("random-affine-crop" . 0)
      ("random-color-jitter" . 0)
      ("random-grayscale" . 0)
      ("random-solarize" . 0)
      ("crop-view global" . 0)
      ("crop-view local" . 0)
      ("multi-crop 2+6" . 0)))

  (define (input kind n on)
    (if (eq? kind 'uint8) (bytes-images n on) (images n on)))

  (define (op-calls kind run n on)
    (define x (input kind n on))
    (define (once) (run x (make-generator 1 #:device on)))
    (once)
    (op-calls-of once))

  (bounded-test "A1: every op makes as many native calls at 128 images as at 2"
    (for* ([on (in-list devices)]
           [c (in-list cases)])
      (match-define (list name kind run) c)
      (define small (op-calls kind run 2 on))
      (define large (op-calls kind run 128 on))
      (check-equal? large small (format "~a [~a]: ~a at 2, ~a at 128"
                                        name on small large))
      (check-true (<= small (cdr (assoc name golden-op-calls)))
                  (format "~a [~a]: ~a calls" name on small))))

  (define (draws-after run n on)
    (define g (make-generator 9 #:device on))
    (run (images n on) g)
    (values-of (rand 1 #:device on #:generator g)))

  (define (draws-of k n on)
    (define g (make-generator 9 #:device on))
    (for ([_ (in-range k)])
      (rand n #:device on #:generator g))
    (values-of (rand 1 #:device on #:generator g)))

  (define draw-table
    (list
     (list "random-affine-crop flip-p 0" 4
           (lambda (x g) (random-affine-crop x 8 #:generator g)))
     (list "random-affine-crop flip-p 0.5" 5
           (lambda (x g) (random-affine-crop x 8 #:flip-p 0.5 #:generator g)))
     (list "random-color-jitter" 5
           (lambda (x g) (random-color-jitter x #:brightness 0.4 #:p 0.8
                                              #:generator g)))
     (list "random-color-jitter p 0" 0
           (lambda (x g) (random-color-jitter x #:brightness 0.4 #:p 0
                                              #:generator g)))
     (list "random-grayscale" 1
           (lambda (x g) (random-grayscale x #:generator g)))
     (list "random-grayscale p 0" 0
           (lambda (x g) (random-grayscale x #:p 0 #:generator g)))
     (list "random-solarize" 1
           (lambda (x g) (random-solarize x 0.5 #:generator g)))
     (list "random-solarize p 0" 0
           (lambda (x g) (random-solarize x 0.5 #:p 0 #:generator g)))
     (list "crop-view default" 11
           (lambda (x g) ((crop-view #:size 8) x #:generator g)))
     (list "crop-view without colour" 5
           (lambda (x g) ((crop-view #:size 8 #:jitter-p 0 #:gray-p 0) x
                                                                       #:generator g)))
     (list "crop-view solarized" 12
           (lambda (x g) ((crop-view #:size 8 #:solarize-p 0.2) x #:generator g)))
     (list "multi-crop 1+2" 33
           (lambda (x g)
             (multi-crop x (list (crop-view #:size 8))
                         (list (crop-view #:size 4) (crop-view #:size 4))
                         #:generator g)))))

  (bounded-test "A2: each op advances its generator by the draw table's count"
    (for* ([on (in-list devices)]
           [c (in-list draw-table)])
      (match-define (list name k run) c)
      (check-equal? (draws-after run 5 on) (draws-of k 5 on)
                    (format "~a [~a]" name on))))

  (bounded-test "A3: a seed replays an op bitwise and another seed changes it"
    (for* ([on (in-list devices)]
           [c (in-list (list-tail cases 7))])
      (match-define (list name kind run) c)
      (define x (input kind 6 on))
      (define (with-seed s) (run x (make-generator s #:device on)))
      (define (flat-out v) (if (list? v) (map values-of v) (values-of v)))
      (check-equal? (flat-out (with-seed 0)) (flat-out (with-seed 0))
                    (format "~a [~a]" name on))
      (check-not-equal? (flat-out (with-seed 0)) (flat-out (with-seed 1))
                        (format "~a [~a]" name on))))

  (define (uniform-from g n on lo hi)
    (+ lo (* (- hi lo) (rand n #:device on #:generator g))))

  (bounded-test "A4: a random op at p 0 is the identity and at p 1 the functional op"
    (for ([on (in-list devices)])
      (define x (images 5 on))
      (define (g) (make-generator 3 #:device on))
      (check-true (same? (random-grayscale x #:p 0 #:generator (g)) x))
      (check-true (same? (random-solarize x 0.5 #:p 0 #:generator (g)) x))
      (check-true (same? (random-color-jitter x #:brightness 0.4 #:hue 0.1 #:p 0
                                              #:generator (g))
                         x))
      (check-true (same? (random-grayscale x #:p 1 #:generator (g))
                         (rgb->grayscale x #:channels 3)))
      (check-true (same? (random-solarize x 0.5 #:p 1 #:generator (g))
                         (solarize x 0.5)))
      (define twin (g))
      (rand 5 #:device on #:generator twin)
      (define fb (uniform-from twin 5 on 0.6 1.4))
      (define fc (uniform-from twin 5 on 0.7 1.3))
      (define fs (uniform-from twin 5 on 0.8 1.2))
      (define fh (uniform-from twin 5 on -0.1 0.1))
      (check-true
       (same? (random-color-jitter x #:brightness 0.4 #:contrast 0.3
                                   #:saturation 0.2 #:hue 0.1 #:p 1
                                   #:generator (g))
              (adjust-hue (adjust-saturation (adjust-contrast
                                              (adjust-brightness x fb) fc)
                                             fs)
                          fh))
       (format "jitter [~a]" on))
      (check-true (close? (random-affine-crop x 12 #:scale '(1 1) #:ratio '(1 1)
                                              #:generator (g))
                          x 1e-6)
                  (format "affine crop [~a]" on))))

  (bounded-test "A4: the functional ops at their neutral and extreme parameters"
    (for ([on (in-list devices)])
      (define x (images 4 on))
      (check-true (same? (adjust-brightness x 1) x))
      (check-true (same? (adjust-contrast x 1) x))
      (check-true (same? (adjust-saturation x 1) x))
      (check-true (same? (solarize x 1.01) x))
      (check-true (same? (solarize x 0) (- 1 x)))))

  (define (per-image op x factors)
    (stack (for/list ([i (in-naturals)] [f (in-list factors)])
             (select (op (to (stack (list (select x 0 i)) 0) (device x)) f) 0 0))
           0))

  (bounded-test "A5: a factor per image equals the images adjusted one at a time"
    (for ([on (in-list devices)])
      (define x (images 4 on))
      (for ([op (in-list (list adjust-brightness adjust-contrast adjust-saturation
                               adjust-hue solarize))]
            [factors (in-list '((0.5 1.0 1.3 2.0) (0.2 1.0 1.5 0.7)
                                (0.0 0.6 1.0 1.8) (-0.4 0.0 0.2 0.45)
                                (0.1 0.5 0.9 0.3)))])
        (define column (to (tensor factors) on))
        (check-true (same? (op x column) (per-image op x factors))
                    (format "~a [~a]" (object-name op) on)))))

  (bounded-test "A6: hue by h then -h, and 0.5 twice, returns the image"
    (for ([on (in-list devices)])
      (define x (images 4 on))
      (for ([h (in-list '(0.1 -0.3 0.45))])
        (check-true (close? (adjust-hue (adjust-hue x h) (- h)) x 1e-5)
                    (format "~a [~a]" h on)))
      (check-true (close? (adjust-hue (adjust-hue x 0.5) 0.5) x 1e-5))))

  (bounded-test "A7: results stay on the input's device as float32"
    (for* ([on (in-list devices)]
           [c (in-list cases)])
      (match-define (list name kind run) c)
      (define x (input kind 3 on))
      (define out (run x (make-generator 2 #:device on)))
      (for ([t (in-list (if (list? out) out (list out)))])
        (check-equal? (device t) (device x) name)
        (check-eq? (dtype t) 'float32 name))))

  (define blames-this-test (blames-caller? "augment-test"))

  (bounded-test "A8: malformed arguments are blamed on the caller"
    (define x (images 2 'cpu))
    (define u8 (bytes-images 2 'cpu))
    (check-true (crop-view? (dino-local)))
    (check-exn blames-this-test (lambda () (adjust-brightness u8 1.2)))
    (check-exn blames-this-test (lambda () (adjust-contrast u8 1.2)))
    (check-exn blames-this-test (lambda () (rgb->grayscale u8)))
    (check-exn blames-this-test
               (lambda () (random-color-jitter (rand 2 1 8 8) #:brightness 0.4)))
    (check-exn blames-this-test
               (lambda () (resized-crop x (tensor '((0.0 0.0 4.0) (0.0 0.0 4.0))) 4)))
    (check-exn blames-this-test
               (lambda () (adjust-saturation x (tensor '(1.0 1.0 1.0)))))
    (check-exn blames-this-test
               (lambda () (random-affine-crop x 4 #:scale '(0.9 0.1))))
    (check-exn blames-this-test
               (lambda () (random-color-jitter x #:hue 0.6)))
    (check-exn blames-this-test (lambda () (adjust-hue x 0.7)))
    (check-exn blames-this-test
               (lambda () (multi-crop u8 (list (crop-view #:size 8)
                                               (crop-view #:size 6))
                                      '())))
    (check-exn blames-this-test
               (lambda () (multi-crop u8 (list (crop-view #:size 8))
                                      (list (crop-view #:size 4)
                                            (crop-view #:size '(4 6))))))
    (check-exn blames-this-test (lambda () (multi-crop (rand 3 8 8) (list (dino-local)) '())))
    (check-exn blames-this-test (lambda () (crop-view #:size 0)))
    (check-exn (blames-caller? "caller")
               (lambda () ((dino-local) (rand 3 8 8))))
    (for ([on (in-list (cdr devices))])
      (define there (images 2 on))
      (check-exn blames-this-test
                 (lambda () (random-grayscale there #:generator (make-generator 1))))
      (check-exn blames-this-test
                 (lambda () (random-solarize x 0.5 #:generator (make-generator 1 #:device on))))
      (check-exn blames-this-test
                 (lambda () (resized-crop there (boxes-for x) 4)))
      (check-exn blames-this-test
                 (lambda () (adjust-brightness there (tensor '(1.0 1.0)))))
      (check-exn blames-this-test
                 (lambda () (multi-crop (to u8 on) (list (dino-local)) '()
                                        #:generator (make-generator 1))))))

  (define (bytes-on on)
    (for/sum ([entry (in-list (native-memory-use))]
              #:when (equal? (car entry) on))
      (cdr entry)))

  (define (peak-above on thunk)
    (reclaim-native-memory!)
    (define before (cdr (assq 'allocated (cuda-memory-stats on))))
    (cuda-reset-peak-stats! on)
    (define out (thunk))
    (define peak (cdr (assq 'peak-allocated (cuda-memory-stats on))))
    (define kept (for/sum ([t (in-list out)]) (* 4 (numel t))))
    (- peak before kept))

  (bounded-test "A9: multi-crop leaves no native memory behind and runs its views in turn"
    (for ([on (in-list devices)])
      (define x (bytes-images 256 on))
      (dino-crop x (make-generator 4 #:device on))
      (reclaim-native-memory!)
      (define before (bytes-on (device x)))
      (dino-crop x (make-generator 4 #:device on))
      (reclaim-native-memory!)
      (check-equal? (bytes-on (device x)) before (format "~a" on)))
    (when (cuda-available?)
      (define on (cuda-device))
      (define x (bytes-images 256 on))
      (dino-crop x (make-generator 4 #:device on))
      (define all
        (peak-above on (lambda ()
                         (define-values (big small)
                           (dino-crop x (make-generator 4 #:device on)))
                         (list big small))))
      (define one
        (peak-above on (lambda ()
                         (list ((dino-global) x
                                              #:generator (make-generator 4 #:device on))))))
      (check-true (<= all (+ one (expt 2 20)))
                  (format "multi-crop ~a bytes above its outputs, one view ~a"
                          all one)))))
