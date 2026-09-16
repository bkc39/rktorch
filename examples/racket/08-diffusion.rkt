#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/vision/cifar10
                     torch/vision/diffusion))

@section[#:tag "ex-diffusion"]{Training a denoising diffusion model on CIFAR-10}

The generative-vision capstone: a DDPM in the epsilon-prediction form. A
noising process adds Gaussian noise to a training image over @tt{T} steps;
a UNet learns to predict the noise that was added at a random step;
sampling (the next example) runs that predictor backwards from pure noise.
Training needs only three pieces beyond the loader: a @racket[schedule] of
noise levels, the closed-form @racket[q-sample] that jumps straight to step
@tt{t}, and the network. All three live in @racketmodname[torch/vision/diffusion];
this program is the loop around them.

@chunk[<r08-require>
(require torch torch/nn torch/data/loader
         (only-in torch/vision/cifar10 cifar10-dataset load-cifar10-fixture)
         torch/vision/diffusion)]

@chunk[<r08-provide>
(provide pick-device train-step run-example train-cifar10)]

@bold{One step.} Draw a timestep per image uniformly from @tt{[0, T)}, draw
the noise, form @tt{x_t} with @racket[q-sample], and regress the network's
output on that noise with a mean squared error. A class-conditional network
also sees the labels, with a tenth of them replaced by the null label so
the same network learns the unconditional estimate too, which is what
classifier-free guidance needs at sampling time. Every draw is made on the
CPU whatever device the model trains on, and the seed is set again once the
model is built, since building it consumes the CPU stream only on a CPU
run; a seeded run therefore replays the same timesteps, noise and dropped
labels on the GPU as on the CPU, the way the PyTorch twin of this example
draws them. The device is the accelerator when there is one. On Apple
silicon the @racket[GroupNorm] layers normalise on the CPU, since libtorch
2.9 has no MPS kernel for that backward, and everything else in the UNet
stays on the GPU.

@chunk[<r08-step>
(define (pick-device)
  (accelerator-if-available))

(define (train-step net sched opt xs ys device #:null-label [null-label 10]
                    #:label-dropout [label-dropout 0.1])
  (define n (length xs))
  (define steps (schedule-steps sched))
  (define t (to (to-dtype (mul (rand n #:device 'cpu) steps) 'int64) device))
  (define noise (to (randn-like xs #:device 'cpu) device))
  (define y
    (and ys
         (where (to (lt (rand n #:device 'cpu) label-dropout) device)
                (full null-label n #:dtype 'int64 #:device device)
                ys)))
  (zero-grads! opt)
  (define loss (mse-loss (net (q-sample sched xs t noise) t y) noise))
  (backward! loss)
  (step! opt)
  (item loss))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the test harness and the parity twin drive: a small unconditional
@racket[UNet], base 64, two levels with attention at 16x16 and no dropout, a
@racket[linear-schedule], five full-batch @racket[adam] steps on the committed
256-image fixture, and the per-step losses back. As in the earlier examples
the process default device is set for the dynamic extent of the run with
@racket[with-default-device], so the schedule's tables and the model land
together.

@chunk[<r08-run>
(define (run-example #:steps [steps 5] #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define-values (xs _ys) (load-cifar10-fixture))
    (define net (UNet #:base 64 #:mults '(1 2) #:blocks 1 #:attention '(16)
                      #:dropout 0))
    (define sched (linear-schedule))
    (define opt (adam (parameters net) #:lr 0.001))
    (manual-seed! 0)
    (define losses
      (for/list ([_ (in-range steps)])
        (train-step net sched opt xs #f device)))
    (values losses net device)))]

@bold{The headline run.} The full training set through the shuffled loader,
device resident so a batch is one gather on the GPU, the class-conditional
DDPM network at the paper's CIFAR-10 size, with the mean training loss per
epoch reported. An epoch of 391 batches takes minutes on an RTX 3090 Ti;
the mean loss is near @tt{0.06} after the first epoch and settles around
@tt{0.03} over tens of epochs, which is the regime the sampler in the next
example works in.

@chunk[<r08-train>
(define (train-cifar10 #:epochs [epochs 10] #:batch [batch 128]
                       #:base [base 128] #:lr [lr 2e-4]
                       #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define loader
      (dataloader (cifar10-dataset 'train #:device device)
                  #:batch-size batch #:shuffle? #t
                  #:generator (make-generator 0)))
    (define net (UNet #:base base #:classes 10))
    (define sched (linear-schedule))
    (define opt (adam (parameters net) #:lr lr))
    (manual-seed! 0)
    (for/list ([epoch (in-range epochs)])
      (define total
        (for/sum ([(xb yb) loader])
          (train-step net sched opt xb yb device)))
      (/ total (length loader)))))]

@chunk[<*>
  <r08-require>
  <r08-provide>
  <r08-step>
  <r08-run>
  <r08-train>]
