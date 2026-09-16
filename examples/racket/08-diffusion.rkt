#lang scribble/lp2

@(require (for-label (except-in racket/base abs cos exp log sin sqrt max min length + - * /)
                     torch torch/nn torch/data/loader torch/vision/cifar10
                     torch/vision/diffusion))

@section[#:tag "ex-diffusion"]{Training a denoising diffusion model on CIFAR-10}

The generative-vision capstone: a DDPM in the epsilon-prediction form. A
noising process adds Gaussian noise to a training image over @tt{T} steps;
a small UNet learns to predict the noise that was added at a random step;
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
output on that noise with a mean squared error. Both draws are made on the
CPU whatever device the model trains on, and the seed is set again once the
model is built, since building it consumes the CPU stream only on a CPU
run; a seeded run therefore replays the same timesteps and noise on the GPU
as on the CPU, the way the PyTorch twin of this example draws them. The
device is the accelerator when there is one. On Apple silicon the
@racket[GroupNorm] layers normalise on the CPU, since libtorch 2.9 has no
MPS kernel for that backward, and everything else in the UNet stays on the
GPU.

@chunk[<r08-step>
(define (pick-device)
  (accelerator-if-available))

(define (train-step net sched opt xs device)
  (define n (length xs))
  (define steps (schedule-steps sched))
  (define t (to (to-dtype (mul (rand n #:device 'cpu) steps) 'int64) device))
  (define noise (to (randn-like xs #:device 'cpu) device))
  (zero-grads! opt)
  (define loss (mse-loss (net (q-sample sched xs t noise) t) noise))
  (backward! loss)
  (step! opt)
  (item loss))]

@bold{The deterministic core.} @racket[run-example] is the seeded, offline
entry the test harness and the parity twin drive: a fresh @racket[UNet] and a
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
    (define net (UNet))
    (define sched (linear-schedule))
    (define opt (adam (parameters net) #:lr 0.001))
    (manual-seed! 0)
    (define losses
      (for/list ([_ (in-range steps)])
        (train-step net sched opt xs device)))
    (values losses net device)))]

@bold{The headline run.} The full training set through the shuffled loader,
device resident so a batch is one gather on the GPU, with the mean training
loss per epoch reported. On an RTX 3090 Ti an epoch of 391 batches takes
about half a minute at the default width; the mean loss is near @tt{0.16}
after the first epoch and keeps falling for tens of epochs, which is the
regime the sampler in the next example works in.

@chunk[<r08-train>
(define (train-cifar10 #:epochs [epochs 10] #:batch [batch 128]
                       #:base [base 32] #:lr [lr 2e-4]
                       #:device [device (pick-device)])
  (with-default-device device
    (manual-seed! 0)
    (define loader
      (dataloader (cifar10-dataset 'train #:device device)
                  #:batch-size batch #:shuffle? #t
                  #:generator (make-generator 0)))
    (define net (UNet #:base base))
    (define sched (linear-schedule))
    (define opt (adam (parameters net) #:lr lr))
    (manual-seed! 0)
    (for/list ([epoch (in-range epochs)])
      (define total
        (for/sum ([(xb _yb) loader])
          (train-step net sched opt xb device)))
      (/ total (length loader)))))]

@chunk[<*>
  <r08-require>
  <r08-provide>
  <r08-step>
  <r08-run>
  <r08-train>]
