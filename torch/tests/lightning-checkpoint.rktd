#hasheq((lightning . "2.6.5")
        (cases
         .
         #hasheq(
   (every-2-steps
    .
    (
     (save "epoch=0-step=2.ckpt")
     (save "epoch=1-step=4.ckpt")
     (save "epoch=2-step=6.ckpt")
     (save "epoch=3-step=8.ckpt")
     (save "epoch=4-step=10.ckpt")
     (save "epoch=5-step=12.ckpt")
     (best-k)
     (best "epoch=5-step=12.ckpt" #f)
     (kth #f +inf.0)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 2, 'every_n_epochs': 0, 'train_time_interval': None}")))
   (max-top-all
    .
    (
     (save "epoch=0-step=3.ckpt")
     (save "epoch=1-step=6.ckpt")
     (save "epoch=2-step=9.ckpt")
     (save "epoch=3-step=12.ckpt")
     (save "epoch=4-step=15.ckpt")
     (save "epoch=5-step=18.ckpt")
     (best-k ("epoch=0-step=3.ckpt" . 0.5) ("epoch=1-step=6.ckpt" . 0.30000001192092896) ("epoch=2-step=9.ckpt" . 0.4000000059604645) ("epoch=3-step=12.ckpt" . -inf.0) ("epoch=4-step=15.ckpt" . 0.20000000298023224) ("epoch=5-step=18.ckpt" . 0.6000000238418579))
     (best "epoch=5-step=18.ckpt" 0.6000000238418579)
     (kth "epoch=3-step=12.ckpt" -inf.0)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': 'val-loss', 'mode': 'max', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")))
   (min-top-2
    .
    (
     (save "epoch=0-val-loss=0.50.ckpt")
     (save "epoch=1-val-loss=0.30.ckpt")
     (save "epoch=2-val-loss=0.40.ckpt")
     (delete "epoch=0-val-loss=0.50.ckpt")
     (save "epoch=4-val-loss=0.20.ckpt")
     (delete "epoch=2-val-loss=0.40.ckpt")
     (best-k ("epoch=1-val-loss=0.30.ckpt" . 0.30000001192092896) ("epoch=4-val-loss=0.20.ckpt" . 0.20000000298023224))
     (best "epoch=4-val-loss=0.20.ckpt" 0.20000000298023224)
     (kth "epoch=1-val-loss=0.30.ckpt" 0.30000001192092896)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': 'val-loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")))
   (no-monitor
    .
    (
     (save "epoch=0-step=3.ckpt")
     (save "epoch=1-step=6.ckpt")
     (delete "epoch=0-step=3.ckpt")
     (save "epoch=2-step=9.ckpt")
     (delete "epoch=1-step=6.ckpt")
     (save "epoch=3-step=12.ckpt")
     (delete "epoch=2-step=9.ckpt")
     (save "epoch=4-step=15.ckpt")
     (delete "epoch=3-step=12.ckpt")
     (save "epoch=5-step=18.ckpt")
     (delete "epoch=4-step=15.ckpt")
     (best-k)
     (best "epoch=5-step=18.ckpt" #f)
     (kth #f +inf.0)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")))
   (pre-existing
    .
    (
     (save "epoch=0-step=3-v1.ckpt")
     (save "epoch=1-step=6.ckpt")
     (delete "epoch=0-step=3-v1.ckpt")
     (save "epoch=2-step=9.ckpt")
     (delete "epoch=1-step=6.ckpt")
     (save "epoch=3-step=12.ckpt")
     (delete "epoch=2-step=9.ckpt")
     (save "epoch=4-step=15.ckpt")
     (delete "epoch=3-step=12.ckpt")
     (save "epoch=5-step=18.ckpt")
     (delete "epoch=4-step=15.ckpt")
     (best-k)
     (best "epoch=5-step=18.ckpt" #f)
     (kth #f +inf.0)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': None, 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")))
   (save-last
    .
    (
     (save "epoch=0-step=3.ckpt")
     (save "last.ckpt")
     (save "epoch=1-step=6.ckpt")
     (delete "epoch=0-step=3.ckpt")
     (save "last.ckpt")
     (save "epoch=4-step=15.ckpt")
     (delete "epoch=1-step=6.ckpt")
     (save "last.ckpt")
     (best-k ("epoch=4-step=15.ckpt" . 0.20000000298023224))
     (best "epoch=4-step=15.ckpt" 0.20000000298023224)
     (kth "epoch=4-step=15.ckpt" 0.20000000298023224)
     (last . "last.ckpt")
     (state-key . "ModelCheckpoint{'monitor': 'val-loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}")))
   (unformatted
    .
    (
     (save "epoch=0-val-loss=0.5.ckpt")
     (save "epoch=1-val-loss=0.30000001192092896.ckpt")
     (delete "epoch=0-val-loss=0.5.ckpt")
     (save "epoch=4-val-loss=0.20000000298023224.ckpt")
     (delete "epoch=1-val-loss=0.30000001192092896.ckpt")
     (best-k ("epoch=4-val-loss=0.20000000298023224.ckpt" . 0.20000000298023224))
     (best "epoch=4-val-loss=0.20000000298023224.ckpt" 0.20000000298023224)
     (kth "epoch=4-val-loss=0.20000000298023224.ckpt" 0.20000000298023224)
     (last . #f)
     (state-key . "ModelCheckpoint{'monitor': 'val-loss', 'mode': 'min', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"))))))
