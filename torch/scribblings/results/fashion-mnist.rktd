#hasheq((command
         .
         "EPOCHS=5 nix develop .#cuda --command racket scripts/fashion-mnist-results.rkt")
        (commit . "e906d5d192a6")
        (config
         .
         #hasheq((accumulate . 2)
                 (batch . 128)
                 (clip . 0.5)
                 (epochs . 5)
                 (lightning . "2.6.5")
                 (lr . 0.002)
                 (seed . 0)
                 (total . 1175)
                 (warmup . 294)
                 (weight-decay . 0.05)))
        (cuda . "13.0")
        (data . "Fashion-MNIST: 60000 train, 10000 test")
        (date . "2026-11-16")
        (device . "cuda")
        (hardware . "NVIDIA GeForce RTX 3090 Ti")
        (lightning
         .
         #hasheq((accuracy . 0.9122)
                 (epochs
                  .
                  ((1 0.5281 0.3424 0.8776 2.3)
                   (2 0.3201 0.3229 0.8839 1.8)
                   (3 0.2521 0.279 0.897 1.7)
                   (4 0.2053 0.2441 0.9113 1.6)
                   (5 0.1711 0.2371 0.9122 1.9)))
                 (seconds . 15.4)))
        (lightning-version . "2.6.5")
        (model
         .
         "two conv blocks (3x3 conv, batch norm, ReLU, 2x2 max pool), Linear 1568-128-10")
        (precision . "16-mixed")
        (python . "3.14.4")
        (python-torch . "2.14.0+cu130")
        (racket
         .
         #hasheq((accuracy . 0.9122)
                 (epochs
                  .
                  ((1 0.5281 0.3424 0.8776 1.9)
                   (2 0.3201 0.3229 0.8839 1.3)
                   (3 0.2521 0.279 0.897 1.3)
                   (4 0.2053 0.2441 0.9113 1.5)
                   (5 0.1711 0.2371 0.9122 1.4)))
                 (seconds . 11.4)))
        (racket-version . "9.3")
        (seed . 0)
        (torch . "2.14.0"))
