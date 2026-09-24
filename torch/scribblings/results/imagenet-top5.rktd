#hasheq((date . "2026-09-24")
        (device . "cuda")
        (model . "resnet18, torchvision IMAGENET1K_V1")
        (photos
         .
         (#hasheq((path . ("ants" "11381045_b352a47d8c.jpg"))
                  (top5
                   .
                   (("ant" . 0.689)
                    ("leaf beetle" . 0.115)
                    ("ladybug" . 0.0766)
                    ("hip" . 0.0423)
                    ("fly" . 0.0412))))
          #hasheq((path . ("ants" "8398478_50ef10c47a.jpg"))
                  (top5
                   .
                   (("centipede" . 0.2594)
                    ("tick" . 0.2385)
                    ("barn spider" . 0.1233)
                    ("ant" . 0.0406)
                    ("garden spider" . 0.03))))
          #hasheq((path . ("bees" "10870992_eebeeb3a12.jpg"))
                  (top5
                   .
                   (("bee" . 0.5229)
                    ("cricket" . 0.1857)
                    ("grasshopper" . 0.0672)
                    ("ant" . 0.047)
                    ("sulphur butterfly" . 0.0197))))
          #hasheq((path . ("bees" "26589803_5ba7000313.jpg"))
                  (top5
                   .
                   (("bee" . 0.7668)
                    ("fly" . 0.222)
                    ("leafhopper" . 0.0045)
                    ("weevil" . 0.0027)
                    ("lacewing" . 0.0008))))))
        (torch . "2.9.0")
        (weights . "resnet18-imagenet1k-v1.safetensors"))
