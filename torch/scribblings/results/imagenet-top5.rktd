#hasheq((date . "2026-09-26")
        (device . "cuda")
        (model . "resnet18, torchvision IMAGENET1K_V1")
        (photos
         .
         (#hasheq((path . ("ants" "formica-rufa.jpg"))
                  (top5
                   .
                   (("ant" . 0.997)
                    ("black widow" . 0.0015)
                    ("tick" . 0.0006)
                    ("fiddler crab" . 0.0003)
                    ("scorpion" . 0.0002))))
          #hasheq((path . ("ants" "hedge-mustard.jpg"))
                  (top5
                   .
                   (("rapeseed" . 0.6145)
                    ("ant" . 0.12)
                    ("buckeye" . 0.0444)
                    ("bee" . 0.0297)
                    ("fig" . 0.0233))))
          #hasheq((path . ("bees" "coneflower.jpg"))
                  (top5
                   .
                   (("bee" . 0.8516)
                    ("sulphur butterfly" . 0.0302)
                    ("admiral" . 0.0226)
                    ("monarch" . 0.0198)
                    ("ringlet" . 0.0115))))
          #hasheq((path . ("bees" "honey-bee.jpg"))
                  (top5
                   .
                   (("bee" . 0.9988)
                    ("cardoon" . 0.0004)
                    ("fly" . 0.0003)
                    ("grasshopper" . 0.0001)
                    ("ant" . 0.0001))))))
        (torch . "2.9.0")
        (weights . "resnet18-imagenet1k-v1.safetensors"))
