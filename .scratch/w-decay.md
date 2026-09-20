Correct, and `torch.optim` does reject it: every one of SGD, Adam and
RMSprop raises `ValueError: Invalid weight_decay value: -0.1`. All three
now take `(>=/c 0)`, with a test that each refuses a negative. Fixed in
5f93371.
