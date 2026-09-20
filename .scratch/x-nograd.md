Correct, and torchvision agrees: `make_grid` carries `@torch.no_grad()`.
Without it every `copy!` extended the caller's graph into a picture of
the batch. The whole body, singleton path included, is under
`with-no-grad` now, with a test that a grid of a `#:requires-grad?` batch
does not itself require grad.
