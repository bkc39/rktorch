#lang racket/base

(require (only-in ffi/unsafe _fun _int64 _stdbool)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-hann-window/raw
         tr-stft/raw)

(define-torch tr-hann-window/raw
  (_fun (window-length : _int64)
        (periodic : _stdbool)
        -> _Tensor/null)
  #:c-id tr_hann_window
  #:wrap tensor-allocator)

(define-torch tr-stft/raw
  (_fun (self : _Tensor)
        (n-fft : _int64)
        (hop-length : _int64)
        (win-length : _int64)
        (window : _Tensor/null)
        (center : _stdbool)
        (normalized : _stdbool)
        -> _Tensor/null)
  #:c-id tr_stft
  #:wrap tensor-allocator)
