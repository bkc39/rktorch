#lang racket/base

(require (only-in ffi/unsafe _fun _int _int32 _int64 _path _ptr)
         (only-in ffi/vector _s32vector)
         (only-in "memory.rkt" tensor-allocator)
         (only-in "syntax.rkt" _Tensor _Tensor/null define-torch))

(provide tr-audio-info/raw
         tr-audio-load/raw
         tr-audio-save/raw)

(define-torch tr-audio-info/raw
  (_fun (path : _path)
        (frames : (_ptr o _int64))
        (rate : (_ptr o _int32))
        (channels : (_ptr o _int32))
        -> (rc : _int)
        -> (values rc frames rate channels))
  #:c-id tr_audio_info)

;; rate rides an s32vector out-slot: a (values handle rate) return would
;; break tensor-allocator, which registers the single pointer result
(define-torch tr-audio-load/raw
  (_fun (path : _path)
        (frame-offset : _int64)
        (num-frames : _int64)
        (rate-out : _s32vector)
        -> _Tensor/null)
  #:c-id tr_audio_load
  #:wrap tensor-allocator)

(define-torch tr-audio-save/raw
  (_fun (path : _path)
        (samples : _Tensor)
        (rate : _int32)
        -> _int)
  #:c-id tr_audio_save)
