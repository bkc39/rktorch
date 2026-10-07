#lang racket/base

;; Plain bindings to the staged shim, libtorch and OpenMP for the probes:
;; no allocator wrap, no ledger, no fault latch. The library itself must
;; never bind native code this way (AGENTS.md); these exist so a probe can
;; run an op on the calling thread's own OS thread and free it by hand.

(require (only-in ffi/unsafe
                  _fun _int _int64 _pointer _ptr _string _uint _void
                  ffi-lib get-ffi-obj)
         (only-in ffi/vector _s64vector s64vector))

(provide shim-randn
         shim-add
         shim-matmul
         shim-matmul/blocking
         shim-free
         shim-nbytes
         shim-version
         at-get-num-threads
         at-set-num-threads
         omp-get-max-threads
         omp-set-num-threads
         mkl-get-max-threads
         usleep
         usleep/blocking)

(define shim
  (ffi-lib (build-path (collection-path "torch") "native-libs" "libtorchrkt")))

(define shim-version (get-ffi-obj "tr_version" shim (_fun -> _string)))

(define randn/raw
  (get-ffi-obj "tr_randn" shim (_fun _s64vector _int64 -> _pointer)))

(define (shim-randn . dims)
  (randn/raw (apply s64vector dims) (length dims)))

(define shim-add (get-ffi-obj "tr_add" shim (_fun _pointer _pointer -> _pointer)))

(define shim-matmul
  (get-ffi-obj "tr_matmul" shim (_fun _pointer _pointer -> _pointer)))

(define shim-matmul/blocking
  (get-ffi-obj "tr_matmul" shim (_fun #:blocking? #t _pointer _pointer -> _pointer)))

(define shim-free (get-ffi-obj "tr_tensor_free" shim (_fun _pointer -> _void)))

(define nbytes/raw
  (get-ffi-obj "tr_tensor_nbytes" shim
               (_fun _pointer (out : (_ptr o _int64)) -> (rc : _int)
                     -> (and (zero? rc) out))))

(define (shim-nbytes t) (nbytes/raw t))

;; Looked up through the shim's handle, which reaches the libraries it was
;; linked against: libtorch_cpu everywhere, and on Linux the MKL inside it and
;; libgomp. A symbol a platform lacks binds to a procedure that raises when
;; called, so the probes that never call it still load.
(define (shim-symbol name type)
  (get-ffi-obj name shim type
               (lambda ()
                 (lambda _
                   (error 'probes "~a is not reachable from the shim on this platform"
                          name)))))

(define at-get-num-threads (shim-symbol "_ZN2at15get_num_threadsEv" (_fun -> _int)))

(define at-set-num-threads (shim-symbol "_ZN2at15set_num_threadsEi" (_fun _int -> _void)))

(define mkl-get-max-threads (shim-symbol "mkl_get_max_threads" (_fun -> _int)))

(define omp-get-max-threads (shim-symbol "omp_get_max_threads" (_fun -> _int)))

(define omp-set-num-threads (shim-symbol "omp_set_num_threads" (_fun _int -> _void)))

(define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))

(define usleep/blocking
  (get-ffi-obj "usleep" #f (_fun #:blocking? #t _uint -> _int)))
