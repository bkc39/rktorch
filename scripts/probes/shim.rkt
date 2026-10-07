#lang racket/base

;; Plain bindings to the staged shim, libtorch and libgomp for the probes:
;; no allocator wrap, no ledger, no fault latch. The library itself must
;; never bind native code this way (AGENTS.md); these exist so a probe can
;; run an op on the calling thread's own OS thread and free it by hand.

(require (only-in ffi/unsafe
                  _fun _int _int64 _pointer _ptr _string _uint _void
                  ffi-lib get-ffi-obj)
         (only-in ffi/vector _s64vector s64vector)
         (only-in racket/file file->lines)
         (only-in racket/list last)
         (only-in racket/string string-split))

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

(define (loaded-library-path name)
  (for/first ([line (in-list (file->lines "/proc/self/maps"))]
              #:when (regexp-match? (regexp-quote name) line))
    (last (string-split line))))

(define libtorch-cpu (ffi-lib (loaded-library-path "libtorch_cpu.so")))
(define libgomp (ffi-lib (loaded-library-path "libgomp")))

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

(define at-get-num-threads
  (get-ffi-obj "_ZN2at15get_num_threadsEv" libtorch-cpu (_fun -> _int)))

(define at-set-num-threads
  (get-ffi-obj "_ZN2at15set_num_threadsEi" libtorch-cpu (_fun _int -> _void)))

(define mkl-get-max-threads
  (get-ffi-obj "mkl_get_max_threads" libtorch-cpu (_fun -> _int)))

(define omp-get-max-threads
  (get-ffi-obj "omp_get_max_threads" libgomp (_fun -> _int)))

(define omp-set-num-threads
  (get-ffi-obj "omp_set_num_threads" libgomp (_fun _int -> _void)))

(define usleep (get-ffi-obj "usleep" #f (_fun _uint -> _int)))

(define usleep/blocking
  (get-ffi-obj "usleep" #f (_fun #:blocking? #t _uint -> _int)))
