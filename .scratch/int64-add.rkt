#lang racket/base
(require torch)
(define c (to-dtype (tensor 16777216) 'int64))
(printf "scalar add -> dtype ~a, value ~a\n"
        (tensor-dtype (add c 1)) (item (to-dtype (add c 1) 'int64)))
(printf "tensor add -> dtype ~a, value ~a\n"
        (tensor-dtype (add c (ones-like c))) (item (add c (ones-like c))))
(printf "ones-like dtype ~a\n" (tensor-dtype (ones-like c)))
