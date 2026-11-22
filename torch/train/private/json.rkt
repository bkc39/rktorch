#lang racket/base

(require (only-in json jsexpr->string string->jsexpr)
         (only-in racket/match match))

(provide json-text
         text->json)

;; `string->jsexpr` reads -0.0 as 0.0, so numbers are read here; objects are
;; written with their keys sorted, so equal trees give equal bytes
(define (json-text j)
  (define out (open-output-string))
  (let emit ([j j])
    (match j
      [#t (write-string "true" out)]
      [#f (write-string "false" out)]
      [(? string?) (write-string (jsexpr->string j) out)]
      [(? number?) (write-string (number->string j) out)]
      [(? list?)
       (write-string "[" out)
       (for ([x (in-list j)] [i (in-naturals)])
         (unless (zero? i) (write-string "," out))
         (emit x))
       (write-string "]" out)]
      [(? hash?)
       (write-string "{" out)
       (for ([k (in-list (sort (hash-keys j) symbol<?))] [i (in-naturals)])
         (unless (zero? i) (write-string "," out))
         (write-string (jsexpr->string (symbol->string k)) out)
         (write-string ":" out)
         (emit (hash-ref j k)))
       (write-string "}" out)]))
  (get-output-string out))

(define number-pattern #px"^-?(?:0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?")

(define (text->json s)
  (define n (string-length s))
  (define i 0)
  (define (fail what)
    (raise-arguments-error 'read-checkpoint "malformed checkpoint JSON"
                           "expected" what
                           "position" i))
  (define (peek) (if (< i n) (string-ref s i) #\nul))
  (define (skip!)
    (when (and (< i n) (memv (string-ref s i) '(#\space #\tab #\newline #\return)))
      (set! i (add1 i))
      (skip!)))
  (define (expect! c)
    (skip!)
    (unless (char=? (peek) c) (fail (string c)))
    (set! i (add1 i)))
  (define (literal! word v)
    (define end (+ i (string-length word)))
    (unless (and (<= end n) (string=? (substring s i end) word)) (fail word))
    (set! i end)
    v)
  (define (string!)
    (define start i)
    (let scan ([k (add1 i)])
      (cond
        [(>= k n) (fail "a closing quote")]
        [(char=? (string-ref s k) #\\) (scan (+ k 2))]
        [(char=? (string-ref s k) #\") (set! i (add1 k))]
        [else (scan (add1 k))]))
    (string->jsexpr (substring s start i)))
  (define (number!)
    (match (regexp-match number-pattern s i)
      [#f (fail "a value")]
      [(list text fraction exponent)
       (set! i (+ i (string-length text)))
       (if (or fraction exponent)
           (string->number text 10 'number-or-false 'decimal-as-inexact)
           (string->number text))]))
  (define (sequence! close item!)
    (set! i (add1 i))
    (skip!)
    (cond
      [(char=? (peek) close) (set! i (add1 i)) '()]
      [else
       (let more ([acc (list (item!))])
         (skip!)
         (define c (peek))
         (set! i (add1 i))
         (cond
           [(char=? c close) (reverse acc)]
           [(char=? c #\,) (more (cons (item!) acc))]
           [else (fail (format "`,` or `~a`" close))]))]))
  (define (member!)
    (skip!)
    (unless (char=? (peek) #\") (fail "a key"))
    (define k (string->symbol (string!)))
    (expect! #\:)
    (cons k (value!)))
  (define (value!)
    (skip!)
    (match (peek)
      [#\{ (make-immutable-hasheq (sequence! #\} member!))]
      [#\[ (sequence! #\] value!)]
      [#\" (string!)]
      [#\t (literal! "true" #t)]
      [#\f (literal! "false" #f)]
      [_ (number!)]))
  (begin0 (value!)
    (skip!)
    (unless (= i n) (fail "the end of the text"))))
