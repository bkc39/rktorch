#lang racket/base

(require (only-in racket/list remove-duplicates)
         (only-in "markdown.rkt" ref))

(provide arm-blocks
         block-key
         complete-blocks
         outside-clean?)

(define (block-key r)
  (list (ref r 'run_id) (ref r 'round) (ref r 'block)))

(define (arm-blocks arms)
  (for/list ([key (in-list (remove-duplicates (map block-key arms)))])
    (for/hash ([r (in-list arms)] #:when (equal? (block-key r) key))
      (values (ref r 'arm) r))))

(define (outside-clean? r [limit (ref r 'max_load)])
  (define outside (ref r 'load_outside))
  (and (not (ref r 'error))
       (not (and (real? outside) (real? limit) (>= outside limit)))))

(define (complete-blocks blocks #:expected expected #:clean? clean?)
  (define valid
    (filter (lambda (b)
              (and (andmap (lambda (arm) (hash-ref b arm #f)) (expected b))
                   (andmap clean? (hash-values b))))
            blocks))
  (values valid (- (length blocks) (length valid))))

(module+ test
  (require rackunit)

  (define (arm name block #:outside [outside 0.5] #:error [error #f])
    `#hasheq((kind . "arm") (run_id . "r") (round . 0) (block . ,block) (arm . ,name)
             (load_outside . ,outside) (max_load . 2.0) (error . ,error)))

  (test-case "a block is kept whole only with every expected arm present and clean"
    (define blocks
      (arm-blocks (list (arm "rktorch" 0) (arm "pytorch" 0)
                        (arm "rktorch" 1) (arm "pytorch" 1 #:outside 2.5)
                        (arm "rktorch" 2)
                        (arm "rktorch" 3 #:error "died") (arm "pytorch" 3))))
    (define-values (kept dropped)
      (complete-blocks blocks #:expected (lambda (_b) '("rktorch" "pytorch")) #:clean? outside-clean?))
    (check-equal? (map (lambda (b) (ref (hash-ref b "rktorch") 'block)) kept) '(0))
    (check-equal? dropped 3)
    (check-true (outside-clean? (arm "rktorch" 0 #:outside 2.5) 3.0))))
