#lang racket/base

(require (only-in racket/file file->value)
         (only-in racket/list filter-map last)
         (only-in racket/match match)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-replace))

(provide d2-teardown-after-exception
         expected-events
         fixture
         lightning->rktorch
         log-permissions)

(define-runtime-path fixture-path "../lightning-hooks.rktd")

(define fixture (file->value fixture-path))

(define d10
  '(transfer_batch_to_device configure_model configure_sharded_model
    prepare_data optimizer_step optimizer_zero_grad backward
    configure_gradient_clipping clip_gradients lr_scheduler_step forward))

(define (d10? name)
  (or (memq name d10)
      (regexp-match? #rx"^on_(validation|test|predict)_model_|_dataloader$"
                     (symbol->string name))))

(define (hyphenated name)
  (match name
    ['load_state_dict 'load-state!]
    [_ (string->symbol (string-replace (symbol->string name) "_" "-"))]))

(define (lightning->rktorch name)
  (match (regexp-match #rx"^Callback[.](.*)$" (symbol->string name))
    [(list _ hook) (and (not (d10? (string->symbol hook)))
                        (list 'callback (hyphenated (string->symbol hook))))]
    [#f (and (not (d10? name)) (list 'trainer (hyphenated name)))]))

(define (d2-teardown-after-exception events)
  (match (last events)
    [(list 'callback 'on-exception epoch step _ lr)
     (append events
             (list (list 'callback 'teardown epoch step #f lr)
                   (list 'trainer 'teardown epoch step #f lr)))]
    [_ events]))

(define deviations
  (hasheq 'exception d2-teardown-after-exception))

(define (expected-events scenario)
  (define events
    (filter-map (lambda (e)
                  (match e
                    [(list name epoch step batch lr)
                     (define mapped (lightning->rktorch name))
                     (and mapped (append mapped (list epoch step batch lr)))]))
                (hash-ref (hash-ref fixture 'scenarios) scenario)))
  ((hash-ref deviations scenario (lambda () values)) events))

(define (log-permissions)
  (for/hasheq ([(name opts) (in-hash (hash-ref fixture 'log-permissions))]
               #:unless (d10? name))
    (values (cadr (lightning->rktorch name))
            (match opts
              [#f #f]
              [(list '(#f #t) '(#f #t) #t #f) 'step]
              [(list '(#f #t) '(#f #t) #f #t) 'epoch]
              [(list '(#f) '(#t) #f #t) 'epoch-only]))))
