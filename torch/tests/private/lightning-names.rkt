#lang racket/base

(require (only-in racket/file file->value)
         (only-in racket/list filter-map last)
         (only-in racket/match match)
         (only-in racket/runtime-path #%datum define-runtime-path)
         (only-in racket/string string-replace)
         (only-in "../../train/private/hooks.rkt" log-rule))

(provide expected-events
         fixture
         lightning->rktorch
         log-permissions
         teardown-after-exception)

(define-runtime-path fixture-path "../lightning-hooks.rktd")

(define fixture (file->value fixture-path))

(define unsupported-hooks
  '(transfer_batch_to_device configure_model configure_sharded_model
    prepare_data optimizer_step optimizer_zero_grad backward
    configure_gradient_clipping clip_gradients lr_scheduler_step forward))

(define (unsupported-hook? name)
  (or (memq name unsupported-hooks)
      (regexp-match? #rx"^on_(validation|test|predict)_model_|_dataloader$"
                     (symbol->string name))))

(define (hyphenated name)
  (match name
    ['load_state_dict 'load-state!]
    [_ (string->symbol (string-replace (symbol->string name) "_" "-"))]))

(define (lightning->rktorch name)
  (match (regexp-match #rx"^Callback[.](.*)$" (symbol->string name))
    [(list _ hook) (and (not (unsupported-hook? (string->symbol hook)))
                        (list 'callback (hyphenated (string->symbol hook))))]
    [#f (and (not (unsupported-hook? name)) (list 'trainer (hyphenated name)))]))

(define (teardown-after-exception events)
  (match (last events)
    [(list 'callback 'on-exception epoch step _ lr)
     (append events
             (list (list 'callback 'teardown epoch step #f lr)
                   (list 'trainer 'teardown epoch step #f lr)))]
    [_ events]))

(define deviations
  (hasheq 'exception teardown-after-exception))

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
               #:unless (unsupported-hook? name))
    (match (lightning->rktorch name)
      [(list _ hook) (values hook (and opts (apply log-rule opts)))])))
