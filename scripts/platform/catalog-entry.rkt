#lang racket/base
(require (only-in racket/file file->string)
         (only-in racket/match match-define)
         (only-in racket/string string-trim))

(match-define (vector name source checksum-file) (current-command-line-arguments))

(write (hash 'name name
             'source source
             'checksum (string-trim (file->string checksum-file))
             'author ""
             'description ""
             'tags '()
             'dependencies '()
             'modules '()))
(newline)
