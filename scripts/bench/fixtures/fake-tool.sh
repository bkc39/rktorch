#!/bin/sh
# Stands in for a system tool in system-tools-test.rkt: prints the directory it ran from, then its arguments.
d=${0%/*}
echo "${d##*/} $*"
