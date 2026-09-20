Right — `#:init` reads indices 0 through 3, so anything shorter raised
`list-ref` from inside the constructor rather than blaming the caller.
`#:blocks` is now a `list/c` of exactly four positive integers, the
docs say the same, and nn-contract-test.rkt checks that three and five
are both contract violations naming `ResNet`. Fixed in 7c82f74.
