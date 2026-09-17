# The REPL recording at the top of the README's Usage section. Regenerate from
# the repository root, inside `nix develop`. No dev shell ships VHS; the flake's
# pinned nixpkgs has 0.11.0, the version this needs (0.12.0 exits without
# writing the GIF: charmbracelet/vhs#787):
#   nix shell --inputs-from . nixpkgs#vhs --command vhs docs/demo/repl.tape

Output docs/images/repl.gif

Set Shell "bash"
Set Theme "GitHub Dark"
Set FontSize 20
Set Width 960
Set Height 620
Set Padding 24
Set WindowBar Colorful
Set TypingSpeed 40ms
Set WaitTimeout 60s

Hide
Type "clear && racket"
Enter
Wait />$/
Show

Sleep 500ms
Type "(require torch torch/nn)"
Enter
Wait />$/
Type "(manual-seed! 0)"
Enter
Wait />$/
Sleep 300ms

Type "(define a (randn 2 3))"
Enter
Wait />$/
Type "a"
Enter
Wait />$/
Sleep 1.5s

Type "(@ a (T a))"
Enter
Wait />$/
Sleep 1.5s

Type `(tensor #"\0\1\2\377")`
Enter
Wait />$/
Sleep 1.5s

Type "(define x (tensor '(1.0 2.0 3.0) #:requires-grad? #t))"
Enter
Wait />$/
Type "(backward! (Σ (* x x)))"
Enter
Wait />$/
Type "(grad x)"
Enter
Wait />$/
Sleep 1.5s

Type "(define fc (Linear 3 2))"
Enter
Wait />$/
Type "(fc a)"
Enter
Wait />$/
Sleep 4s
