# stb_image

`stb_image.h` v2.30, vendored unmodified from
<https://github.com/nothings/stb> at commit
`013ac3beddff3dbffafd5177e7972067cd2b5083` (2024-05-31).

sha256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3`

Licence: MIT or public domain (the Unlicense), at the user's choice; the
full text is at the end of the header.

`src/torchrkt/detail/stb_image.c` compiles the implementation with only
the JPEG and PNG decoders and without stdio: the shim hands it bytes, so
`tr_image_decode` (`include/torchrkt/c_api/image.h`) decodes a file and an
archive member the same way.

To update, replace the header from a newer commit, record the commit and
checksum here, and rerun the image gtests and the reader's parity cases.
