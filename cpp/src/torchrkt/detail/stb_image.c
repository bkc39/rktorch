#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#ifndef STBI_THREAD_LOCAL
#error "stb_image's failure reason must be thread-local, like tr_last_error"
#endif
