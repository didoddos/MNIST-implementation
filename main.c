#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//we're going to use arena allocation instead of malloc and free
// reversing big-endian to little-endian
uint32_t swap_endian(uint32_t val) {
    return ((val << 24)               | 
            ((val << 8) & 0x00FF0000) | 
            ((val >> 8) & 0x0000FF00) | 
            (val >> 24));
}
typedef struct{
    size_t size;
    size_t offset;
    uint8_t *buffer;
}Arena;

Arena* arena_init(size_t capacity) {
    Arena *a = malloc(sizeof(Arena));
    a->size = capacity;
    a->offset = 0;
    a->buffer = malloc(capacity);
    if (!a->buffer) {
        perror("Failed to allocate Arena buffer");
        exit(1);
    }
    return a;
}
void* arena_alloc(Arena *a, size_t size) {
    // 8-byte Alignment (Essential for SIMD/Modern CPUs)
    size_t aligned_size = (size + 7) & ~7;
    
    if (a->offset + aligned_size > a->size) {
        fprintf(stderr, "Arena out of memory!\n");
        return NULL;
    }
    
    void *ptr = &a->buffer[a->offset];
    a->offset += aligned_size;
    return ptr;
}

void arena_reset(Arena *a, size_t save_point) {
    a->offset = save_point;
}

void arena_destroy(Arena *a) {
    free(a->buffer);
    free(a);
}



// using pragma as a precaution
#pragma pack(push, 1)
typedef struct {
    uint32_t magic_number;
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
} MNIST_Header;
#pragma pack(pop)

int main() {
    // opening the file and reading it in binary mode
    FILE *file = fopen("data/train-images.idx3-ubyte", "rb");
    
    if (file == NULL) {
        perror("Error: Could not open the file. Is it in the /data folder?");
        return 1;
    }

    // reading the data into our struct and packing it tightly
    MNIST_Header header;
    size_t read_check = fread(&header, sizeof(MNIST_Header), 1, file);

    if (read_check != 1) {
        printf("Error: Failed to read the header.\n");
        fclose(file);
        return 1;
    }

    // swapping the bytes that we are reading from big to little-endian using the function above
    uint32_t magic = swap_endian(header.magic_number);
    uint32_t count = swap_endian(header.num_images);
    uint32_t rows  = swap_endian(header.num_rows);
    uint32_t cols  = swap_endian(header.num_cols);

    // Sanity check to make sure everything is working
    printf("--- MNIST Dataset Sanity Check ---\n");
    printf("Magic Number: %u (Should be 2051)\n", magic);
    printf("Number of Images: %u (Should be 60000)\n", count);
    printf("Resolution: %u x %u pixels\n", rows, cols);
    printf("----------------------------------\n");

    fclose(file);
    return 0;
}