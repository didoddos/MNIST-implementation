#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// --- Memory Management ---
typedef struct {
    size_t size;
    size_t offset;
    uint8_t *buffer;
} Arena;

Arena* arena_init(size_t capacity) {
    Arena *a = malloc(sizeof(Arena));
    a->size = capacity;
    a->offset = 0;
    a->buffer = malloc(capacity);
    if (!a->buffer) {
        perror("Arena buffer allocation failed");
        exit(1);
    }
    return a;
}

void* arena_alloc(Arena *a, size_t size) {
    size_t aligned_size = (size + 7) & ~7;
    if (a->offset + aligned_size > a->size) return NULL;
    void *ptr = &a->buffer[a->offset];
    a->offset += aligned_size;
    return ptr;
}

// --- Utility Functions ---
uint32_t swap_endian(uint32_t val) {
    return ((val << 24)               | 
            ((val << 8) & 0x00FF0000) | 
            ((val >> 8) & 0x0000FF00) | 
            (val >> 24));
}

int main() {
    // 1. Initialize Arena (128MB)
    Arena *train_arena = arena_init(128 * 1024 * 1024);

    // 2. Open Files
    FILE *img_file = fopen("data/train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");

    if (!img_file || !lbl_file) {
        printf("Error: Could not find MNIST files in data/ folder.\n");
        return 1;
    }

    // 3. Read Metadata (Manual read to avoid struct padding issues)
    uint32_t magic, count, rows, cols;
    
    fseek(img_file, 0, SEEK_SET);
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows, 4, 1, img_file);
    fread(&cols, 4, 1, img_file);

    count = swap_endian(count);
    rows  = swap_endian(rows);
    cols  = swap_endian(cols);

    // 4. Validate and Align
    // MNIST images ALWAYS start at byte 16. Labels ALWAYS start at byte 8.
    fseek(img_file, 16, SEEK_SET);
    fseek(lbl_file, 8, SEEK_SET);

    // 5. Load Data
    size_t img_size = (size_t)count * rows * cols;
    uint8_t *images_data = (uint8_t*)arena_alloc(train_arena, img_size);
    uint8_t *labels_data = (uint8_t*)arena_alloc(train_arena, count);

    if (!images_data || !labels_data) {
        printf("Arena out of memory!\n");
        return 1;
    }

    fread(images_data, 1, img_size, img_file);
    fread(labels_data, 1, count, lbl_file);

    // 6. Visualizer (Using the "Double-Wide" trick for better proportion)
    int target_idx = 0; 
    printf("Label: %u\n", labels_data[target_idx]);
    printf("+------------------------------------------------------+\n");
    for (int y = 0; y < 28; y++) {
        printf("|");
        for (int x = 0; x < 28; x++) {
            uint8_t pixel = images_data[target_idx * 784 + y * 28 + x];
            if (pixel > 150)      printf("@@"); // Use 2 chars to make it square
            else if (pixel > 50)  printf("..");
            else                  printf("  ");
        }
        printf("|\n");
    }
    printf("+------------------------------------------------------+\n");

    fclose(img_file);
    fclose(lbl_file);
    return 0;
}