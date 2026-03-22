#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>   // Required for sqrtf in weight initialization
#include <time.h>   // Required for seeding the random number generator

// Structure to manage a contiguous block of pre-allocated memory
typedef struct {
    size_t size;    // Total capacity of the allocated pool
    size_t offset;  // Cursor tracking the next available byte in the pool
    uint8_t *buffer; // Pointer to the start of the memory block
} Arena;

// Fills a weight matrix using Xavier Initialization to prevent signal vanishing/explosion
void initialize_weights(float *w, int rows, int cols) {
    // Calculate the variance scaling factor based on layer dimensions
    float range =  sqrtf(6.0f / (rows + cols)); 

    for (int i = 0; i < rows * cols; i++) {
        // Generate a normalized random float between -1.0 and 1.0
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // Scale the random value by the calculated range for mathematical stability
        w[i] = r * range;
    }
}

// Allocates the master memory block from the OS to be managed by the Arena
Arena* arena_init(size_t capacity) {
    // Allocate the management structure on the heap
    Arena *a = malloc(sizeof(Arena));
    a->size = capacity;
    a->offset = 0;
    // Allocate the actual data pool where all project variables will reside
    a->buffer = malloc(capacity);
    // Safety check to ensure the Operating System granted the memory request
    if (!a->buffer) {
        perror("Arena buffer allocation failed");
        exit(1);
    }
    return a;
}

// "Allocates" a slice of memory by simply moving the Arena's internal pointer
void* arena_alloc(Arena *a, size_t size) {
    // Align the request to 8 bytes to ensure CPU word-alignment for performance
    size_t aligned_size = (size + 7) & ~7;
    // Prevent buffer overflow by checking remaining capacity
    if (a->offset + aligned_size > a->size) return NULL;
    // Calculate the physical address of the newly claimed memory slice
    void *ptr = &a->buffer[a->offset];
    // Advance the offset so the next allocation starts after this one
    a->offset += aligned_size;
    return ptr;
}

// Reverses byte order to convert Big-Endian file data to Little-Endian CPU format
uint32_t swap_endian(uint32_t val) {
    return ((val << 24)               | 
            ((val << 8) & 0x00FF0000) | 
            ((val >> 8) & 0x0000FF00) | 
            (val >> 24));
}

int main() {
    // Seed the random number generator with the current system time
    srand(time(NULL));
    // Reserve a large, contiguous memory block for the entire application
    Arena *train_arena = arena_init(256 * 1024 * 1024);

    // Open the binary image and label files for reading
    FILE *img_file = fopen("data/train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");

    // Ensure the data files exist before proceeding
    if (!img_file || !lbl_file) {
        printf("Error: Could not find MNIST files in data/ folder.\n");
        return 1;
    }

    // Temporary variables to store dataset dimensions from the file headers
    uint32_t magic, count, rows, cols;
    
    // Position file pointer at the start and read the 4-byte metadata fields
    fseek(img_file, 0, SEEK_SET);
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows, 4, 1, img_file);
    fread(&cols, 4, 1, img_file);

    // Convert dimensions from Big-Endian to the CPU's native Little-Endian format
    count = swap_endian(count);
    rows  = swap_endian(rows);
    cols  = swap_endian(cols);

    // Skip the fixed-size headers to reach the actual raw data bytes
    fseek(img_file, 16, SEEK_SET);
    fseek(lbl_file, 8, SEEK_SET);

    // Calculate total size required for images and allocate from the Arena
    size_t img_size = (size_t)count * rows * cols;
    uint8_t *images_data = (uint8_t*)arena_alloc(train_arena, img_size);
    // Allocate space for the classification labels in the Arena
    uint8_t *labels_data = (uint8_t*)arena_alloc(train_arena, count);

    // Check if the Arena had enough room for the dataset
    if (!images_data || !labels_data) {
        printf("Arena out of memory!\n");
        return 1;
    }

    // Stream the raw pixel data from the disk into the Arena's memory
    fread(images_data, 1, img_size, img_file);
    // Stream the label data from the disk into the Arena's memory
    fread(labels_data, 1, count, lbl_file);

    // Select the very first image in the dataset to test the visualizer
    int target_idx = 0; 
    // Display the expected digit value for the chosen image
    printf("Label: %u\n", labels_data[target_idx]);
    printf("+------------------------------------------------------+\n");
    // Iterate through the rows and columns of the selected 28x28 image
    for (int y = 0; y < 28; y++) {
        printf("|");
        for (int x = 0; x < 28; x++) {
            // Index into the flat pixel array using the (ImageIndex * Size + Row * Width + Col) formula
            uint8_t pixel = images_data[target_idx * 784 + y * 28 + x];
            // Threshold pixels to create high-contrast ASCII art (bright = @, dim = ., black = space)
            if (pixel > 150)      printf("@@"); 
            else if (pixel > 50)  printf("..");
            else                  printf("  ");
        }
        printf("|\n");
    }
    printf("+------------------------------------------------------+\n");

    // Claim memory for the first layer's weight matrix (Inputs to Hidden)
    float *w_ih = (float*)arena_alloc(train_arena, 784 * 128 * sizeof(float));
    // Claim memory for the second layer's weight matrix (Hidden to Outputs)
    float *w_ho = (float*)arena_alloc(train_arena, 128 * 10 * sizeof(float));

    // Claim memory for the bias vectors of both neural layers
    float *b_h = (float*)arena_alloc(train_arena, 128 * sizeof(float));
    float *b_o = (float*)arena_alloc(train_arena, 10 * sizeof(float));

    // Safety check for network weight allocation
    if (!w_ih || !w_ho || !b_h || !b_o) {
        printf("Failed to allocate weights in Arena!\n");
        return 1;
    }

    // Fill weight matrices with optimized random numbers to start the learning process
    initialize_weights(w_ih, 784, 128);
    initialize_weights(w_ho, 128, 10);

    // Initialize the hidden layer biases to a neutral zero state
    for(int i=0; i<128; i++) b_h[i] = 0.0f;
    // Initialize the output layer biases to a neutral zero state
    for(int i=0; i<10; i++)  b_o[i] = 0.0f;

    printf("\nNeural Network weights initialized successfully.\n");

    // Close file pointers to release system resources
    fclose(img_file);
    fclose(lbl_file);
    
    // Return success code to the Operating System
    return 0;
}