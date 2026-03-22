#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>   // For sqrtf in weight initialization
#include <time.h>   // For seeding random numbers with system time

// Defines a memory pool structure to manage a single large block of RAM
typedef struct {
    size_t size;    // Total capacity of the memory pool in bytes
    size_t offset;  // The current "high-water mark" tracking used bytes
    uint8_t *buffer; // Pointer to the start of the contiguous heap memory
} Arena;

// Implements Xavier Initialization for weight matrices to ensure stable signal flow
void initialize_weights(float *w, int rows, int cols) {
    // Range formula based on the number of inputs and outputs of the layer
    float range = sqrtf(6.0f / (rows + cols)); 

    for (int i = 0; i < rows * cols; i++) {
        // Generates a random float scaled to a -1.0 to 1.0 distribution
        float r = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        // Assigns the scaled random value to the current matrix element
        w[i] = r * range;
    }
}

// Requests a large block of memory from the OS to initialize the Arena
Arena* arena_init(size_t capacity) {
    // Allocates the control structure itself
    Arena *a = malloc(sizeof(Arena));
    a->size = capacity;
    a->offset = 0;
    // Allocates the actual raw memory buffer that the Arena will manage
    a->buffer = malloc(capacity);
    // Verifies that the Operating System successfully provided the requested RAM
    if (!a->buffer) {
        perror("Arena buffer allocation failed");
        exit(1);
    }
    return a;
}

// Claims a specific slice of the Arena by advancing the offset pointer
void* arena_alloc(Arena *a, size_t size) {
    // Aligns the request to 8 bytes for CPU efficiency (ensures word-alignment)
    size_t aligned_size = (size + 7) & ~7;
    // Prevents reading or writing outside the Arena's physical bounds
    if (a->offset + aligned_size > a->size) return NULL;
    // Gets the memory address at the current offset
    void *ptr = &a->buffer[a->offset];
    // Moves the offset forward so the next allocation starts after this one
    a->offset += aligned_size;
    return ptr;
}

// Converts 32-bit Big-Endian (file format) to Little-Endian (CPU format)
uint32_t swap_endian(uint32_t val) {
    return ((val << 24)               | 
            ((val << 8) & 0x00FF0000) | 
            ((val >> 8) & 0x0000FF00) | 
            (val >> 24));
}

int main() {
    // Initializes the random number generator using the system clock
    srand(time(NULL));
    // Pre-allocates a 256MB master block of memory for the whole application
    Arena *train_arena = arena_init(256 * 1024 * 1024);

    // Opens the MNIST image and label binary files in Read-Binary mode
    FILE *img_file = fopen("data/train-images.idx3-ubyte", "rb");
    FILE *lbl_file = fopen("data/train-labels.idx1-ubyte", "rb");

    // Validates that files were found to prevent null pointer crashes
    if (!img_file || !lbl_file) {
        printf("Error: Could not find MNIST files in data/ folder.\n");
        return 1;
    }

    // Declares variables to hold the metadata from the file headers
    uint32_t magic, count, rows, cols;
    
    // Reads the first four 32-bit integers from the image file header
    fseek(img_file, 0, SEEK_SET);
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows, 4, 1, img_file);
    fread(&cols, 4, 1, img_file);

    // Corrects the byte-order of the metadata for the local CPU
    count = swap_endian(count);
    rows  = swap_endian(rows);
    cols  = swap_endian(cols);

    // Sets file cursors to where actual data begins (skipping headers)
    fseek(img_file, 16, SEEK_SET);
    fseek(lbl_file, 8, SEEK_SET);

    // Reserves a large block in the Arena for 60,000 raw image arrays
    size_t img_size = (size_t)count * rows * cols;
    uint8_t *images_data = (uint8_t*)arena_alloc(train_arena, img_size);
    // Reserves a block in the Arena for 60,000 classification labels
    uint8_t *labels_data = (uint8_t*)arena_alloc(train_arena, count);

    // Ensures the Arena had enough space for the dataset loading
    if (!images_data || !labels_data) {
        printf("Arena out of memory!\n");
        return 1;
    }

    // Loads the entire image dataset from disk into the Arena in one operation
    fread(images_data, 1, img_size, img_file);
    // Loads the entire label dataset from disk into the Arena in one operation
    fread(labels_data, 1, count, lbl_file);

    // Defines which image in the set we want to view (0 to 59,999)
    int target_idx = 0; 
    // Prints the numerical answer (label) for the chosen image
    printf("Label: %u\n", labels_data[target_idx]);
    printf("+------------------------------------------------------+\n");
    // Nested loop to process the 28x28 2D grid of pixels
    for (int y = 0; y < 28; y++) {
        printf("|"); // Draws the left border of the frame
        for (int x = 0; x < 28; x++) {
            // Maps the 3D index (Image, Row, Col) into the flat 1D memory space
            uint8_t pixel = images_data[target_idx * 784 + y * 28 + x];
            // Prints dense characters for bright pixels and light ones for dark pixels
            if (pixel > 150)      printf("@@"); 
            else if (pixel > 50)  printf("..");
            else                  printf("  ");
        }
        printf("|\n"); // Draws the right border and moves to the next line
    }
    printf("+------------------------------------------------------+\n");

    // Allocates memory for the weight matrix connecting Input (784) to Hidden (128)
    float *w_ih = (float*)arena_alloc(train_arena, 784 * 128 * sizeof(float));
    // Allocates memory for the weight matrix connecting Hidden (128) to Output (10)
    float *w_ho = (float*)arena_alloc(train_arena, 128 * 10 * sizeof(float));

    // Allocates memory for the bias vectors of the two network layers
    float *b_h = (float*)arena_alloc(train_arena, 128 * sizeof(float));
    float *b_o = (float*)arena_alloc(train_arena, 10 * sizeof(float));

    // Validates that the weights successfully claimed space in the Arena
    if (!w_ih || !w_ho || !b_h || !b_o) {
        printf("Failed to allocate weights in Arena!\n");
        return 1;
    }

    // Randomizes the weights using Xavier math to prepare for training
    initialize_weights(w_ih, 784, 128);
    initialize_weights(w_ho, 128, 10);

    // Loops to initialize the hidden layer biases to a starting value of zero
    for(int i=0; i<128; i++) b_h[i] = 0.0f;
    // Loops to initialize the output layer biases to a starting value of zero
    for(int i=0; i<10; i++)  b_o[i] = 0.0f;

    printf("\nNeural Network weights initialized successfully.\n");
    printf("\n--- Weight Initialization Test ---\n");

    // Prints specific indices to prove the matrix is populated with floats
    printf("Weight w_ih[0]: %f\n", w_ih[0]);
    printf("Weight w_ih[50000]: %f\n", w_ih[50000]);
    printf("Weight w_ih[100351]: %f\n", w_ih[100351]);

    // Accumulates the sum of all weights to check for statistical bias
    float sum = 0;
    int total_w = 784 * 128;
    for(int i = 0; i < total_w; i++) sum += w_ih[i];
    // A mean near zero indicates a healthy, non-biased initialization
    printf("Average weight value: %f (Should be near 0.0)\n", sum / total_w);

    // Displays current Arena consumption (MB used vs MB reserved)
    printf("Arena Space Used: %zu MB / %zu MB\n", 
            train_arena->offset / (1024 * 1024), 
            train_arena->size / (1024 * 1024));

    // Closes the open file descriptors to avoid resource leaks
    fclose(img_file);
    fclose(lbl_file);
    
    // Terminates the program and returns control to the shell
    return 0;
}