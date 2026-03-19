#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//we're going to use arena allocation instead of 
// The "Brain": Swaps Big-Endian to Little-Endian
uint32_t swap_endian(uint32_t val) {
    return ((val << 24)               | 
            ((val << 8) & 0x00FF0000) | 
            ((val >> 8) & 0x0000FF00) | 
            (val >> 24));
}

// The "Mirror": Matches the MNIST file header exactly
#pragma pack(push, 1)
typedef struct {
    uint32_t magic_number;
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
} MNIST_Header;
#pragma pack(pop)

int main() {
    // 1. Open the file in Binary Read mode
    FILE *file = fopen("data/train-images.idx3-ubyte", "rb");
    
    if (file == NULL) {
        perror("Error: Could not open the file. Is it in the /data folder?");
        return 1;
    }

    // 2. Read the 16-byte header into our struct
    MNIST_Header header;
    size_t read_check = fread(&header, sizeof(MNIST_Header), 1, file);

    if (read_check != 1) {
        printf("Error: Failed to read the header.\n");
        fclose(file);
        return 1;
    }

    // 3. Swap the endianness of the values we just read
    uint32_t magic = swap_endian(header.magic_number);
    uint32_t count = swap_endian(header.num_images);
    uint32_t rows  = swap_endian(header.num_rows);
    uint32_t cols  = swap_endian(header.num_cols);

    // 4. Print the results to the terminal
    printf("--- MNIST Dataset Sanity Check ---\n");
    printf("Magic Number: %u (Should be 2051)\n", magic);
    printf("Number of Images: %u (Should be 60000)\n", count);
    printf("Resolution: %u x %u pixels\n", rows, cols);
    printf("----------------------------------\n");

    fclose(file);
    return 0;
}