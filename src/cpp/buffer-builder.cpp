#include "buffer-builder.hpp"

const size_t INITIAL_SIZE = 16;

void BufferBuilder::growTmpBuffer() {
    currentSize *= 2;
    tmpBuffer = realloc(tmpBuffer, currentSize);
    memset(tmpBuffer + (currentSize / 2), 0, (currentSize / 2));
}

BufferBuilder::BufferBuilder() {
    tmpBuffer = calloc(INITIAL_SIZE, 1);
    currentSize = INITIAL_SIZE;
    currentOffset = 0;
}

/*
 * Gets the current offset within the buffer in bytes.
 */
size_t BufferBuilder::getOffset() {
    return currentOffset;
}

/*
 * Pads the given amount with zeros.
 */
void BufferBuilder::pad(size_t amt) {
    currentOffset += amt;
}

/*
 * Writes the built buffer to a location in memory.
 */
void BufferBuilder::write(void *memory) {
    memcpy(memory, tmpBuffer, currentOffset);
}

BufferBuilder::~BufferBuilder() {
    free(tmpBuffer);
    tmpBuffer = NULL;
}