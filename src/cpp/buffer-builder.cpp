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

size_t BufferBuilder::getOffset() {
    return currentOffset;
}

void BufferBuilder::pad(size_t amt) {
    currentOffset += amt;
}

void BufferBuilder::write(void *memory) {
    memcpy(memory, tmpBuffer, currentOffset);
}

BufferBuilder::~BufferBuilder() {
    free(tmpBuffer);
    tmpBuffer = NULL;
}