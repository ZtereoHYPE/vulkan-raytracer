#pragma once

#include <cstdlib>
#include <cstring>

/**
 * Utility class used to making writing raw data to memory a bit more pleasant
 */
class BufferBuilder {
    void *tmpBuffer;
    size_t currentSize;
    size_t currentOffset;

    void growTmpBuffer();

   public:
    BufferBuilder();
    BufferBuilder(const BufferBuilder &obj) = delete; // do not allow copies of this class

    /**
     * Generic function that appends data to the buffer.
     * The stride is computed from the size of the appended object.
     */
    template<typename T>
    void append(T value) {
        while (currentSize <= currentOffset + sizeof(T))
            growTmpBuffer();

        // this is a GCC-ism.
        memcpy(tmpBuffer + currentOffset, &value, sizeof(T));
        currentOffset += sizeof(T);
    }

    size_t getOffset();
    void pad(size_t amt);
    void write(void *memory);
    ~BufferBuilder();
};

