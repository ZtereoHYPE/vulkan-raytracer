#pragma once

#include "../pch.hpp"

class BufferBuilder {
   private:
    void *tmpBuffer;
    size_t currentSize;
    size_t currentOffset;

    void growTmpBuffer();

   public:
    BufferBuilder();
    
    BufferBuilder(const BufferBuilder &obj) = delete; // do not allow copies of this class

    /*
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
    };
    
    /*
     * Overload of the append() function where the stride is explicitly
     * given. 
     * 
     * Useful when writing unpadded data that requires some form
     * of memory padding.
     */
    template<typename T>
    void append(T value, size_t size) {
        while (currentSize <= currentOffset + size)
            growTmpBuffer();

        // this is a GCC-ism.
        memcpy(tmpBuffer + currentOffset, &value, sizeof(T));
        currentOffset += size;
    };

    size_t getOffset();

    void pad(size_t amt);


    void write(void *memory);

    ~BufferBuilder();
};

