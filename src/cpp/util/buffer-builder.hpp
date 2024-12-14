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

    /*
     * Gets the offset of a given memory location relative to a specific
     * type.
     * 
     * For example, if the index of an integer in an array of integers is needed,
     * it can be obtained by calling getRelativeOffset<int>() on the buffer builder.
     */
    template<typename Type>
    size_t getRelativeOffset() {
        if (currentOffset == 0)
            return 0;

        if ((currentOffset % sizeof(Type)) != 0)
            throw std::runtime_error("Trying to get offset with regards to type not granular enough");
    
        return currentOffset / sizeof(Type);
    };

    void write(void *memory);

    ~BufferBuilder();
};

