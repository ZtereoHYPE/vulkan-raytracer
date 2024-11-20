#pragma once

#include "pch.hpp"

// todo: find a more elegant way to structure a buffer builder with known structs
//  but for now this will do
class BufferBuilder {
   private:
    void *tmpBuffer;
    size_t currentSize;
    size_t currentOffset;

    void growTmpBuffer();

   public:
    BufferBuilder();
    
    BufferBuilder(const BufferBuilder &obj) = delete; // do not allow copies of this class

    template<typename T>
    void append(T value) {
        while (currentSize <= currentOffset + sizeof(T))
            growTmpBuffer();

        memcpy(tmpBuffer + currentOffset, &value, sizeof(T));
        currentOffset += sizeof(T);
    };
    
    template<typename T>
    void append(T value, size_t size) {
        while (currentSize <= currentOffset + size)
            growTmpBuffer();

        // this is a GCC-ism
        memcpy(tmpBuffer + currentOffset, &value, sizeof(T));
        currentOffset += size;
    };

    size_t getOffset();

    void pad(size_t amt);

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

