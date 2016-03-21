#include <stdio.h>

// 输入一个BYTE数组，输出用游程编码压缩后的数组
// 要求压缩后的数组长度不得大于原数组长度，若大于，则说明无法用RLE压缩，返回错误

// Design Tip：在这种对一个缓冲区进行处理，然后将结果输出到另一个缓冲区返回给caller的函数中，
// 应该遵循由caller预分配内存的做法。
//
// 如果由callee分配，那么caller将不知道如何返还内存，这取决于callee采用何种内存分配方式。
// 是new还是malloc还是VirtualAlloc/LocalAlloc? caller并不知道.
// 同样的,callee也不能对传进来的指针做任何销毁动作,因为callee也不知道这内存是采用何种方式分配的,
// 有可能caller用malloc分配内存, callee就不能用delete去销毁.

// return -1 means error.
//        0 means cannot compress using RLE encoding, the result data is larger than input.
//        > 0 means acturally size of dest buffer.
//        So the return value <= nSrcSz
int RLEEncoding(const unsigned char* pData, size_t nSrcSz, // input data buffer and its length
                unsigned char* pResult, size_t nDstBufSz, size_t& nDstDataSz) // result buffer and actually data length
{
    if (pData == nullptr || nSrcSz == 0 || pResult == nullptr || nDstBufSz == 0)
    {
        return -1; //error parameter
    }

    size_t i = 0;
    size_t j = 0;
    while(i < nSrcSz && (j + 1) < nDstBufSz)
    {
        pResult[j] = pData[i];
        pResult[j + 1] = 1;

        while (pData[i] == pData[i + 1])
        {
            if (pResult[j + 1] < 255)
            {
                ++pResult[j + 1];
            }
            else // overflow
            {
                j += 2;
                pResult[j] = pData[i];
                pResult[j + 1] = 1;
            }

            ++i;
        }

        // here, pData[i] != pData[i + 1], so next element is a new value.
        ++i;
        j += 2;
    }

    // input data is not completely processed, so it cannot be compressed and stored
    // in the result buffer, the result data is larger than original input data.
    if (i < nSrcSz) 
    {
        return 0;
    }
    else
    {
        nDstDataSz = j;
        return j;
    }
}


void RLETest()
{
    unsigned char sz[] = "abcd";
    unsigned char* szRes = new unsigned char[4];

    size_t res;
    int b = RLEEncoding(sz, 4, szRes, 4, res);
    if (b == -1)
    {
        printf("invalid input parameter");
    }
    else if (b == 0)
    {
        printf("cannot encoding with RLE.");
    }
    else
    {
        printf("Success.");
    }
}
