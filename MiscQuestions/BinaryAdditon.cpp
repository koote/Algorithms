//////////////////////////////////////////////////////////////////////////
// 用字符串表示的两个二进制数, 做加法, 输出仍然是字符串

#include <assert.h>

// res = s1 + s2.
// caller need to allocate the buffer first.
int BinAdd(const char* s1, const char* s2, char* res, size_t sizeInChars)
{
    // todo: check if one of them is nullptr. or both.

    const char* p = s1;
    size_t nLen1 = 0;
    while (*p != '\0')
    {
        ++nLen1;
        ++p;
    }

    const char* q = s2;
    size_t nLen2 = 0;
    while (*q != '\0')
    {
        ++nLen2;
        ++q;
    }

    // Check if we have enough space for potential carry and null terminator.
    size_t nLen = (nLen1 > nLen2 ? nLen1 : nLen2) + 1; // carry
    if (sizeInChars < nLen + 1) //null terminator
    {
        return 1;
    }
    res[nLen] = '\0';

    // Move s1, s2 and r to the last character.
    int carry = 0;
    while (nLen1 > 0 && nLen2 > 0 && nLen > 0)
    {
        // *s1 + *s2 + carry
        int v1 = s1[nLen1-- - 1] - '0';
        int v2 = s2[nLen2-- - 1] - '0';
        int v = v1 + v2 + carry;

        res[nLen-- - 1] = (v & 0x1) + '0';
        carry = (v >> 1) & 0x1;
    }

    while (nLen1 > 0 && nLen > 0)
    {
        int v1 = s1[nLen1-- - 1] - '0';
        int v = v1 + carry;

        res[nLen-- - 1] = (v & 0x1) + '0';
        carry = (v >> 1) & 0x1;
    }

    while (nLen2 > 0 && nLen > 0)
    {
        int v2 = s2[nLen2-- - 1] - '0';
        int v = v2 + carry;

        res[nLen-- - 1] = (v & 0x1) + '0';
        carry = (v >> 1) & 0x1;
    }

    res[nLen - 1] = carry + '0';

    return 0;
}
