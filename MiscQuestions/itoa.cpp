
#include <string.h>

// 给定十进制数字,转换成2~36进制并以字符串形式输出

// itoa (-12, 10);
#define LEN 512

char* itoa(int val, int radix) // radix : 2~36
{
    const char table[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    bool bNeg = false;
    char* pResult = new char[LEN];
    if (pResult == nullptr)
    {
        return nullptr;
    }

    memset(pResult, 0, LEN * sizeof(char));
    char* p = pResult;

    if (val == 0)
    {
        *p = '0';
        return pResult;
    }

    if (val < 0) // Make it postitive
    {
        bNeg = true;
        *p++ = '-';
        val = -val;
    }

    while (val > 0)
    {
        *p++ = table[val % radix];
        val /= radix;
    }
    *p-- = '\0';

    // reverse the string q..p in place
    char* q = bNeg ? pResult+1 : pResult;
    while (q < p)
    {
        char ch = *q;
        *q = *p;
        *p = ch;

        q++;
        p--;
    }

    return pResult;
}

// -1 indicates invalid parameter
// 0 means success
int itoa2(int val, int radix, char* buffer, size_t sizeInChars)
{
    const char table[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    bool bNeg = false;

    if (buffer == nullptr || sizeInChars < 2) //最小数字0，也需要两个字符: '0', '\0'.
    {
        return -1;
    }

    memset(buffer, 0, sizeInChars);
    char* p = buffer;

    if (val == 0)
    {
        *p = '0';
        return 0;
    }

    if (val < 0)
    {
        bNeg = true;
        *p++ = '-';
        val = -val;
    }

    char* pEnd = buffer + sizeInChars - 1; // For null terminator

    while (val > 0 && p < pEnd)
    {
        *p++ = table[val % radix];
        val /= radix;
    }

    if (val > 0) // buffer is full but the convertion hasn't finished yet.
    {
        return -1; 
    }

    *p-- = '\0';

    // Now reverse the result string back.
    char* q = buffer;
    if(bNeg)
    {
        q++; // because the first char is the sign.
    }

    // reverse q..pResult
    while (q < p)
    {
        char ch = *q;
        *q = *p;
        *p = ch;

        q++;
        p--;
    }

    return 0;
}
