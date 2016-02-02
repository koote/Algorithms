//////////////////////////////////////////////////////////////////////////
// 基数排序

#include <stdio.h>

// a 是待排序数组，len为其长度
// digit_count为数组中数字的最大位数
void RadixSort(int a[], int len, int digit_count)
{
    if (a == nullptr || len == 0)
    {
        return;
    }

    // build the temp array for counting sort
    // every digit ranges from 0..9
    int c[10] = {0};
    
    // temp array to store the counting sort result.
    int* t = new int[len];

    // every number has digit_count digits, that means we need to 
    // do digit_count passes.
    for (int i = 0; i < digit_count; ++i)
    {
        for (int x = 0; x < len; ++x)
        {
            t[x] = 0;
        }

        for (int y = 0; y < 10; ++y)
        {
            c[y] = 0;
        }

        // for every pass, use counting sort as inner stable sort.
        for (int j = 0; j < len; ++j)
        {
            // get the correct digit
            int digit = a[j];
            for (int k = 0; k < i; ++k)
            {
                digit = (int)(digit / 10);
            }
            digit %= 10;

            ++c[digit];
        }

        for (int l = 1; l <= 9; ++l)
        {
            c[l] += c[l-1];
        }

        // Output the counting result to temp array and then copy back.
        for (int m = len - 1; m >= 0; --m)
        {
            // get the correct digit
            int digit = a[m];
            for (int k = 0; k < i; ++k)
            {
                digit = (int)digit / 10;
            }
            digit %= 10;

            t[c[digit]-1] = a[m];
            --c[digit];
        }

        for (int n = 0; n < len; ++n)
        {
            a[n] = t[n];
        }
    }
}

void TestRadixSort()
{
    int array[7] = { 332, 653, 632, 755, 433, 722, 48 };

    printf("input array:");
    for (int i = 0; i < 7; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("\n");

    RadixSort(array, 7, 3);

    printf("sorted array:");
    for (int i = 0; i < 7; ++i)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
}
