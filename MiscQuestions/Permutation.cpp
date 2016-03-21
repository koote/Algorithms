
#include <stdio.h>

#define SWAP(x, y) do {int t=x; x=y; y=t;} while(0);

void GeneratePermutation(int a[], int k, int m)
{
    if(k == m)
    {
        for(int i = 0; i <= m; ++i)
        {
            printf("%d", a[i]);
        }
        printf("\r\n");
    }
    else
    {
        for(int i = k; i <= m; ++i)
        {
            SWAP(a[i], a[k]);
            GeneratePermutation(a, k+1, m);
            SWAP(a[i], a[k]);
        }
    }
}  

void TestPrintPermuation()
{
    int a[] = {0,1,2,3,4,5,6,7,8,9};
    GeneratePermutation(a, 0, 3);
}
