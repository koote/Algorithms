
#include <stdio.h>

void TreeTest();
void YoungTest();
void TestFuzzySort();
void TestRadixSort();
void RLETest();
int BinAdd(const char* s1, const char* s2, char* res, size_t sizeInChars);
void ShuffleCards();

void main()
{
    ShuffleCards();
//     TreeTest();
//     YoungTest();
//     TestFuzzySort();
//     TestRadixSort();
//     RLETest();

    char s1[] = "111111";
    char s2[] = "1";
    char res[100] = {0};
    BinAdd(s1, s2, res, 100);
    printf(res);
    printf("\n");
}
