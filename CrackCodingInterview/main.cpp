// CrackCodingInterview.cpp : Defines the entry point for the console application.
//
#include <memory.h>
#include <iostream>
using namespace std;

void ReplaceSpaceCharToUnicode(char* lpszText);
void Rotate90Clockwise(int** image, unsigned int nTopLeft, unsigned int nSideLength);
void Rotate90Clockwise2(int** image, unsigned int n);

void TestRotationMatrix()
{
    int a[4][4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // 用一位数组的分配方式构建二维数组
    int nSize = 4;
    int* body = new int[nSize * nSize];
    memset(body, 0, sizeof(int) * nSize * nSize);

    int** b = new int*[nSize];
    memset(b, 0, sizeof(int*) * nSize);

    b[0] = body;
    for (int i = 1; i < nSize; ++i)
    {
        b[i] = b[i-1] + nSize;
    }

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            b[i][j] = a[i][j];
            cout<<b[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout<<endl;

    Rotate90Clockwise(b, 0, 4);
//     Rotate90Clockwise2(b, 4);

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            cout<<b[i][j]<<"\t";
        }
        cout<<endl;
    }

    delete[] body;
    delete[] b;
}

void main()
{

}
