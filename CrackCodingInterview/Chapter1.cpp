//////////////////////////////////////////////////////////////////////////
// Chapter 1 Arrays and Strings

#include <memory.h>
#include <iostream>
using namespace std;

// 1.1 判断一个字符串是否由不重复的字符组成，不准使用其他的数据结构
bool IsStringCharUnique(const char* szText)
{
    unsigned int uAsciiTbl[256] = {0};
    while(*szText != '\0')
    {
        if (uAsciiTbl[*szText] > 0)
        {
            return false;
        }

        uAsciiTbl[*szText]++;
        ++szText;
    }

    return true;
}

// 存储空间精简版，使用一个bit来表示每个字符是否已经出现。
bool IsStringCharUnique2(const char* szText)
{
    // ASCII 0 ~ 255 共256个字符
    // 32bits * 8 = 256，需要8个整数

    // 假设字符集有N个字符，32个为一组,划分成组
    // 组序号就是 N / 32取整，组内偏移就是 N % 32.
    // 总共需要多少组呢? C = N / 32, 同时判断 N % 32是否为零,有余数的话C加1
    // C = (INT)(N / 32) + (N % 32 ? 1 : 0);
    unsigned int bitmap[8] = {0};
    while(*szText != '\0')
    {
        unsigned int i = *szText / sizeof(unsigned int);
        unsigned int j = *szText % sizeof(unsigned int);

        if (bitmap[i] & (1 << j))
        {
            return false;
        }

        bitmap[i] |= 1 << j;
        ++szText;
    }

    return true;
}

// 1.2 逆序一个C字符串
void Reverse(char* szText)
{
    char* j = szText;
    while (*j != '\0')
    {
        ++j;
    }

    char* i = szText;
    --j; //Now j points to the last character.

    while (i < j)
    {
        //swap szText[i] & szText[j]
        char temp = *i;
        *i++ = *j;
        *j-- = temp;
    }
}

//1.3 去除字符串中的重复出现的字符, 不准使用多余的存储空间

// 思路：从a[1]开始向后依次访问所有字符，每访问到一个新字符，就从已经访问过的区间内搜索看
// 是否之前已经有了同样的字符,有的话说明就是重复的.
void RemoveDupChar(char* szText)
{
    if (szText == nullptr)
    {
        return;
    }

    for (int i = 1; szText[i] != '\0';)
    {
        int j = 0;
        for (j = 0; j < i; ++j)
        {
            if (szText[j] == szText[i]) // dup!
            {
                // Shift szText[i+1 .. len -1] left.
                int k = 0;
                for (k = i+1; szText[k] != '\0'; ++k)
                {
                    szText[k-1] = szText[k];
                }
                szText[k] = '\0';
            }
        }

        if (j == i)
        {
             ++i;
        }
    }
}

// 1.4 判断两个字符串是不是打乱顺序
bool IsAnagram(const char* s1, const char* s2)
{
    if (s1 == nullptr || s2 == nullptr)
    {
        return false;
    }

    int nCharSet[256] = {0};
    const char* p = s1;
    while (*p != '\0')
    {
        ++nCharSet[*p];
        ++p;
    }

    const char* q = s2;
    while (*q != '\0')
    {
        if (nCharSet[*q] == 0)
        {
            return false;
        }
        --nCharSet[*q];
        ++q;
    }

    for (int i = 0; i < 256; ++i)
    {
        if (nCharSet[i] != 0)
        {
            return false;
        }
    }

    return true;
}


//1.5 把字符串中的空格符全部换成%20, 假设原空间足够大
void ReplaceSpaceCharToUnicode(char* lpszText)
{
    if (lpszText == nullptr)
    {
        return;
    }

    int nStrLength = 0;
    int nCntOfSpaces = 0;
    char* p = lpszText;
    while (*p != '\0')
    {
        ++nStrLength;
        if (*p == ' ')
        {
            ++nCntOfSpaces;
        }
        ++p;
    }

    if (nCntOfSpaces == 0)
    {
        return;
    }

    // Now replace all spaces to '%20'. Assume that the original string
    // has enough space to store the result string.
    int nNewLength = nCntOfSpaces * 2 + nStrLength;
    char* q = lpszText + nNewLength - 1; //q points to the last position in result string.
    for (int i = nStrLength-1; i >= 0; --i)
    {
        if (lpszText[i] == ' ')
        {
            *q-- = '0';
            *q-- = '2';
            *q-- = '%';
        }
        else
        {
            *q-- = lpszText[i];
        }
    }
}


// 1.6 一幅图像用N×N的矩阵表示,每个像素点占4个字节, 写一个函数把图像旋转90度, 需原地进行
// 原地进行就是不占用另外的存储空间, 因此简单起见,必须是方阵.

// 思路: 以顺时针旋转90度为例. 最左边一列变成最上面一行, 最上面一行变成最右边一列, 最右边一列变成最下面一行.
// 整个圈旋转90度. 考虑到这个动作的相似性, 可以用递归来进行, 最外圈做完后, 方阵尺寸减一, 再重复, 直到只有一个元素.

// 方阵为:
// image[nTopLeft][nTopLeft]       ....     image[nTopLeft][nBottomRight]
//        .                                               .
//        .                                               .
//        .                                               .
//        .                                               .
//        .                                               .
// image[nBottomRight][nTopLeft]   ....     image[nBottomRight][nBottomRight]
void Rotate90Clockwise(int** image, unsigned int nTopLeft, unsigned int nSideLength)
{
    // The square matrix has only 1 element, nothing will be done.
    if (nSideLength <= 1)
    {
        return;
    }

    int nBottomRight = nTopLeft + nSideLength - 1;
    for (unsigned int i = 0; i < nSideLength - 1; ++i)
    {
        // save top
        int temp = image[nTopLeft][nTopLeft + i];

        // left => top
        image[nTopLeft][nTopLeft + i] = image[nBottomRight - i][nTopLeft];

        // bottom => left
        image[nBottomRight - i][nTopLeft] = image[nBottomRight][nBottomRight - i];

        //right => bottom
        image[nBottomRight][nBottomRight - i] = image[nTopLeft + i][nBottomRight];

        // saved top => right
        image[nTopLeft + i][nBottomRight] = temp;
    }

    Rotate90Clockwise(image, nTopLeft + 1, nSideLength - 2);
}

void Rotate90Clockwise2(int** image, unsigned int n)
{
    for (unsigned int layer = 0; layer < n / 2; ++layer)
    {
        int nTopLeft = layer;
        int nBottomRight = n - 1- layer;
        for (int i = nTopLeft; i < nBottomRight; ++i)
        {
            int offset = i - nTopLeft;
            int top = image[nTopLeft][i];
            image[nTopLeft][i] = image[nBottomRight-offset][nTopLeft];
            image[nBottomRight-offset][nTopLeft] = image[nBottomRight][nBottomRight-offset];
            image[nBottomRight][nBottomRight-offset] = image[i][nBottomRight];
            image[i][nBottomRight] = top;
        }
    }
}

// 矩阵中某个元素为零，则将其所在的行和列都置为零
// 思路：先扫描整个矩阵，碰到一个零元素就把它的行号和列号都记下来，扫描完成后再一次性把标记过的行列都置为零。
void ClearRowCol(int** matrix, int m, int n)
{
    int* row = new(nothrow) int[m];
    if (row == nullptr)
    {
        return;
    }
    memset(row, 0, sizeof(int) * m);

    int* col = new(nothrow) int[n];
    if (col == nullptr)
    {
        return;
    }
    memset(col, 0, sizeof(int) * n);

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (matrix[i][j] == 0)
            {
                row[i] = 1;
                col[j] = 1;
            }
        }
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (row[i] == 1 || col[j] == 1)
            {
                matrix[i][j] = 0;
            }
        }
    }
}
