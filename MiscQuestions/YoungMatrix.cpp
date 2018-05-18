#include <iostream>
using namespace std;

#define MAX 100

//////////////////////////////////////////////////////////////////////////
// 从Young矩阵中删除最小的数,并保持young矩阵的特性

// i and j define the top left of matrix, so the top left element is a[i][j]
// row and col define the size of matrix, so the bottom right element is a[i+row-1][j+col-1]
void YoungAdjustmentFromTopLeft(int** a, int i, int j, int row, int col)
{
    if (row == 1)
    {
        for (int k = j; k <= j + col - 2; ++k)
        {
            if (a[i][k] > a[i][k+1])
            {
                int temp = a[i][k];
                a[i][k] = a[i][k+1];
                a[i][k+1] = temp;
            }
        }
    }
    else if (col == 1)
    {
        for (int k = i; k <= i + row - 2; ++k)
        {
            if (a[k][j] > a[k+1][j])
            {
                int temp = a[k][j];
                a[k][j] = a[k+1][j];
                a[k+1][j] = temp;
            }
        }
    }
    else
    {
        if (a[i][j+1] < a[i+1][j])
        {
            // swap a[i][j] and a[i][j+1]
            int temp = a[i][j];
            a[i][j] = a[i][j+1];
            a[i][j+1] = temp;

            YoungAdjustmentFromTopLeft(a, i, j+1, row, col-1);
        }
        else
        {
            //swap a[i][j] and a[i+1][j]
            int temp = a[i][j];
            a[i][j] = a[i+1][j];
            a[i+1][j] = temp;

            YoungAdjustmentFromTopLeft(a, i+1, j, row-1, col);
        }
    }
}

// Return and remove the smallest element in Young matrix.
int ExtractMin(int ** a, int m, int n)
{
    if (m == 0 || n == 0)
    {
        return -1;
    }

    int min = a[0][0];
    a[0][0] = a[m-1][n-1];
    a[m-1][n-1] = MAX;

    YoungAdjustmentFromTopLeft(a, 0, 0, m, n);

    return min;
}

//////////////////////////////////////////////////////////////////////////////
// 向Young矩阵中插入一个数

void YoungAdjustmentFromBottomRight(int** a, int row, int col)
{
    if (row == 1)
    {
        if (a[row-1][col-1] > a[row-1][col-2])
        {
            return;
        }

        for (int k = col - 1; k > 0; --k)
        {
            if (a[row-1][k] < a[row-1][k-1])
            {
                int temp = a[row-1][k-1];
                a[row-1][k-1] = a[row-1][k];
                a[row-1][k] = temp;
            }
        }
    }
    else if (col == 1)
    {
        if (a[row-1][col-1] > a[row-2][col-1])
        {
            return;
        }

        for (int k = row - 1; k > 0; --k)
        {
            if (a[k][col-1] < a[k-1][col-1])
            {
                int temp = a[k-1][col-1];
                a[k-1][col-1] = a[k][col-1];
                a[k][col-1] = temp;
            }
        }
    }
    else
    {
        // compare and find the largest element.
        int i = row - 1;
        int j = col - 1;
        if(a[row-1][col-2] > a[row-2][col-1])
        {
            --j;
        }
        else
        {
            --i;
        }
        
        if (a[row-1][col-1] > a[i][j])
        {
            return;
        }

        //swap a[row-1][col-1] and a[i][j]
        int temp = a[i][j];
        a[i][j] = a[row-1][col-1];
        a[row-1][col-1] = temp;

        YoungAdjustmentFromBottomRight(a, i+1, j+1);
    }
}

// a[m][n] is a Young matrix.
// val is the value that will be inserted into a.
// i and j is the location of val.
bool YoungInsert(int** a, int row, int col, int val)
{
    if (a[row-1][col-1] != MAX) //full
    {
        return false;
    }

    a[row-1][col-1] = val;
    YoungAdjustmentFromBottomRight(a, row, col);

    return true;
}

//////////////////////////////////////////////////////////////////////////////
// 在Young矩阵中查找一个数
bool Find(int** a, int row, int col, int val, int& i, int& j)
{
    i = row - 1;
    j = 0;

// 注释掉的循环和下面的循环作用一样.
//     while (i >= 0 && j < col)
//     {
//         if (a[i][j] == val) 
//         {
//             return true;
//         }
// 
//         a[i][j] < val ? ++j : --i;
//     }

    for(i = row - 1; i >= 0; --i)
    {
        while (j <= col - 1 && a[i][j] < val)
        {
            ++j;
        }

        if (a[i][j] == val)
        {
            return true;
        }
    }
    
    return false;
}

//////////////////////////////////////////////////////////////////////////////
void PrintMatrix(int** a, int row, int col)
{
    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            a[i][j] == MAX ? cout<<"\t" : cout<<a[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout<<endl;
}

void YoungTest()
{
    int a[4][4] = {2, 4, 9, 12, 3, 5, 14, MAX, 8, 16, MAX, MAX, MAX, MAX, MAX, MAX};

    int** b = new int*[4];
    for(int i = 0; i < 4; i++)
        b[i] = new int[4];
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
            b[i][j] = a[i][j];
    }

    cout<<"initial matrix is"<<endl;
    PrintMatrix(b, 4, 4);

    int i,j;
    bool res = Find(b, 4, 4, 2, i, j);
    if (res)
    {
        cout<<"got value "<<10<<", location:"<<i<<j<<endl;
    }

    YoungInsert(b, 4, 4, 1);
    PrintMatrix(b, 4, 4);

//     ExtractMin(b, 4, 4);
// 
//     for(int i = 0; i < 4; i++)
//     {
//         for(int j = 0; j < 4; j++)
//             cout<<b[i][j]<<"\t";
//         cout<<endl;
//     }
//     cout<<endl;
// 
//     ExtractMin(b, 4, 4);
// 
//     for(int i = 0; i < 4; i++)
//     {
//         for(int j = 0; j < 4; j++)
//             cout<<b[i][j]<<"\t";
//         cout<<endl;
//     }
// 
//     cout<<endl;
}
