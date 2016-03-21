//////////////////////////////////////////////////////////////////////////
// 模拟洗牌程序

#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

// 给一个数组进行洗牌
void Shuffle(int arrEle[], int nSize)
{
    srand((unsigned)time(0));

    for (int i = nSize-1; i >= 1; --i)
    {
        int j = rand() % i; //产生一个0~i之间的随机数

        //swap arrEle[i] and arrEle[j]
        int temp = arrEle[i];
        arrEle[i] = arrEle[j];
        arrEle[j] = temp;
    }
}

void ShuffleCards()
{
    int* cards = new int [52];
    for (int i = 0; i < 52; ++i)
    {
        cards[i] = i + 1;
        cout << cards[i] << ' ';
    }
    cout<<endl;

    Shuffle(cards, 52);

    for (int i = 0; i < 52; ++i)
    {
        cout << cards[i]<<' ';
    }
    cout<<endl;

    delete[] cards;
}
