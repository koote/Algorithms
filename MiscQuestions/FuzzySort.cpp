// 区间模糊排序,算法导论习题

#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <algorithm>

using namespace std;

struct SEGMENT
{
    int l;
    int r;

    SEGMENT& operator=(const SEGMENT& rval)
    {
        l = rval.l;
        r = rval.r;
        return *this;
    }
};

// -1 : seg1 < seg2
// 1 : seg1 > seg2
// 0 : seg1 == seg2, overlapped.
int SegCmp(const SEGMENT& seg1, const SEGMENT& seg2)
{
    if (seg1.r < seg2.l)
    {
        return -1;
    }
    else if (seg1.l > seg2.r)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void SegPartition(SEGMENT* arrSegs, int start, int end, int& ms, int& me)
{
    if (start >= end)
    {
        ms = start;
        me = start;
        return;
    }

    SEGMENT pivot = arrSegs[end];
    int j = start - 1; // arrSeg[j] is currently last element that smaller than pivot.
    int m = start - 1; // arrSeg[m] is currently first element that equal to pivot.
    for (int i = start; i < end; ++i) // start .. end-1
    {
        // 凡小于pivot的，都
        if (SegCmp(arrSegs[i], pivot) < 0) //arrSegs[i] < pivot
        {
            // 先把arrSeg[i]放到前半部分的末尾,arrSegs[j]永远指向最后一个小于pivot的元素
            ++j;

            assert(j<=end && j>=start);
            assert(i<=end && i>=start);

            SEGMENT temp = arrSegs[j];
            arrSegs[j] = arrSegs[i];
            arrSegs[i] = temp;

            // 判断前半部分是否包含了一段等于pivot的区段,如果有的话,就把arrSeg[i](小于pivot)和
            // 第一个等于pivot的元素对调,确保等于pivot的元素都连续集中在左半部分的尾部
            // 这里m一定小于j,假设左半部分的尾部只有一个等于pivot的元素,则m == j, 在补了一个小于
            // pivot的元素之后, j增一, m<j.
            if (m > start - 1 && m < j) // arrSegs[m] is the first element that equal to pivot.
            {
                // swap arrSegs[m] and arrSegs[j]

                assert(j<=end && j>=start);
                assert(m<=end && m>=start);

                SEGMENT temp = arrSegs[m];
                arrSegs[m] = arrSegs[j];
                arrSegs[j] = temp;
                ++m;
            }
        }
        else if (SegCmp(arrSegs[i], pivot) == 0)
        {
            // 到最后，判定为相等的那些区间，一定是有公共交集的。
            // 假设a[i]，a[j]，a[k]三个区间，a[i]选作最初的pivot。先碰到a[j]，a[j]和a[i]相交，判定
            // a[i] == a[j]，但如果此时不收缩pivot为两者公共区间，假设再碰到区间a[k]，区间a[k]和区间
            // a[i]相交但是和区间a[j]不相交（例如a[i]=[10,20],a[j]=[0, 11],a[k]=[15,25])，那么会错
            // 误判定a[i]==a[k],根据传导性,会得出a[i]==a[j]==a[k]的错误结论,这显然是不成立的,因为无论
            // 在a[j]和a[k]中怎么取c[j]和c[k],c[j]都一定小于c[k].
            // 回顾区间相等的定义,就是要找到公共子区间, 使得当c在该子区间内时,c[i] == c[j] == c[k]
            
            // correct the pivot.
            pivot.l = max(arrSegs[i].l, pivot.l);
            pivot.r = min(arrSegs[i].r, pivot.r);

            // put arrSeg[i] to the left part.
            ++j;

            assert(j<=end && j>=start);
            assert(i<=end && i>=start);

            SEGMENT temp = arrSegs[j];
            arrSegs[j] = arrSegs[i];
            arrSegs[i] = temp;

            if (m == start - 1) // Find the first element equals to pivot.
            {
                m = j;
            }
        }
        else //arrSegs[i] > pivot
        {
            continue;
        }
    }

    // Put original pivot element on the correct position.
    ++j;

    assert(j<=end && j>=start);

    SEGMENT temp = arrSegs[j];
    arrSegs[j] = arrSegs[end];
    arrSegs[end] = temp;

    me = j; // arrSegs[j]是原始的pivot的最终位置
    ms = m == start - 1 ? j : m; // 若m==start-1,说明pivot没有相等区间.否则m指向第一个相等元素
}

void RandomSegPartition(SEGMENT* arrSegs, int start, int end, int& ms, int& me)
{
    if (start >= end)
    {
        ms = start;
        me = start;
        return;
    }

    int r = rand() % (end - start) + start;

    // swap arrSegs[r] and arrSegs[end]
    SEGMENT temp = arrSegs[r];
    arrSegs[r] = arrSegs[end];
    arrSegs[end] = temp;

    SegPartition(arrSegs, start, end, ms, me);
}

void FuzzySort(SEGMENT* arrSegs, int start, int end)
{
    int ms = 0; //ms,me分别指向判定为和pivot相等的元素序列的起始和结尾.
    int me = 0;

    if (start >= end)
    {
        return;
    }

    SegPartition(arrSegs, start, end, ms, me);
    
    if (me - ms > 0)
    {
        // sort arrSegs[ms..me], using their left end point.
    }

    FuzzySort(arrSegs, start, ms-1);
    FuzzySort(arrSegs, me+1, end);
}

bool PrintExitCArray(SEGMENT arrSegs[], int n)
{
    int* c = new int[n];
    cout<<arrSegs[0].l<<" ";
    c[0] = arrSegs[0].l;
    for(int i = 1; i < n; i++)
    {
        c[i] = max(c[i-1], arrSegs[i].l);
        if(c[i] > arrSegs[i].r)
        {
            cout<<"error!\n";
            return false;
        }

        cout<<c[i]<<" ";
    }

    return true;
}

void TestFuzzySort()
{
    srand((unsigned)time(0));
    SEGMENT arrSegs[10] = {0};
    int n = 10;
    cout<<"Building original segments:"<<endl;
    for(int i = 0; i < n; i++)
    {
        arrSegs[i].l = rand() % 1000;
        arrSegs[i].r = rand() % 1000;
        
        while(arrSegs[i].l > arrSegs[i].r) // Ensure r >= l
        {
            arrSegs[i].r = rand() % 1000;
        }
        
        cout<<arrSegs[i].l<<" "<<arrSegs[i].r<<endl;
    }

    FuzzySort(arrSegs, 0, n-1);

    cout<<endl<<"Sorted segments:"<<endl;
    for(int i = 0; i < n; i++)
    {
        cout<<arrSegs[i].l<<" "<<arrSegs[i].r<<endl;
    }

    cout<<endl<<"Now check sort result:"<<endl;
    if(PrintExitCArray(arrSegs, n))
    {
        cout<<endl<<"Test case passed"<<endl;
    }
    else
    {
        cout<<"Failed."<<endl;
    }
    cout<<endl;
}
