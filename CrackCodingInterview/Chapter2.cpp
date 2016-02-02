//////////////////////////////////////////////////////////////////////////
// Chapter2 Linked Lists

#include "..\MiscQues\CommonDataStruct.h"

// 2.1 从一个链表中删除重复节点,不适用额外存储空间。
void RemoveDup(SLNODE* head)
{
    // 空表或者只有一个节点的话都是不用处理的。
    if (head == nullptr)
    {
        return;
    }

    SLNODE* p = head; //p用来指向q的前驱
    SLNODE* q = head->next;
    while (q != nullptr)
    {
        // 从头结点开始搜索直到q的前驱，看看有没有找到重复节点，有的话说明当前q指向的节点是重复的.
        SLNODE* r = head;
        while (r != q)
        {
            if (r->val = q->val) // find a dup
            {
                // Remove q from list.
                p->next = q->next;
                delete q;
                q = p->next;

                break;
            }

            r = r->next;
        }

        if (r == q) //r == q说明r在head .. q前驱（也就是p）中没找到重复的节点。
        {
            p = q;
            q = q->next;
        }
    }
}

// 2.2 寻找未排序的单链表中的倒数第N个元素
// 第一种思路是用递归。显然先递归到链表只剩下最后一个元素，然后回溯，并计算递归层数。
SLNODE* FindNthToLast(SLNODE* head, int n)
{
    static int nCount = 0;
    static SLNODE* pNode = nullptr;

    // 先写递归出口
    if (head == nullptr)
    {
        nCount = 0; 
        return nullptr;
    }

    FindNthToLast(head->next, n);

    ++nCount;
    if (nCount == n)
    {
        pNode = head;
    }

    return pNode;
}

// 另一种思路是用两个指针量出N长度的标尺，然后滑动到链表末尾
SLNODE* FindNthToLast2(SLNODE* head, int n)
{
    if (head == nullptr || n < 1)
    {
        return nullptr;
    }

    // fast先跳过前N-1个表项,这样循环正常结束时，fast就指向了第N个元素（从1计数）
    // 循环共执行n-1次，每次fast指针移动一次，fast指针开始是指向head，所以循环结束的时候
    // fast指针移动了n-1次，也就跳过了前面n-1个元素，所以fast指针最后指向了list(n-1)也就是第N个元素。
    SLNODE* fast = head;
    for (int i = 0; i <= n-2; ++i)
    {
        if (fast == nullptr)
        {
            return nullptr;
        }
        fast = fast->next;
    }

    if (fast == nullptr)
    {
        return nullptr;
    }

    // 此时fast指向的是从左向右数第N个元素（第一个元素计数为1）
    // slow指向的是第一个元素。这样slow和fast的距离是N
    SLNODE* slow = head;
    while (fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next;
    }

    return slow;
}

// 2.3 给定一个单链表，和一个指向该表中间某个节点的指针, 要求删除该节点。
// 方法就是从该节点开始往后遍历,然后用后面的元素覆盖前面的,直到最后一个节点,删之.
void RemoveMidNode(SLNODE* mid)
{
    if (mid == nullptr)
    {
        return;
    }

    SLNODE* p = mid;
    SLNODE* q = mid->next;
    while (q != nullptr)
    {
        p->val = q->val;
        p = q;
        q = q->next;
    }

    // now q points to the last element ,delete it.
    delete q;
    p->next = nullptr;
}

// 2.4单链表表示的数字加法
SLNODE* Add(SLNODE* a, SLNODE* b)
{
    bool bError = false;

    // the temp head node for result.
    SLNODE* dummy = new SLNODE;
    dummy->next = nullptr;

    SLNODE* p = a;
    SLNODE* q = b;
    SLNODE* r = dummy;
    int c = 0; //进位
    while(p != nullptr && q != nullptr)
    {
        if (p->val > 9 || p->val < 0)
        {
            bError = true;
            goto Done;
        }

        if (q->val > 9 || q->val < 0)
        {
            bError = true;
            goto Done;
        }

        int s = p->val + q->val + c;
        c = s / 10;

        r->next = new SLNODE;
        r->next->val = s % 10;

        r = r->next;
        p = p->next;
        q = q->next;
    }

    while (p != nullptr)
    {
        int s = p->val + c;
        c = s / 10;

        r->next = new SLNODE;
        r->next->val = s % 10;

        r = r->next;
        p = p->next;
    }

    while (q != nullptr)
    {
        int s = q->val + c;
        c = s / 10;

        r->next = new SLNODE;
        r->next->val = s % 10;

        r = r->next;
        q = q->next;
    }

    r->next = nullptr;

Done:
    if (bError)
    {
        // Destroy the result list
        while (dummy != nullptr)
        {
            SLNODE* temp = dummy;
            dummy = dummy->next;
            delete temp;
        }

        return nullptr;
    }
    else
    {
        SLNODE* res = dummy->next;
        delete dummy;
        return res;
    }
}

//2.5 单链表中有个环,要求找出环的起点,也就是环的部分开始的地方.
SLNODE* FindLoopStart(SLNODE* head)
{
    bool bLoopExists = false;

    // 首先我们检测有没有环. 慢指针一次走一格,快指针一次走两格
    // 快慢指针相遇的时候就说明有环.
    SLNODE* slow = head;
    SLNODE* fast = head;
    while (fast != nullptr && fast->next != nullptr)
    {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast)
        {
            bLoopExists = true;
            break;
        }
    }

    if (!bLoopExists)
    {
        return nullptr;
    }

    // 现在让slow回到链表头, fast停在相遇的地方,同时以速度1前进
    slow = head;
    while (slow != head)
    {
        slow = slow->next;
        fast = fast->next;
    }

    return slow;
}
