//////////////////////////////////////////////////////////////////////////
// 单链表基本操作

#include "CommonDataStruct.h"

// p节点后面插入一个新节点q
// q->next = p->next;
// p->next = q;

// 从单链表中删除一个值为v的节点。
SLNODE* Delete(SLNODE* head, int v)
{
    SLNODE* sentinel = new SLNODE;
    sentinel->next = head;

    SLNODE* p = sentinel;
    SLNODE* q = nullptr; //保存要删除的那个节点
    while (p->next != nullptr)
    {
        if (p->next->val == v)
        {
            q = p->next;
            p->next = q->next;
            break;
        }
    }

    // 之所以要修改head是因为有可能要删除的就是原来的头结点。
    head = sentinel->next;
    delete sentinel;
    return q;
}

// 三指针原地逆转单链表，返回新的头结点
SLNODE* Reverse(SLNODE* head)
{
    //若有0个或者1个元素，无须逆转
    //这里只检查空表的情况，如果只有1个元素，q == nullptr，后面的while循环不会执行。
    if (head == nullptr)
    {
        return head;
    }

    SLNODE* p = head;
    SLNODE* q = head->next;

    // 头结点的next是nullptr，单独处理一下。
    head->next = nullptr;

    while (q != nullptr)
    {
        SLNODE* temp = q->next;
        q->next = p;

        p = q;
        q = temp;
    }

    // 当循环结束的时候，q一定是空指针了。因为p是慢指针，q是快指针
    // 所以此时p指向的是最后一个节点也就是新表头。
    return p;
}

// 特殊的单链表复制
#include <map>
#include <assert.h>

struct SSLNODE
{
    int val;
    SSLNODE* next;
    SSLNODE* opt;
};

SSLNODE* CopySpecialSingleList(SSLNODE* head)
{
    if (head == nullptr)
    {
        return nullptr;
    }

    // a dummy head node, to simplify the code
    std::map<SSLNODE*, SSLNODE*> mapNodes;
    SSLNODE* dummy = new SSLNODE;
    SSLNODE* q = dummy;
    SSLNODE* p = head;
    while (p != nullptr)
    {
        SSLNODE* r = new SSLNODE;
        r->val = p->val;
        mapNodes.insert(std::pair<SSLNODE*, SSLNODE*>(p, r));

        // chain it into the new list.
        q->next = r;

        q = r;
        p = p->next;
    }

    // Next iteration, setup the opt field.
    p = head;
    q = dummy->next;
    while (p != nullptr && q != nullptr)
    {
        std::map<SSLNODE*, SSLNODE*>::iterator i = mapNodes.find(p->opt);
        assert(i != mapNodes.end());
        q->opt = i->second;
    }

    SSLNODE* result = dummy->next;
    delete dummy;
    return result;
}
