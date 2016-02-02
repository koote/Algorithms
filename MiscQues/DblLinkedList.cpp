
#include "CommonDataStruct.h"

// Ë«ÏòÁ´±í, Á½Á½ÄæÖÃ
DLNODE* SwapPair(DLNODE* head)
{
    if (head == nullptr)
    {
        return head; // Nothing changed.
    }

    DLNODE* p = head;
    DLNODE* q = head->next;
    DLNODE* r = nullptr;

    // If q is nullptr, the list only have 1 node, nothing will be changed, 
    // and also the head node remains unchanged.
    while (q != nullptr)
    {
        r = q->next; //save the next location, could be null.

        q->next = p;
        q->prev = p->prev;
        if (p->prev != nullptr)
        {
            p->prev->next = q;
        }
        p->prev = q;
        p->next = r;
        if (r != nullptr)
        {
            r->prev = p;
        }
        else // If r is null, means we have reached the end.
        {
            break;
        }

        // before swap, p-q-r-x, after swap, q-p-r-x
        // so now we move to next pair (r-x)
        p = p->next; //r, we already know that r is not null here.
        q = p->next; //x could be null, if so, next loop will exit.
    }

    return (head->prev == nullptr) ? head : head->prev;
}

void TestDblLinkedList()
{

}
