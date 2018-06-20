#include <vector>
#include "DataStructure.h"
using namespace std;

ListNode* createList(vector<int>& nums)
{
    ListNode head(-1);
    ListNode* last = &head;
    for (int num : nums)
    {
        last->next = new ListNode(num);
        last = last->next;
    }

    return head.next;
}
