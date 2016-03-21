struct NODE
{
    NODE(int val) { this->val = val; next = nullptr; }

    int val;
    NODE* next;
};

// n1 and n2 must have same length.
NODE* AddEqualList(NODE* n1, NODE* n2)
{
    if (n1 == nullptr && n2 == nullptr)
    {
        return new NODE(0); // next is set to nullptr automatically in ctor.
    }

    NODE* sum = AddEqualList(n1->next, n2->next);

    // We always use the first node to store carry.
    // Now sum points to the calculated portion of result list, first node (sum->val) is carry.
    int val = n1->val + n2->val + sum->val;
    int carry = (int)(val / 10);
    val = val % 10;

    // Reuse carry to store current val, and create a new NODE to store current carry.
    sum->val = val;
    NODE* t = new NODE(carry);
    t->next = sum;
    sum = t;

    return sum;
}

NODE* ListAdd(NODE* num1, NODE* num2)
{
    int len1 = 0;
    NODE* p1 = num1;
    while (p1 != nullptr)
    {
        p1 = p1->next;
        ++len1;
    }

    int len2 = 0;
    NODE* p2 = num2;
    while (p2 != nullptr)
    {
        p2 = p2->next;
        ++len2;
    }

    // padding
    NODE* head1 = num1;
    NODE* head2 = num2;
    if (len1 < len2)
    {
        int diff = len2 - len1;
        while (diff > 0)
        {
            NODE* t = new NODE(0);
            t->next = head1;
            head1 = t;
        }
    }
    else if (len1 > len2)
    {
        int diff = len1 - len2;
        while (diff > 0)
        {
            NODE* t = new NODE(0);
            t->next = head2;
            head2 = t;
        }
    }

    // Now head1 and head2 are new head node, their lens are equal.
    NODE* result = AddEqualList(head1, head2);

    // Be aware the first node is always carry. Let's check if first node is 0 or not.
    // If first node is 0, which means it is not a valid carry, we can remove it.
    // Otherwise we just keep it.
    if (result->val == 0)
    {
        NODE* t = result;
        result = result->next;
        delete t;
    }

    return result;
}
