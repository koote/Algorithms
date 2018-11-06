#pragma once

struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct TreeNode
{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

struct Interval
{
    int start;
    int end;
    Interval() : start(0), end(0) {}
    Interval(const int s, const int e) : start(s), end(e) {}
};

struct RandomListNode
{
    int label;
    RandomListNode* next;
    RandomListNode* random;
    RandomListNode(int x) : label(x), next(nullptr), random(nullptr) {}
};
