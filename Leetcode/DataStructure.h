#pragma once

struct ListNode
{
    int val;
    ListNode* next;
    explicit ListNode(const int x) : val(x), next(nullptr) {}
};

struct TreeNode
{
    int val;
    TreeNode* left;
    TreeNode* right;
    explicit TreeNode(const int x) : val(x), left(nullptr), right(nullptr) {}
};

struct TreeLinkNode
{
    int val;
    TreeLinkNode* left;
    TreeLinkNode* right;
    TreeLinkNode* next;
    explicit TreeLinkNode(const int x) : val(x), left(nullptr), right(nullptr), next(nullptr) {}
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
    explicit RandomListNode(const int x) : label(x), next(nullptr), random(nullptr) {}
};

struct Node
{
    int val;
    vector<Node*> neighbors;
    explicit Node() : val(0), neighbors(vector<Node*>()) {}
    explicit Node(int _val) : val(_val), neighbors(vector<Node*>()) {}
    explicit Node(int _val, vector<Node*> _neighbors) : val(_val), neighbors(_neighbors) {}
};
