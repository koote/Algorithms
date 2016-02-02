#pragma once

struct BTNODE //二叉树
{
    int val;
    BTNODE* lchild;
    BTNODE* rchild;
};

struct DLNODE //双向链表
{
    int val;
    DLNODE* next;
    DLNODE* prev;
};

struct SLNODE //单链表
{
    int val;
    SLNODE* next;
};
