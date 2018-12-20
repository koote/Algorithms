// 二叉树有关的操作

#include "CommonDataStruct.h"
#include <cstddef>
#include <stack>
#include <queue>

// 给定二叉搜索树,转换成双向链表
// lchild变成双链表的prev指针, rchild变成双链表的next指针.
BTNODE* BST2DLL(BTNODE* root)
{
    if (root == nullptr)
    {
        return nullptr;
    }

    BTNODE* head = nullptr;
    BTNODE* p = nullptr; //p指向已经建好的部分双链表的最后一个元素
    BTNODE* q = root;
    std::stack<BTNODE*> stkTreeNode;
    while (q != nullptr || !stkTreeNode.empty())
    {
        if (q != nullptr) //压栈过程不建表
        {
            stkTreeNode.push(q);
            q = q->lchild;
        }
        else // 在pop过程中拼接双链表
        {
            q = stkTreeNode.top();
            stkTreeNode.pop();

            if (p != nullptr)
            {
                p->rchild = q; //p->next = p
            }
            q->lchild = p; //q->prev = p

            if (head == nullptr) // q is the head node.
            {
                head = q;
            }

            // 指向双链表新节点
            p = q;

            q = q->rchild;
        }
    }

    return head;
}

//////////////////////////////////////////////////////////////////////////
// 给定二叉树,求二叉树最大的宽度
struct QNODE
{
    QNODE(BTNODE* p, int l) { pTreeNode = p, nLevel = l; }
    BTNODE* pTreeNode;
    int nLevel;
};

unsigned int GetBTreeMaxWidth(BTNODE* root)
{
    if (root == nullptr)
    {
        return 0;
    }

    std::queue<QNODE*> q;
    QNODE* r = new QNODE(root, 0);
    q.push(r);
    unsigned int uMaxDepth = 1;
    while (!q.empty())
    {
        // dequeue all elements that on the same level, and enqueue their children.
        QNODE* pFront = q.front();
        int nCurLevel = pFront->nLevel;
        while (!q.empty() && (pFront = q.front())->nLevel == nCurLevel)
        {
            q.pop();
            if (pFront->pTreeNode->lchild != nullptr)
            {
                QNODE* lch = new QNODE(pFront->pTreeNode->lchild, pFront->nLevel + 1);
                q.push(lch);
            }
            if (pFront->pTreeNode->rchild != nullptr)
            {
                QNODE* rch = new QNODE(pFront->pTreeNode->rchild, pFront->nLevel + 1);
                q.push(rch);
            }

            delete pFront;
        }

        // After the while loop is finished, the queue only contains next level elements.
        // So let's check the length of the queue.
        if (uMaxDepth < q.size())
        {
            uMaxDepth = q.size();
        }
    }

    return uMaxDepth;
}

//////////////////////////////////////////////////////////////////////////
// 已知二叉树的前序和中序,递归构造二叉树
// ps, pe是preorder前序数组的起始元素下标和结束元素下标
// is, ie是inorder中序数组的起始元素下标和结束元素下标
BTNODE* BuildBinaryTree(int preorder[], int ps, int pe, int inorder[], int is, int ie)
{
    if (ps > pe || is > ie)
    {
        return nullptr;
    }

    int nPreOrderLen = pe - ps + 1;
    int nInOrderLen = ie - is + 1;
    if (nPreOrderLen != nInOrderLen)
    {
        return nullptr;
    }

    BTNODE* root = new BTNODE();
    root->val = preorder[ps];
    if (nPreOrderLen == 1) //树只有一个元素，直接返回了。
    {
        root->lchild = nullptr;
        root->rchild = nullptr;
    }
    else
    {
        // Search the root in inorder
        int r; // r是根在中序数组中的下标位置
        for (r = is; r <= ie; ++r)
        {
            if (inorder[r] == preorder[ps])
            {
                break;
            }
        }

        // if we cannot find root element in inorder, something wrong.
        if (r == ie && inorder[r] != preorder[ps])
        {
            delete root;
            return nullptr;
        }

        if (r - is > 0) // have left subtree.
        {
            root->lchild = BuildBinaryTree(preorder, ps + 1, ps + (r - is), inorder, is, r - 1);
        }

        if (ie - r > 0) // have right subtree
        {
            root->rchild = BuildBinaryTree(preorder, ps + (r - is) + 1, pe, inorder, r + 1, ie);
        }
    }

    return root;
}

//////////////////////////////////////////////////////////////////////////
// 递归后序遍历二叉树
void PostOrder(BTNODE* root)
{
    if (root != nullptr)
    {
        PostOrder(root->lchild);
        PostOrder(root->rchild);
        printf("%d ", root->val);
    }
}

// 递归前序遍历二叉树
void PreOrder(BTNODE* root)
{
    if (root != nullptr)
    {
        printf("%d ", root->val);
        PreOrder(root->lchild);
        PreOrder(root->rchild);
    }
}

// 非递归中序遍历二叉树
void NonRescursionInOrder(BTNODE* root)
{
    if (root == nullptr)
    {
        return;
    }

    std::stack<BTNODE*> stkTreeNode;
    BTNODE* p = root;

    while (p != nullptr || !stkTreeNode.empty())
    {
        if (p != nullptr)
        {
            stkTreeNode.push(p);
            p = p->lchild;
        }
        else
        {
            p = stkTreeNode.top();
            stkTreeNode.pop();
            printf("%d ", p->val);
            p = p->rchild;
        }
    }
}

// 非递归逆中序遍历二叉树
void NonRescursionReverseInOrder(BTNODE* root)
{
    if (root == nullptr)
    {
        return;
    }

    std::stack<BTNODE*> stkNode;
    BTNODE* p = root;

    while (p != nullptr || !stkNode.empty())
    {
        if (p != nullptr)
        {
            stkNode.push(p);
            p = p->rchild;
        }
        else
        {
            p = stkNode.top();
            stkNode.pop();
            printf("%d ", p->val);
            p = p->lchild;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// 给定二叉搜索树,查找第K大个数,注意是大,所以是逆中序访问,修改非递归中序的访问左右子树的顺序即可.
void FindKthMax(BTNODE* root, int k)
{
    if (root == nullptr)
    {
        return;
    }

    std::stack<BTNODE*> stkNode;
    BTNODE* p = root;
    int nVisited = 0;

    while (p != nullptr || !stkNode.empty())
    {
        if (p != nullptr)
        {
            stkNode.push(p);
            p = p->rchild;
        }
        else
        {
            p = stkNode.top();
            stkNode.pop();
            if (++nVisited == k)
            {
                printf("We've got it, %dth maximum number is %d", k, p->val);
                break;
            }
            p = p->lchild;
        }
    }

    if (nVisited < k)
    {
        printf("Sorry, K overflowed.");
    }
}

//////////////////////////////////////////////////////////////////////////
// Test

#define countof(a) sizeof(a)/sizeof(a[0])

void TreeTest()
{
    int preorder[] = { 11, 8, 3, 1, 4, 9, 17, 13, 12, 14, 19 };
    int inorder[] = { 1, 3, 4, 8, 9, 11, 12, 13, 14, 17, 19 };

    BTNODE* root = BuildBinaryTree(preorder, 0, countof(preorder) - 1, inorder, 0, countof(inorder) - 1);

    printf("Pre order: ");
    PreOrder(root);
    printf("\r\n");

    printf("Post order : ");
    PostOrder(root);
    printf("\r\n");

    printf("In order : ");
    NonRescursionInOrder(root);
    printf("\r\n");

    printf("Reverse in order : ");
    NonRescursionReverseInOrder(root);
    printf("\r\n");

    FindKthMax(root, 5);
    printf("\r\n");

    printf("tree max width is %d\r\n", GetBTreeMaxWidth(root));

    BTNODE* head = BST2DLL(root);
    //正反向验证双链表
    BTNODE* a = head;
    while (a != nullptr)
    {
        printf("%d ", a->val);
        a = a->rchild;
    }
    printf("\r\n");
    a = head;
    while (a->rchild != nullptr)
    {
        a = a->rchild;
    }
    while (a != nullptr)
    {
        printf("%d ", a->val);
        a = a->lchild;
    }
    printf("\r\n");
}
