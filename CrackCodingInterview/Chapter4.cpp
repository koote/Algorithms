#include <math.h>
#include "..\MiscQues\CommonDataStruct.h"

//4.1 检查一个二叉树是不是平衡二叉树
int CalcBSTHeight(BTNODE* root)
{
    if (root == nullptr)
    {
        return 0;
    }
    else if (root->lchild == nullptr || root->rchild == nullptr)
    {
        return 1;
    }
    else
    {
        int lh = CalcBSTHeight(root->lchild);
        int rh = CalcBSTHeight(root->rchild);
        return 1 + (lh > rh ? lh : rh);
    }
}

bool IsBalancedBST(BTNODE* root)
{
    if (root == nullptr)
    {
        return true;
    }

    int lh = CalcBSTHeight(root->lchild);
    int rh = CalcBSTHeight(root->rchild);

    if (abs(lh - rh) > 1)
    {
        return false;
    }

    return IsBalancedBST(root->lchild) && IsBalancedBST(root->rchild);
}
