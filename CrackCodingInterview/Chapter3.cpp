//3.3 设计一个有上限的栈，并能扩展
#define STACK_CELL_SIZE 100 //每个物理栈能有100个单元
#include <stdexcept>
class SetOfStack
{
private:
    //对每个物理栈而言，栈底是数组最后一个元素arrEle[STACK_CELL_SIZE - 1],
    //对整个逻辑栈而言，当前活动的物理栈，永远在这个单链表的表头节点。
    struct StackCell
    {
        // 采取让m_nTop指向当前栈顶元素的做法。而不是让m_nTop指向下一个压栈元素的位置的做法。
        int top;
        int arrEle[STACK_CELL_SIZE];
        StackCell* pNext;

        StackCell()
        {
            for (int i = 0; i < STACK_CELL_SIZE; ++i)
            {
                arrEle[i] = 0;
            }
            top = STACK_CELL_SIZE;
        }
    }* m_pStack;

public:
    SetOfStack()
    {
        m_pStack = nullptr;
    }

    virtual ~SetOfStack()
    {
        // release all stack cell
        while (m_pStack != nullptr)
        {
            StackCell* p = m_pStack;
            m_pStack = m_pStack->pNext;
            delete p;
        }
    }

public:
    void Push(int val)
    {
        if (m_pStack == nullptr)
        {
            m_pStack = new StackCell;
        }

        // 取当前活动的物理栈, 看还有没有空位
        // 当top == 0的时候，说明当前活动物理栈已经满了，需要重新分配一个物理栈，并链入物理栈链表，成为
        // 新的表头。
        if (m_pStack->top == 0)
        {
            StackCell* pNewCell = new StackCell;
            pNewCell->pNext = m_pStack;
            m_pStack = pNewCell;
        }

        m_pStack->arrEle[--m_pStack->top] = val;
    }

    int Pop()
    {
        // release empty stack cell
        while (m_pStack != nullptr && m_pStack->top == STACK_CELL_SIZE)
        {
            StackCell* p = m_pStack;
            m_pStack = m_pStack->pNext;
            delete p;
        }

        if (m_pStack == nullptr)
        {
            throw new std::bad_alloc();
        }

        return m_pStack->arrEle[m_pStack->top++];
    }
};

