
#include <stack>

struct STEP
{
    int x;
    int y;
    int d[8];
};

// 提到回溯，一定要马上想到必须用栈

// d, size, direction array and its size
// x, y, position that need to calculate direction
// row, col, grid size, use to detect if we are on edge or corner.
void InitDirection(int d[], int size, int x, int y, int row, int col)
{
    
}

bool FindWord(char** grid, int row, int col, char* word)
{
    if (grid == nullptr || word == nullptr)
    {
        return false;
    }

    char* p = word;
    std::stack<STEP> stkStep;
    int i = 0;
    int j = 0;
    for (i = 0; i < row; ++i)
    {
        for (j = 0; j < col; ++j)
        {
            if (grid[i][j] == *p)
            {
                STEP step;
                step.x = i;
                step.y = j;
                InitDirection(step.d, 8, i, j, row, col);
                stkStep.push(step);
            }
        }
    }

    while (!stkStep.empty() && *(p+1) != '\0')
    {
        STEP curStep = stkStep.top();

        int c = 0;
        int nexti = 0;
        int nextj = 0;
        bool found = false;
        for (int c = 0; c < 8; ++c)
        {
            if (curStep.d[c] != 0) // not visited
            {
                curStep.d[c] = 0; // mark as visited

                switch (c)
                {
                case 1:
                    nexti = curStep.x - 1;
                    nextj = curStep.y - 1;
                    break;

                case 2:
                    nexti = curStep.x - 1;
                    nextj = curStep.y;
                    break;

                case 3:
                    nexti = curStep.x - 1;
                    nextj = curStep.y + 1;
                    break;

                case 4:
                    nexti = curStep.x;
                    nextj = curStep.y - 1;
                    break;

                case 5:
                    nexti = curStep.x;
                    nextj = curStep.y + 1;
                    break;

                case 6:
                    nexti = curStep.x + 1;
                    nextj = curStep.y - 1;
                    break;

                case 7:
                    nexti = curStep.x + 1;
                    nextj = curStep.y;
                    break;

                case 8:
                    nexti = curStep.x + 1;
                    nextj = curStep.y + 1;
                    break;

                default:
                    return false;
                }

                // move on, else search next direction
                if (grid[nexti][nextj] == *(p+1)) 
                {
                    STEP nextStep;
                    nextStep.x = nexti;
                    nextStep.y = nextj;
                    InitDirection(nextStep.d, 8, nexti, nextj, row, col);
                    stkStep.push(nextStep);

                    found = true;
                    ++p; // move to next char

                    break;
                }
            }
        }

        if (!found)
        {
            stkStep.pop();
        }
    }

    return *(p + 1) == '\0';
}
