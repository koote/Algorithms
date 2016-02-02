#include <stack>

struct COORDINATE
{
public:
    COORDINATE(int _x, int _y)
    {
        this->x = _x;
        this->y = _y;
    }

    int x;
    int y;
};

struct STEP
{
    COORDINATE coor;
    int directions[8];
    
public:
    STEP(COORDINATE& c) : coor(c.x, c.y)
    {
        InitializeDirections();
    }

private:
    void InitializeDirections()
    {

    }
};

bool FindPath(int** maze, int m, int n, COORDINATE start, COORDINATE end)
{
    if (maze == nullptr || m == 0 || n == 0)
    {
        return false;
    }

    std::stack<STEP> stkSteps;
    STEP s(start);
    stkSteps.push(s);

    while(!stkSteps.empty())
    {
        STEP cur = stkSteps.top();
        bool bMoveNext = false;

        // try directions that not visited.
        for (int d = 0; d < 8; ++d)
        {
            int nexti = 0;
            int nextj = 0;
            if (cur.directions[d] == 0) // 1 means visited, 0 means not visited
            {
                cur.directions[d] = 1; //mark as visited.

                switch (d)
                {
                case 1:
                    nexti = cur.coor.x - 1;
                    nextj = cur.coor.y - 1;
                    break;

                case 2:
                    nexti = cur.coor.x - 1;
                    nextj = cur.coor.y;
                    break;

                case 3:
                    nexti = cur.coor.x - 1;
                    nextj = cur.coor.y + 1;
                    break;

                case 4:
                    nexti = cur.coor.x;
                    nextj = cur.coor.y + 1;
                    break;

                case 5:
                    nexti = cur.coor.x + 1;
                    nextj = cur.coor.y + 1;
                    break;

                case 6 :
                    nexti = cur.coor.x + 1;
                    nextj = cur.coor.y;
                    break;

                case 7:
                    nexti = cur.coor.x + 1;
                    nextj = cur.coor.y - 1;
                    break;

                case 8:
                    nexti = cur.coor.x;
                    nextj = cur.coor.y - 1;
                    break;
                }

                // move on to next position
                if (end.x == nexti && end.y == nextj) //got it!
                {
                    // print out the path and return;
                    return true;
                }
                
                if (maze[nexti][nextj] == 0)
                {
                    // For every position in maze, its value could be 1, -1 or 0.
                    // 0 means it is a safe position, can move on it.
                    // -1 mean it is a wall.
                    // 1 is temporary, it means this position is still in stack, it is a part of current path.
                    // in the further search, we don't want to cross the path.

                    maze[nexti][nextj] = 1;
                    bMoveNext = true;
                    STEP next(COORDINATE(nexti, nextj));
                    stkSteps.push(next);

                    break;
                }
            }
        }

        if (!bMoveNext) // we cannot move to next position, need to pop and go back
        {
            // before we popup and roll back, we need to revert the value of this position in maze from 1 to 0.
            // it is safe to to that, because this direction is already marked as visited, so even its value is 0,
            // next time we will not move to this position again.
            maze[cur.coor.x][cur.coor.y] = 0;
            stkSteps.pop();
        }
    }

    return false;
}
