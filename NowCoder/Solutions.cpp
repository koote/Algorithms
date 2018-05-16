#include <vector>
using namespace std;

// 机器人的运动范围
int calcWeight(int x, int y)
{
    int digitSum = 0;
    for (; x > 0; digitSum += x % 10, x /= 10);
    for (; y > 0; digitSum += y % 10, y /= 10);
    return digitSum;
}
void search(vector<vector<int>>& visited, const int threshold, int row, int col, int& result)
{
    if (visited[row][col] == 0)
    {
        visited[row][col] = 1;
        result += 1;

        if (row > 0 && calcWeight(row - 1, col) <= threshold)
        {
            search(visited, threshold, row - 1, col, result);
        }

        if (col > 0 && calcWeight(row, col - 1) <= threshold)
        {
            search(visited, threshold, row, col - 1, result);
        }

        if (row < visited.size() - 1 && calcWeight(row + 1, col) <= threshold)
        {
            search(visited, threshold, row + 1, col, result);
        }

        if (col < visited[0].size() - 1 && calcWeight(row, col + 1) <= threshold)
        {
            search(visited, threshold, row, col + 1, result);
        }
    }
}
int movingCount(int threshold, int rows, int cols)
{
    if (threshold < 0) return 0;
    vector<vector<int>> visited(rows, vector<int>(cols, 0));
    int result = 0;
    search(visited, threshold, 0, 0, result);
    return result;
}
