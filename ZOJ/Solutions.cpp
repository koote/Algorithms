#include <vector>
#include <iostream>

using namespace std;

// ZOJ 1074 - To the Max
int maxSubArray(vector<int>& nums)
{
    int sum = 0;
    int maxSum = INT_MIN;
    for (int num : nums)
    {
        sum = sum > 0 ? sum + num : num;
        if (sum > maxSum)
        {
            maxSum = sum;
        }
    }

    return maxSum;
}
void maxSumSubMatrix()
{
    unsigned size = 0;
    cin >> size;
    vector<vector<int>> matrix(size, vector<int>(size));

    for (unsigned i = 0; i < size; ++i)
    {
        for (unsigned j = 0; j < size; ++j)
        {
            cin >> matrix[i][j];
        }
    }

    int maxSum2d = INT_MIN;
    for (unsigned i = 0; i < matrix.size(); ++i) // select up boundary row: 0 -> size-1
    {
        vector<int> packed(matrix[0].size(), 0);

        for (unsigned j = i; j< matrix.size(); ++j) // select bottom boundary row : i -> size-1.
        {
            // pack row(i) -> row(j) to one dimension array packed : packed = row(i) + row(i+1) + row(i+2) + ... + row(j)
            for (unsigned k = 0; k < matrix[j].size(); ++k)
            {
                packed[k] += matrix[j][k];
            }

            // we get the max submatrix that between row(i) and row(j).
            const int maxSum1d = maxSubArray(packed);
            if (maxSum1d > maxSum2d)
            {
                maxSum2d = maxSum1d;
            }
        }
    }

    cout << maxSum2d;
}

// ZOJ 1163 - The Staircases
// Initial thinking
// This is a DP problem. Thinking in this way, let's say we have N bricks, we can choose to use
// j = {1, 2, ... N-1} bricks to build current stair (using all N bricks is not considered since
// a staircases has only 1 stair is not valid), then problem becomes how many valid staircases can
// be built from remaining N-j bricks, so the key is how to determine how many valid staircases 
// could be built from remaining N-j bricks. For example, when N=4, j=1, we have 3 bricks remaining,
// 3 bricks can build a valid stair but in current case it is invalid, because from top to bottom, 
// stairs has 1|2|1 bricks. So when storing the solution of sub-problems, we cannot just store how
// many valid staircases could be built from a given number of bricks.
//
// This is my first solution. When use j bricks for current stair and N-j bricks remaining, we need
// to know how many staircases can be built from N-j bricks whose longest/bottommost stair's brick 
// count < j. That reminds me to use a 2 dimensions array. Given an element dp[K], it stores the 
// bottommost stair's brick count in every valid staircase built from K bricks. For example, when 
// K=5, dp[K] = {3,4}, they are the bottommost stairs' brick count of 2 staircases could be built 
// from 5 bricks (2|3 and 1|4). Also keep in mind that every dp[K] has a hidden case, that is all 
// K bricks are used to build only 1 stair, this is only valid when we are in middle of calculation.
void staircase1()
{
    int n;
    while (cin >> n && n != 0)
    {
        // every dp[i] is an array, every element of dp[i] is the bottommost stair's brick count.
        vector<vector<int>> dp(n + 1);
        dp[0] = vector<int>();
        for (int i = 0; i <= n; ++i)
        {
            for (int j = 1; j <= i - 1; ++j)
            {
                for (int lastStairBrickCount : dp[i - j])
                {
                    if (j > lastStairBrickCount)
                    {
                        dp[i].push_back(j);
                    }
                }

                if (j > i - j) // check hidden case
                {
                    dp[i].push_back(j);
                }
            }
        }

        cout << dp[n].size() << endl;
    }
}
// The first solution uses too much memory and also slow because it has 3 loops nested. Back to origin,
// when there are i bricks and we decide to use j bricks to construct the bottommost stair, how many
// valid staircases can we get in this case? Obviously it depends on how many valid staircases we can 
// build from remaining i-j bricks whose bottommost stair's brick count <= j-1. So dp[i][j] is defined 
// as: when we have i bricks, the count of staircases whose bottommost stair <= j:
// dp[i][j] = count of staircases has i bricks and bottommost stair's brick count == 0 + 
//            count of staircases has i bricks and bottommost stair's brick count == 1 + 
//            count of staircases has i bricks and bottommost stair's brick count == 2 + 
//            ...
//            count of staircases has i bricks and bottommost stair's brick count == j-1 + 
//            count of staircases has i bricks and bottommost stair's brick count == j
// When calculate from dp[i][0] -> dp[i][i], we keep adding elements, so formula is: 
// dp[i][j] = dp[i-j][j-1] + dp[i][j-1]
// If we explain the formula directly:
// (when there are i bricks, count of staircases whose bottommost stair's brick count <= j) 
//            = (count of staircases whose bottommost stair's brick count == j) +
//              (count of staircases whose bottommost stair's brick count < j) (or <= j-1)
// The difficulty is how to get to this definition of dp[i][j]. We can also define dp[i][j] as: count
// of staircases has i bricks and whose bottommost stair's brick count == j, but then when calculating 
// dp[i][j], we cannot just use value of dp[i-j][j-1], we need to add dp[i-j][0] to dp[i-j][j-1], that
// we still need the third loop.
void staircase2()
{
    int n;
    while (cin >> n && n != 0)
    {
        vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));
        fill(dp[0].begin(), dp[0].end(), 1);
        for (int i = 0; i <= n; ++i)
        {
            // I split the 2nd loop into 2 parts to make it clearly that dp[][] is a echelon matrix.
            for (int j = 1; j <= i; ++j)
            {
                dp[i][j] = dp[i - j][j - 1] + dp[i][j - 1];
            }

            for (int j = i + 1; j <= n; ++j)
            {
                dp[i][j] = dp[i][j - 1];
            }
        }

        // Why need to minus 1? Because dp[n][n] contains a condition that all n bricks are used to build
        // the bottommost stair, such a staircase has only 1 stair, this is allowed when building staircase
        // is a part of bigger staircase (i<n), but not allowed if n is our target.
        cout << dp[n][n] - 1 << endl;
    }
}
// The third solution is focus on space complexity optimization. In solution 2 we use a 2 dimensions array,
// we can get:
// (1) diagonal elements are the finally answers after whole matrix was filled
// (2) when calculating a column dp[i][j], only its previous column (dp[i-j][j-1]) is used
// So it is possible to use 1 dimension array.
