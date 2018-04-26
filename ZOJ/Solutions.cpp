#include <vector>
#include <iostream>

using namespace std;

// ZOJ 1163
// Initial thinking
// This is a DP problem. Thinking in this way, let's say we have N bricks, we can choose to use
// j = {1, 2, ... N-1} bricks to build current stair (using all N bricks is not considered since
// a staircases has only 1 stair is not valid), then problem becomes how many valid staircases can
// be built from remaining N-j bricks, so the key is how to determine how many valid staircases 
// could be built from remaining N-j bricks. For example, when N=4, j=1, we have 3 bricks remaining,
// 3 bricks can build a valid stair but in current case it is invalid, because from top to bottom, 
// stairs has 1|2|1 bricks. So when storing the solution of subproblems, we cannot just store how
// many valid staircases could be built from a given number of bricks.

// This is my first solution. When we use j bricks for current stair and N-j bricks remaining, we need
// to know how many staircases that built from N-j bricks have its longest/bottommost stair's length < j.
// That reminds me to use a 2 dimensions array. Given an element dp[K], it stores the bottommost stair's 
// brick count in every valid stairscse built from K bricks. For example, when K=5, dp[K] = {3,4}, they 
// are the bottommost stairs' brick count of 2 staircases could be built from 5 bricks (2|3 and 1|4).
// Also keep in mind that every dp[K] has a hidden case, that is all K bricks are used to build 
// only 1 stair, this is only valid when we are in middle of calculation.
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
                for (auto lastStairBrickCount : dp[i - j])
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
// when there are i bricks and we decide to use j bricks to construct the first stair, how many valid 
// staircases can we get in this case? Obviously it depends on how many valid staircases we can build 
// using remaining i-j bricks whose bottommost stair's brick count <= j-1. So dp[i][j] is defined as:
// when we have i bricks, the count of staircases whose bottommost stair <= j:
// dp[i][j] = count of staircases has i bricks and bottommost stair's brick count == 0 + 
//            count of staircases has i bricks and bottommost stair's brick count == 1 + 
//            count of staircases has i bricks and bottommost stair's brick count == 2 + 
//            ...
//            count of staircases has i bricks and bottommost stair's brick count == j-1 + 
//            count of staircases has i bricks and bottommost stair's brick count == j
// When calculate from dp[i][0] -> dp[i][i], we keep adding elements, so formular is: 
// dp[i][j] = dp[i-j][j-1] + dp[i][j-1]
// If we explain the formular directly:
// (count of staircases have i bricks and bottommost stair <= j)  = 
//              (count of staircases has i bricks and bottommost stair == j) +
//              (count of staircases has i bricks and bottommost stair <= j-1)
// The difficuly is how to define dp[i][j]. We can also define dp[i][j] as: count of staircases has i
// bricks and bottommost stair's brick count == j, but then when calculating dp[i][j], we cannot just 
// use value of dp[i-j][j-1], we need to add dp[i-j][0] to dp[i-j][j-1], that we need third loop.
void staircase2()
{
    int n;
    while (cin >> n && n != 0)
    {
        vector<vector<long long>> dp(n + 1, vector<long long>(n + 1, 0));
        fill(dp[0].begin(), dp[0].end(), 1);
        for (int i = 0; i <= n; ++i)
        {
            for (int j = 1; j <= i; ++j)
            {
                dp[i][j] = dp[i - j][j - 1] + dp[i][j - 1];
            }

            for (int j = i + 1; j <= n; ++j)
            {
                dp[i][j] = dp[i][j - 1];
            }
        }

        cout << dp[n][n] - 1 << endl;
    }
}


// The third solution is foucs on space optimization. We can see that actually
