#include <string>
#include <algorithm>
#include <vector>
#include <queue>
using namespace std;

// 3#. Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s)
{
    int i = 0, maxlen = 0, n = s.length();
    int loc[256];

    while (i < n)
    {
        // no element's position is n, use this to initialize loc array.
        std::fill_n(loc, 256, n);

        int j = i;
        while (j < n)
        {
            if (loc[s[j]] == n)
            {
                loc[s[j]] = j;
                ++j;
            }
            else
            {
                break;
            }
        }

        // now j either points to first repeat or null terminator.
        int current_subtring_len = j - i;
        if (current_subtring_len > maxlen)
        {
            maxlen = current_subtring_len;
        }

        // we don't need to check if inner loop ends because j reach the end of string or
        // found a repeat character, if j reach end, loc[s[j]] + 1 > n, outer loop will also
        // end, if found a repeat, loc[s[j]] + 1 is the new substring start position.
        i = loc[s[j]] + 1;
    }

    return maxlen;
}

// 4# Median of Two Sorted Arrays
double findKth(int a[], int m, int b[], int n, int k)
{
    if (m < n) { return findKth(b, n, a, m, k); }
    if (n == 0) { return a[k - 1]; }
    if (k == 1) { return min(a[0], b[0]); }
    if (k == m + n) { return max(a[m - 1], b[n - 1]); }

    int j = min(n, k / 2);
    int i = k - j;

    if (a[i - 1] > b[j - 1])
    {
        return findKth(a, i, b + j, n - j, k - j);
    }
    else if (a[i - 1] < b[j - 1])
    {
        return findKth(a + i, m - i, b, j, k - i);
    }
    else
    {
        return a[i - 1];
    }
}

double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
    int m = nums1.size();
    int n = nums2.size();

    if ((m + n) & 1) // odd
    {
        return findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2 + 1);
    }
    else //even
    {
        return (findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2) + findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2 + 1)) / 2.0;
    }
}

// 5# Longest Palindromic Substring
string trySearchPalindromic(string s, int l, int r)
{
    int slen = s.length();
    while (l >= 0 && r <= slen - 1 && s[l] == s[r])
    {
        --l;
        ++r;
    }

    return s.substr(l + 1, r - l - 1);
}

string longestPalindrome(string s) 
{
    int slen = s.length();
    if (slen == 0)
    {
        return "";
    }

    string longest = s.substr(0, 1);

    for (int i = 0; i <= slen - 1; ++i)
    {
        string p1 = trySearchPalindromic(s, i, i);
        if (p1.length() > longest.length())
        {
            longest = p1;
        }

        string p2 = trySearchPalindromic(s, i, i + 1);
        if (p2.length() > longest.length())
        {
            longest = p2;
        }
    }

    return longest;
}

//6# ZigZag Conversion
string zigzagConvert(string s, int numRows)
{
    int len = s.length();

    if (numRows <= 1 || numRows >= len)
    {
        return s;
    }

    string res;
    int delta = 2 * numRows - 2;
    for (int r = 0; r <= numRows - 1; ++r)
    {
        res += s[r];

        for (int k = r + delta; k - 2 * r < len; k += delta)
        {
            if (r >= 1 && r <= numRows - 2)
            {
                res += s[k - 2 * r];
            }

            if (k < len)
            {
                res += s[k];
            }
        }
    }

    return res;
}

// 7# Reverse Integer
int reverse(int x)
{
    int sign = x < 0 ? -1 : 1;
    x = x < 0 ? -x : x;

    queue<int> digits;
    while (x > 0)
    {
        digits.push(x % 10);
        x = x / 10;
    }

    // Key is considering overflow and nth power of 10.
    // nth power of 10 doesn't need special handling.
    int result = 0;
    while (!digits.empty() && result <= ((1 << 31) - 1 - digits.front()) / 10)
    {
        result *= 10;
        result += digits.front();

        digits.pop();
    }

    return digits.empty() ? result * sign : 0;
}
