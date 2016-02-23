#include <string>
#include <algorithm>
#include <vector>
#include <queue>
#include <map>
using namespace std;

// 3# Longest Substring Without Repeating Characters
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

// 6# ZigZag Conversion
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
int reverseInteger(int x)
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

// 8# String to Integer (atoi)
int myAtoi(string str)
{
    int int32max = (1 << 31) - 1;
    int int32min = -(1 << 31);
    int slen = str.length();

    // 1. Trim heading whitespaces.
    int i;
    for (i = 0; i < slen && (str[i] == ' ' || str[i] == '\t' || str[i] == '\n' || str[i] == '\r'); ++i);

    // 2. Process the sign of number.
    int sign = 1;
    if (str[i] == '-')
    {
        sign = -1;
        ++i;
    }
    else if (str[i] == '+')
    {
        ++i;
    }

    // 3. Process overflow or underflow. 
    int r = 0;
    while (i < slen && str[i] >= '0' && str[i] <= '9')
    {
        int digit = str[i] - '0';

        if (sign == 1 && (r > int32max / 10 || (r == int32max / 10 && digit > int32max % 10))) // overflow
        {
            return int32max;
        }
        else if (sign == -1 && (r > 0 - int32min / 10 || (r == 0 - int32min / 10 && digit > 0 - int32min % 10))) //underflow
        {
            return int32min;
        }
        else
        {
            r = r * 10;
            r += digit;
            ++i;
        }
    }

    return r * sign;
}

// 9# Palindrome Number 
bool isPalindrome(int x)
{
    // Leetcode OJ thinks negative integer never could be a palindrome.
    // But I don't agree, sign should not be reversed like digits.
    // This line makes this solution can be accepted by Leetcode OJ.
    // If remove this line, whole function can handle negative integer, it will 
    // think -12321 is a palindrome.

    if (x < 0) return false;

    // get number of digits
    int n = 0;
    for (int y = x; y != 0; y /= 10, ++n);

    int ps = 0;
    int k = n / 2; //compare half of x because reverse X could overflow or underflow.
    while (k > 0)
    {
        int digit = x % 10;
        ps += digit * pow(10, --k);
        x /= 10;
    }

    return n & 0x1 ? ps == x / 10 : ps == x;
}

// 10# Regular Expression Matching
bool isMatch(string s, string p)
{
    // exit of recursion.
    if (p.length() == 0)
    {
        return s.length() == 0;
    }

    if (p.length() > 1 && p[1] == '*') // p[1] is * means p[0] could repeat 0 or any times.
    {
        // try : 
        // if p[0] repeate 1 time, check if s.substr(1) could match p.substr(2);
        // if p[0] repeate 2 times, check if s.substr(2) could match p.substr(2);
        // if p[0] repeate 3 times, check if s.substr(3) could match p.substr(2); 
        // .... 
        // if p[0] repeate k times (k < length of s), check if s.substr(k+1) could match p.substr(2);
        // this process is just to try all possible repeate times of p[0].
        for (int i = 0; i < s.length() && (s[i] == p[0] || p[0] == '.'); ++i)
        {
            if (isMatch(s.substr(i + 1), p.substr(2)))
            {
                return true;
            }
        }

        // if p[0] repeate 0 time, check if s could match p.substr(2);
        return isMatch(s, p.substr(2));
    }
    else // p[1] is either not exists or not a *, just try to match current character and remainings.
    {
        return (p.length() > 0 && s.length() > 0) && (p[0] == '.' || p[0] == s[0]) && isMatch(s.substr(1), p.substr(1));
    }
}

// 11# Container With Most Water
int maxArea(vector<int>& height)
{
    int maxs = 0;
    for (int len = height.size(), i = 0, j = len - 1; i < j; )
    {
        int s = min(height[i], height[j]) * (j - i);
        maxs = max(maxs, s);

        if (height[i] < height[j])
        {
            ++i;
        }
        else
        {
            --j;
        }
    }

    return maxs;
}

// 12# Integer to Roman
string intToRoman(int num)
{
    int n[] = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
    string r[] = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };

    string result;
    for (int k = 1; num != 0; num /= 10, k *= 10)
    {
        int d = (num % 10) * k;

        int i = 0;
        int j = sizeof(n) / sizeof(n[0]) - 1;
        while (j - i > 1)
        {
            int mid = (i + j) / 2;
            if (n[mid] == d)
            {
                result = string(r[mid]).append(result);
                break;
            }
            else if (n[mid] > d)
            {
                i = mid;
            }
            else
            {
                j = mid;
            }
        }

        if (j - i == 1)
        {
            if (n[i] == d)
            {
                result = string(r[i]).append(result);
            }
            else if (n[j] == d)
            {
                result = string(r[i]).append(result);
            }
            else
            {
                int index = 0;
                for (; n[index] != k; ++index);
                result = string(r[j]).append(result);
                for (int count = (d - n[j]) / k; count > 0; --count)
                {
                    result = string(r[index]).append(result);
                }
            }
        }
    }

    return result;
}

// 14# Longest Common Prefix
string longestCommonPrefix(vector<string>& strs)
{
    if (strs.size() == 0)
    {
        return "";
    }

    int index = 0;
    bool flag = true;
    for (index = 0; flag; ++index)
    {
        for (int i = 0, size = strs.size(); i < size; ++i)
        {
            if (index >= strs[i].length() || strs[i][index] != strs[0][index])
            {
                flag = false;
                break;
            }
        }
    }

    return strs[0].substr(0, index - 1);
}

// 15. 3Sum
vector<vector<int>> kSum(vector<int>& nums, unsigned int k, int sum)
{
    const int size = nums.size();
    vector<vector<int>> result(0);

    if (size < k || k == 0)
    {
        return result;
    }

    if (k == 2)
    {
        for (int i = 0, j = size - 1; i < j;)
        {
            int s = nums[i] + nums[j];
            if (s == sum)
            {
                result.push_back(vector<int>{ nums[i], nums[j] });

                // skip duplicates.
                for (++i; i < size && nums[i] == nums[i - 1]; ++i);
                for (--j; j > 0 && nums[j] == nums[j + 1]; --j);
            }
            else if (s > sum)
            {
                for (--j; j > 0 && nums[j] == nums[j + 1]; --j);
            }
            else
            {
                for (++i; i < size && nums[i] == nums[i - 1]; ++i);
            }
        }

        return result;
    }

    for (int i = 0; i < size; ++i)
    {
        // skip duplicates.
        if (i > 0 && nums[i] == nums[i - 1])
        {
            continue;
        }

        // Because nums are sorted, so nums[i+1..i+k-1] are smallest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[i+1] + ... + nums[i+k-1]) > sum, we can give up, because 
        // we cannot find answer, that sum is the smallest we can produce.
        int s = nums[i];
        for (int j = i + 1, p = k - 1; j < size && p > 0; s += nums[j++], --p);
        if (s > sum)
        {
            break;
        }

        // Same reason, nums[size-k+1..size-1] are largest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[size-k+1] + ... + nums[size-1]) < sum, we can ignore current loop (jump over nums[i])
        // because that sum is the largest we can produce in current loop.
        s = nums[i];
        for (int j = size - 1, p = k - 1; j > 0 && p > 0; s += nums[j--], --p);
        if (s < sum)
        {
            continue;
        }

        vector<int> v(0);
        for (int j = i + 1; j < size; v.push_back(nums[j++]));

        vector<vector<int>> cures = kSum(v, k - 1, sum - nums[i]);
        for (int j = 0; j < cures.size(); ++j)
        {
            cures[j].insert(cures[j].begin(), nums[i]);
            result.insert(result.end(), cures[j]);
        }
    }

    return result;
}

vector<vector<int>> threeSum(vector<int>& nums) 
{
    sort(nums.begin(), nums.end());
    return kSum(nums, 3, 0);
}

//16. 3Sum Closest
int threeSumClosest(vector<int>& nums, int target)
{
    const int size = nums.size();
    int result = 0;
    int diff = INT_MAX;

    sort(nums.begin(), nums.end());
    for (int i = 0; i <= size - 3; ++i)
    {
        if (i > 0 && nums[i] == nums[i - 1])
        {
            continue;
        }

        // 2sum in nums[i+1..size-1].
        for (int l = i + 1, r = size - 1; l < r;)
        {
            // Cannot intialize result to INT_MAX then calc difference = abs(result - target) everytime, that could overflow.

            int s = nums[l] + nums[r];
            int d = abs(s - (target - nums[i]));
            if (d < diff)
            {
                result = s + nums[i];
                diff = d;
            }

            if (result == target)
            {
                return result;
            }
            else if (s > target - nums[i])
            {
                for (--r; r > 0 && nums[r] == nums[r + 1]; --r);
            }
            else
            {
                for (++l; l < size && nums[l] == nums[l - 1]; ++l);
            }
        }
    }

    return result;
}

//17. Letter Combinations of a Phone Number
vector<string> letterCombinations(string digits)
{
    //                                0   1     2      3      4      5      6      7       8      9
    const static vector<string> dict{ "", "", "abc", "def", "ghi", "jkl", "mon", "pqrs", "tuv", "xwyz" };
    vector<string> result;

    if (digits.length() == 0)
    {
        return result;
    }

    if (digits.length() == 1)
    {
        string val = dict[digits[0] - '0'];
        for (int i = 0; i < val.length(); ++i)
        {
            result.push_back(val.substr(i, 1));
        }
        return result;
    }

    vector<string> cures = letterCombinations(digits.substr(1));
    string val = dict[digits[0] - '0'];
    for (int i = 0; i < val.length(); ++i)
    {
        for (int j = 0; j < cures.size(); ++j)
        {
            result.push_back(val.substr(i, 1).append(cures[j]));
        }
    }
    return result;
}

//18. 4Sum
vector<vector<int>> fourSum(vector<int>& nums, int target)
{
    sort(nums.begin(), nums.end());
    return kSum(nums, 4, target);
}

//19. Remove Nth Node From End of List
struct ListNode
{
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};
ListNode* removeNthFromEnd(ListNode* head, int n)
{
    ListNode* p;
    for (p = head; n > 0 && p != nullptr; p = p->next, --n);

    if (p == nullptr && n == 0)
    {
        ListNode* r = head->next;
        delete head;
        return r;
    }
    else
    {
        ListNode* q;
        for (q = head; q != nullptr && p != nullptr && p->next != nullptr; q = q->next, p = p->next);

        // q->next is what needs to be removed.
        ListNode* r = q->next;
        q->next = r->next;
        delete r;

        return head;
    }
}
ListNode* removeNthFromEnd2(ListNode* head, int n)
{
    ListNode dummy(-1);
    dummy.next = head;

    ListNode* p;
    for (p = &dummy; n > 0 && p != nullptr; p = p->next, --n);

    ListNode* q;
    for (q = &dummy; q != nullptr && p != nullptr && p->next != nullptr; q = q->next, p = p->next);

    // q->next is what needs to be removed.
    ListNode* r = q->next;
    q->next = r->next;
    delete r;

    return dummy.next;
}
