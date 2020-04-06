#include <stack>
#include <queue>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "DataStructure.h"

using namespace std;

// 1. Two Sum
vector<int> twoSum(vector<int>& nums, const int target)
{
    unordered_map<int, int> dictionary; // <value, index>
    unordered_map<int, int>::iterator found;

    for (unsigned i = 0; i < nums.size(); ++i)
    {
        if ((found = dictionary.find(target - nums[i])) != dictionary.end())
        {
            return { static_cast<int>(i), found->second };
        }

        dictionary.insert(make_pair(nums[i], i)); // use value as dictionary key, so we can quick find if a value exists
    }

    return {};
}

// 2. Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
{
    ListNode dummyHead(0);
    int carry = 0;
    for (ListNode* a = l1, *b = l2, *last = &dummyHead; a != nullptr || b != nullptr || carry != 0; a = a == nullptr ? a : a->next, b = b == nullptr ? b : b->next, last = last->next)
    {
        const int val = (a == nullptr ? 0 : a->val) + (b == nullptr ? 0 : b->val) + carry;
        last->next = new ListNode(val % 10); // when loop ends, no need to seal list since when ListNode is created, its next is set to nullptr.
        carry = val / 10;
    }

    return dummyHead.next;
}

// 3. Longest Substring Without Repeating Characters
int lengthOfLongestSubstring(string s)
{
    int maxlen = 0;
    int existence[256];
    fill_n(existence, 256, -1);

    for (int i = 0; i < s.length();)
    {
        for (int j = i; j < s.length(); ++j)
        {
            if (existence[s[j]] < i)
            {
                existence[s[j]] = j;
                maxlen = max(j - i + 1, maxlen);
            }
            else
            {
                i = existence[s[j]] + 1;
                fill_n(existence, 256, -1);
                break;
            }
        }
    }

    return maxlen;
}

// 4. Median of Two Sorted Arrays
double findKthSortedArrays(vector<int>& nums1, vector<int>& nums2, const unsigned k) // Note k starts from 1.
{
    // we assume that nums1 is longer than nums2.
    if (nums1.size() < nums2.size())
    {
        return findKthSortedArrays(nums2, nums1, k);
    }

    if (nums2.empty())
    {
        return nums1[k - 1];
    }

    if (k == 1)
    {
        return min(nums1.front(), nums2.front());
    }

    if (k == nums1.size() + nums2.size())
    {
        return max(nums1.back(), nums2.back());
    }

    // Ideally every step we reduce k by half, however since we assume that nums1.size() >= nums2.size(), so
    // it is possible that k/2 > nums2.size(), to ensure that it doesn't cause out of boundary exception, we
    // compare k/2 and nums2.size().
    const unsigned j = min(k / 2, nums2.size());
    const unsigned i = k - j;

    // now we get an index i in nums1 and index j in nums2. So we can compare nums1[i-1] and nums2[j-1] (Note that k is
    // one based, so i and j are also 1 based). There are 3 cases:
    // (1) nums1[i-1] < nums2[j-1]. It means nums1[0..i-1] < nums2[j..nums2.size()-1], and k-th element cannot be in
    //     these 2 ranges, according to k-th element's definition, its left side has (i+j)-1 elements and its right side
    //     has nums1.size()+nums2.size()-(i+j) elements, it is obvious that nums1[0..i-1] has i elements and nums2[j..nums2.size()-1]
    //     has nums2.size()-j elements. So we can discard nums1's first range and nums2's second range, then continue searching.
    //     Must note that next time we are searching for (k-i)-th element.
    // (2) nums1[i-1] > nums2[j-1]. It means nums1[i..nums1.size()-1] > nums2[0..j-1], similarly, we can discard nums1's 
    //     second part and nums2's first part, and continue searching (k-j)-th element.
    // (3) nums1[i-1] == nums2[j-1], it means they have k-2 elements left and m+n-k elements right, so either is the k-th.
    if (nums1[i - 1] < nums2[j - 1])
    {
        vector<int> subnums1(nums1.begin() + i, nums1.end());
        vector<int> subnums2(nums2.begin(), nums2.begin() + j);
        return findKthSortedArrays(subnums1, subnums2, k - i);
    }

    if (nums1[i - 1] > nums2[j - 1])
    {
        vector<int> subnums1(nums1.begin(), nums1.begin() + i);
        vector<int> subnums2(nums2.begin() + j, nums2.end());
        return findKthSortedArrays(subnums1, subnums2, k - j);
    }

    return nums1[i - 1];
}
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
    const int m = nums1.size();
    const int n = nums2.size();

    // Note median index starts from 1, not 0.
    if (m + n & 1) // odd
    {
        return findKthSortedArrays(nums1, nums2, (m + n) / 2 + 1);
    }

    // even, return arithmetic average of two medians.
    return (findKthSortedArrays(nums1, nums2, (m + n) / 2) + findKthSortedArrays(nums1, nums2, (m + n) / 2 + 1)) / 2.0;
}

// 5. Longest Palindromic Substring
string searchPalindrome(string s, int left, int right)
{
    // when loop ends, no matter it is because of out of bounds or s[left] and s[right] no longer equal, current palindrome is always s[left+1 .. right-1].
    for (; left >= 0 && right < s.length() && s[left] == s[right]; left--, right++);
    return s.substr(left + 1, right - 1 - left);
}
string longestPalindrome(string s)
{
    string result;
    for (int position = 0; position < s.length(); ++position)
    {
        string palindrome1 = searchPalindrome(s, position, position);
        if (palindrome1.length() > result.length())
        {
            result = palindrome1;
        }

        // don't worry that position+1 could exceed right bound, since searchPalindrome will check j < s.length()
        string palindrome2 = searchPalindrome(s, position, position + 1);
        if (palindrome2.length() > result.length())
        {
            result = palindrome2;
        }
    }

    return result;
}

// 6. ZigZag Conversion
string zigzagConvert(string s, const int numRows)
{
    // We can also handle condition here when numRows>=s.length(), return s directly, but it has been covered in following loop.
    // For condition numRows == 1, because 2*numRows-2=0, if we let distance = 1 when, then it could also be handled by following loop.
    if (numRows == 1)
    {
        return s;
    }

    string result;
    const int distance = 2 * numRows - 2;
    for (int r = 0; r < numRows; ++r)
    {
        const int curDistance1 = distance - r * 2;
        for (int j = r; j < s.length(); j += distance)
        {
            result += s[j];
            if (curDistance1 < distance && curDistance1 > 0 && j + curDistance1 < s.length())
            {
                result += s[j + curDistance1];
            }
        }
    }

    return result;
}

// 7. Reverse Integer
int reverseInteger(int x)
{
    const int sign = x < 0 ? -1 : 1;
    x = x < 0 ? -x : x;
    int result = 0;
    for (; x > 0; x /= 10)
    {
        const int digit = x % 10;
        if (result > INT_MAX / 10 ||
            (result == INT_MAX / 10 && digit > INT_MAX % 10 && sign == 1) ||
            (result == INT_MAX / 10 && digit > INT_MAX % 10 + 1 && sign == -1))
        {
            return 0;
        }

        result = result * 10 + digit;
    }

    return result * sign;
}

// 8. String to Integer (atoi)
int myAtoi(string str)
{
    int i;
    for (i = 0; i < str.length() && str[i] == ' '; ++i);

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

    int total = 0;
    for (; i < str.length() && str[i] >= '0' && str[i] <= '9'; ++i)
    {
        const int digit = str[i] - '0';

        // check overflow
        if (total > INT_MAX / 10 ||
            (total == INT_MAX / 10 && digit > INT_MAX % 10 && sign == 1) ||
            (total == INT_MAX / 10 && digit > INT_MAX % 10 + 1 && sign == -1))
        {
            return sign == 1 ? INT_MAX : INT_MIN;
        }

        total = total * 10 + digit;
    }

    return total * sign;
}

// 9. Palindrome Number
bool isPalindrome(int x)
{
    if (x < 0)
    {
        return false;
    }

    // First calculate how many digits the number x has.
    int numOfDigits = 0;
    for (int temp = x; temp > 0; temp /= 10, ++numOfDigits);

    // calculate low half value of x. Lets say x = 12321, lower half is 21.
    int lowhalf = 0;
    for (int i = 1, k = numOfDigits / 2; i <= k; lowhalf = lowhalf * 10 + x % 10, x /= 10, ++i);

    // after above loop, we got low half value, then we need to get high half value and compare.
    return (numOfDigits & 0x1 ? x / 10 : x) == lowhalf;
}

// 10. Regular Expression Matching
bool isMatch_Regex(string text, string pattern)
{
    // Exit condition should be: when pattern is all processed, text must also be all processed,
    // but not vice versa, because when text is empty, if pattern is a * or .*, they still match.
    if (pattern.length() == 0)
    {
        return text.length() == 0;
    }

    // If there is a * followed, there are 2 cases:
    // (1) the preceding character of * appears zero times, next match would be isRegexMatch(text, pattern.substr(2)).
    // (2) the preceding character of * appears >= 1 times, next match would be isRegexMatch(text.substr(1), pattern.substr(1)) if current character matches pattern.
    return pattern.length() > 1 && pattern[1] == '*' ?
        isMatch_Regex(text, pattern.substr(2)) || (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch_Regex(text.substr(1), pattern) :
        (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch_Regex(text.substr(1), pattern.substr(1));
}

// 11. Container With Most Water
int maxArea(vector<int>& height)
{
    int maxArea = 0;
    for (int i = 0, j = height.size() - 1, currentArea; i < j;)
    {
        if (height[i] < height[j])
        {
            currentArea = (j - i) * height[i];
            ++i;
        }
        else
        {
            currentArea = (j - i) * height[j];
            --j;
        }

        if (currentArea > maxArea)
        {
            maxArea = currentArea;
        }
    }

    return maxArea;
}

// 12. Integer to Roman
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
            const int mid = (i + j) / 2;
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

//13. Roman to Integer
int romanToInt(string s)
{
    throw new exception();
}

// 14. Longest Common Prefix
string longestCommonPrefix(vector<string>& strs)
{
    string prefix;
    for (unsigned i = 0, j = 0; !strs.empty() && (i == 0 || j == strs.size()); prefix.append(1, strs[0][i++]))
    {
        for (j = 0; j < strs.size() && i < strs[j].length() && strs[j][i] == strs[0][i]; ++j);
    }

    return prefix.empty() ? prefix : prefix.substr(0, prefix.length() - 1);
}

// 15. 3Sum
vector<vector<int>> kSum(vector<int>& nums, const unsigned k, const int sum) // nums must be sorted.
{
    vector<vector<int>> result(0);

    if (nums.size() < k || k == 0)
    {
        return result;
    }

    if (k == 2)
    {
        for (unsigned i = 0, j = nums.size() - 1; i < j;)
        {
            if (nums[i] + nums[j] > sum)
            {
                --j;
            }
            else if (nums[i] + nums[j] < sum)
            {
                ++i;
            }
            else
            {
                result.push_back(vector<int>{ nums[i], nums[j] });

                // skip duplicates.
                for (++i; i < nums.size() && nums[i] == nums[i - 1]; ++i);
                for (--j; j > 0 && nums[j] == nums[j + 1]; --j);
            }
        }

        return result;
    }

    for (int i = 0; i < nums.size() - k + 1;)
    {
        /*
        // Because nums are sorted, so nums[i+1..i+k-1] are smallest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[i+1] + ... + nums[i+k-1]) > sum, we can give up, because
        // we cannot find answer, that sum is the smallest we can produce.
        int s = nums[i];
        for (unsigned j = i + 1, p = k - 1; j < nums.size() && p > 0; s += nums[j++], --p);
        if (s > sum)
        {
            for (++i; i < nums.size() && nums[i - 1] == nums[i]; ++i);
            break;
        }

        // Same reason, nums[size-k+1..size-1] are largest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[size-k+1] + ... + nums[size-1]) < sum, we can ignore current loop (jump over nums[i])
        // because that sum is the largest we can produce in current loop.
        s = nums[i];
        for (unsigned j = nums.size() - 1, p = k - 1; j > 0 && p > 0; s += nums[j--], --p);
        if (s < sum)
        {
            for (++i; i < nums.size() && nums[i - 1] == nums[i]; ++i);
            break;
        }
        */

        // downgrade current k sum problem to k-1 sum problem
        vector<int> temp(nums.begin() + i + 1, nums.end());
        vector<vector<int>> cures = kSum(temp, k - 1, sum - nums[i]);
        for (auto& cure : cures)
        {
            cure.insert(cure.begin(), nums[i]);
            result.insert(result.end(), cure);
        }

        for (++i; i < nums.size() && nums[i - 1] == nums[i]; ++i);
    }

    return result;
}
vector<vector<int>> threeSum(vector<int>& nums)
{
    sort(nums.begin(), nums.end());
    return kSum(nums, 3, 0);
}

// 16. 3Sum Closest
int threeSumClosest(vector<int>& nums, int target)
{
    int sum = 0;
    int diff = INT_MAX; // Don't initialize sum to INT_MAX then calculate difference = abs(sum - target), that could overflow.
    sort(nums.begin(), nums.end());

    for (int i = 0; i < nums.size() - 2;)
    {
        for (int j = i + 1, k = nums.size() - 1; j < k && j < nums.size() && k >= 0;)
        {
            const int cursum = nums[j] + nums[k] + nums[i];

            if (cursum > target)
            {
                if (cursum - target < diff)
                {
                    diff = cursum - target;
                    sum = cursum;
                }

                for (--k; k >= 0 && nums[k] == nums[k + 1]; --k);
            }
            else if (cursum < target)
            {
                if (target - cursum < diff)
                {
                    diff = target - cursum;
                    sum = cursum;
                }

                for (++j; j < nums.size() && nums[j] == nums[j - 1]; ++j);
            }
            else
            {
                return target;
            }
        }

        for (++i; i < nums.size() && nums[i] == nums[i - 1]; ++i);
    }

    return sum;
}

// 17. Letter Combinations of a Phone Number
vector<string> letterCombinationsUseRecursion(const string& digits)
{
    const string mapping[9] = { "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    vector<string> result;
    if (!digits.empty())
    {
        vector<string> rest = letterCombinationsUseRecursion(digits.substr(1));
        string first = mapping[digits[0] - '0' - 1];
        for (unsigned i = 0; i < first.length(); ++i)
        {
            string s = first.substr(i, 1);
            if (!rest.empty())
            {
                for (unsigned j = 0; j < rest.size(); ++j)
                {
                    result.push_back(s + rest[j]);
                }
            }
            else
            {
                result.push_back(s);
            }
        }
    }

    return result;
}
vector<string> letterCombinationsUseDP(const string& digits)
{
    const string mapping[9] = { "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    vector<vector<string>> dp(digits.length() + 1);
    for (int i = digits.length() - 1; i >= 0; --i)
    {
        string current = mapping[digits[i] - '0' - 1];
        if (dp[i + 1].empty())
        {
            for (unsigned j = 0; j < current.length(); ++j)
            {
                dp[i].push_back(current.substr(j, 1));
            }
        }
        else
        {
            for (unsigned j = 0; j < current.length(); ++j)
            {
                string s = current.substr(j, 1);
                for (unsigned k = 0; k < dp[i + 1].size(); ++k)
                {
                    dp[i].push_back(s + dp[i + 1][k]);
                }
            }
        }
    }

    return dp[0];
}
vector<string> letterCombinations(const string& digits)
{
    return letterCombinationsUseDP(digits);
}

// 18. 4Sum
vector<vector<int>> fourSum(vector<int>& nums, const int target)
{
    sort(nums.begin(), nums.end());
    return kSum(nums, 4, target);
}

// 19. Remove Nth Node From End of List
ListNode* removeNthFromEnd(ListNode* head, int n)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    // The algorithms is to use two pointers and let distance between first and last is n+1, when first reaches 
    // end of list, last->next is the one needs to be deleted.
    // We start counting from dummy other than real head, this can solve the condition that list length is 1 and
    // n == 1, pointer first stops at real head, pointer last points to dummy head, no special handling is needed.
    // If we start counting from real head, then when first loop ends, first is null.
    ListNode* first;
    for (first = &dummyHead; n > 0 && first != nullptr; first = first->next, --n);

    ListNode* last;
    for (last = &dummyHead; first->next != nullptr; first = first->next, last = last->next);

    // last->next is what needs to be removed.
    ListNode* target = last->next;
    last->next = target->next;
    delete target;

    return dummyHead.next;
}

// 20. Valid Parentheses
bool isValid(string s)
{
    stack<char> stack;
    for (char i : s)
    {
        if (s[i] == '{' || s[i] == '[' || s[i] == '(')
        {
            stack.push(s[i]);
        }
        else
        {
            if (!stack.empty())
            {
                const char top = stack.top();
                if ((s[i] == ')' && top == '(') || (s[i] == ']' && top == '[') || (s[i] == '}' && top == '{'))
                {
                    stack.pop();
                    continue;
                }
            }

            return false;
        }
    }

    return stack.empty();
}

// 21. Merge Two Sorted Lists
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2)
{
    ListNode dummyHead(0);
    for (ListNode* last = &dummyHead; l1 != nullptr || l2 != nullptr;)
    {
        if (l1 != nullptr && l2 != nullptr)
        {
            if (l1->val < l2->val)
            {
                last->next = l1;
                l1 = l1->next;
            }
            else
            {
                last->next = l2;
                l2 = l2->next;
            }

            last = last->next;
        }
        else if (l1 != nullptr)
        {
            last->next = l1;
            break;
        }
        else if (l2 != nullptr)
        {
            last->next = l2;
            break;
        }
    }

    return dummyHead.next;
}

// 22. Generate Parentheses
vector<string> generateParenthesisBottomUp(int n)
{
    vector<vector<string>> dp(n + 1);
    dp[0].push_back("");
    for (int i = 1; i <= n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            // For every string in dp[j] and dp[i-j-1], concatenate a new string (dp[j])+dp[i-j-1]
            for (const string& x : dp[j])
            {
                for (const string& y : dp[i - j - 1])
                {
                    dp[i].push_back(('(' + x + ')').append(y));
                }
            }
        }
    }

    return dp[n];
}
vector<string> generateParenthesisBacktracking(const string trail, int remainingLeftBrackets, int remainingRightBrackets)
{
    vector<string> results;

    if (remainingLeftBrackets == 0 && remainingRightBrackets == 0)
    {
        results.push_back(trail);
        return results;
    }

    // Solution space is a binary tree. But this problem is a little different from other similar problems (e.g. letterCombinations),
    // that is when searching in the solution tree, remaining left brackets must not be more than remaining right brackets.
    if (remainingLeftBrackets <= remainingRightBrackets)
    {
        if (remainingLeftBrackets > 0)
        {
            // NOTE: When calling deeper, always use temp variable, DO NOT change the value of parameter in current level.
            // E.g. DO NOT use trail.append or --remainingLeftBrackets here.
            vector<string> cures1 = generateParenthesisBacktracking(trail + '(', remainingLeftBrackets - 1, remainingRightBrackets);
            results.insert(results.end(), cures1.begin(), cures1.end());
        }

        if (remainingRightBrackets > 0)
        {
            vector<string> cures2 = generateParenthesisBacktracking(trail + ')', remainingLeftBrackets, remainingRightBrackets - 1);
            results.insert(results.end(), cures2.begin(), cures2.end());
        }
    }

    return results;
}
vector<string> generateParenthesis(const int n)
{
    const vector<string> r1 = generateParenthesisBacktracking("", n, n);
    vector<string> r2 = generateParenthesisBottomUp(n);
    assert(r1 == r2);
    return r2;
}

// 23. Merge k Sorted Lists
void minHeapify(vector<ListNode*>& nodes, const int root)
{
    const int leftChild = root * 2 + 1;
    const int rightChild = root * 2 + 2;
    int smallest = root;

    if (leftChild < nodes.size() && nodes[leftChild]->val < nodes[smallest]->val)
    {
        smallest = leftChild;
    }

    if (rightChild < nodes.size() && nodes[rightChild]->val < nodes[smallest]->val)
    {
        smallest = rightChild;
    }

    if (smallest != root)
    {
        ListNode* temp = nodes[root];
        nodes[root] = nodes[smallest];
        nodes[smallest] = temp;

        minHeapify(nodes, smallest);
    }
}
void buildMinHeap(vector<ListNode*>& nodes)
{
    for (int i = nodes.size() / 2 - 1; i >= 0; --i)
    {
        minHeapify(nodes, i);
    }
}
ListNode* mergeKLists(vector<ListNode*>& lists)
{
    vector<ListNode*> minHeap;

    // Build a min heap by pushing first element of every list into heap. Because each element
    // is still chained in its list, so we can also think every whole list is pushed into heap.
    for (ListNode* list : lists)
    {
        if (list != nullptr)
        {
            minHeap.push_back(list);
        }
    }

    buildMinHeap(minHeap);

    ListNode dummyHead(0);
    ListNode* last = &dummyHead;
    while (!minHeap.empty())
    {
        last->next = minHeap[0];
        last = last->next;

        // Keep taking elements from current list.
        minHeap[0] = minHeap[0]->next;

        // When one list's elements are all processed, move it to the end of vector and pop out.
        // Do not erase from vector's head, that shifts all elements left by 1, heap needs rebuild.
        if (minHeap[0] == nullptr)
        {
            minHeap[0] = minHeap[minHeap.size() - 1];
            minHeap.pop_back();
        }

        // No matter if current list is empty or not, we only change the first element, so
        // always adjust heap from root.
        minHeapify(minHeap, 0);
    }

    last->next = nullptr; // seal the list
    return dummyHead.next;
}

// 24. Swap Nodes in Pairs
ListNode* swapPairs(ListNode* head)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    // What will be swapped are p->next and q->next.
    // check q is null is only for empty list. check q->next is null is really needed for non empty list.
    for (ListNode* p = &dummyHead, *q = head; q != nullptr && q->next != nullptr;)
    {
        p->next = q->next;
        p = p->next; // Moving p forward makes code looks a little bit clean, otherwise it would be "q->next = p->next->next;p->next->next = q".
        q->next = p->next;
        p->next = q;

        p = q;
        q = p->next;
    }

    return dummyHead.next;
}

//25. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, const int k)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    for (ListNode* p = &dummyHead, *q = head; q != nullptr;)
    {
        // Moving q forward by k-1 elements.
        int n;
        for (n = k - 1; n > 0 && q != nullptr; --n, q = q->next);

        if (q != nullptr)
        {
            // now reverse segment (p .. q]
            ListNode* r = p->next;
            p->next = q;
            p = r->next;
            r->next = q->next;

            // reuse q and p for linked list reversing. p is q's succeeder
            for (n = k - 1, q = r; n > 0; --n)
            {
                ListNode* temp = p->next;
                p->next = q;

                q = p;
                p = temp;
            }

            p = r;
            q = p->next;
        }
    }

    return dummyHead.next;
}

// 26. Remove Duplicates from Sorted Array
// Here is the generic solution for remove duplicates (only allow k duplicates) from sorted array. It leverages the truth
// that array is sorted. For every number, it can only duplicated for k times, problem 26 is when k==1, problem 80 is when
// k==2.
// nums[0..last-1] is the range that have been packed. Back to the truth of array is sorted and a number is only allowed to
// duplicate for k times, so we have two facts:
// (1) When last < k, it is impossible to have a number duplicated more than k times, just append num to end of packed range.
// (2) When last >= k, means packed range already has no less than k elements, need to check before appending num, here is
//     the key point, since num[0 .. last-1] is also sorted, so we compare num[last-k] and num. if num[last-k] == num, it means
//     if we put num on nums[last], nums[last-k .. last] are all equal to num, so num is duplicated k+1 times, which is not 
//     allowed, so num should be skipped and last keeps unchanged; otherwise if num[last-k] < num, it is safe to append num.
// At last since last always points to the next location of end of packed range, so last is the packed range length.
int removeDuplicatesK(vector<int>& nums, const int k)
{
    int last = 0;
    for (int num : nums)
    {
        if (last < k || num > nums[last - k])
        {
            nums[last++] = num;
        }
    }

    return last;
}
int removeDuplicates(vector<int>& nums)
{
    return removeDuplicatesK(nums, 1);
}

// 27. Remove Element
int removeElement(vector<int>& nums, const int val)
{
    int last = 0;
    for (unsigned probe = 0; probe < nums.size(); ++probe)
    {
        if (nums[probe] != val)
        {
            nums[last++] = nums[probe];
        }
    }

    return last;
}

// 28. Implement strStr()
int strStr(string text, string pattern)
{
    for (unsigned i = 0, j; text.length() >= pattern.length() && i <= text.length() - pattern.length(); ++i)
    {
        for (j = 0; j < pattern.length() && text[j + i] == pattern[j]; ++j);
        if (j == pattern.length())
        {
            return i;
        }
    }

    return -1;
}

// 29. Divide Two Integers
int divide2(const int dividend, const int divisor)
{
    unsigned quotient = 0;

    for (unsigned a = dividend < 0 ? -dividend : dividend, b = divisor < 0 ? -divisor : divisor, k; a >= b; a -= b << k)
    {
        // We are looking for a k that b << k <= a but b << (k + 1) > a. The loop looks complicated since
        // we need to handle overflow. E.g. a == 1 << 31 and b == 1, k should be 31, because b << 31 == a
        // and b << 32 > a, however b << 32 actually overflows so b << 32 == 0, this loop will never end.
        // To prevent overflow, b << k must <= 0x80000000 / 2, otherwise, b << (k + 1) will overflow.
        for (k = 0; a >= b << k && b << k <= (0x80000000 >> 1); ++k);

        // When loop ends, if a < b << k, it means no overflow happens, and k increased one additional time, we
        // need to decrease k; otherwise (a >= b << k), it means overflow happens, k is the maximum we can find.
        quotient |= 1 << (a < b << k ? --k : k);
    }

    // If negative is false, quotient must <= INT_MAX, otherwise overflow, should return INT_MAX.
    return (dividend ^ divisor) >> 31 & 0x1 ? 0 - quotient : quotient > INT_MAX ? INT_MAX : quotient;
}
int divide(const int dividend, const int divisor)
{
    unsigned quotient = 0;
    for (unsigned a = dividend < 0 ? -dividend : dividend, b = divisor < 0 ? -divisor : divisor, k, temp; a >= b; a -= temp, quotient += k)
    {
        // use a-temp >= temp to prevent overflow
        for (k = 1, temp = b; a - temp >= temp; temp <<= 1, k <<= 1);
    }

    return (dividend ^ divisor) >> 31 & 1 ? -quotient : quotient > INT_MAX ? INT_MAX : quotient;
}

// 30. Substring with Concatenation of All Words
vector<int> findSubstring(const string& s, vector<string>& words)
{
    vector<int> result;

    if (words.empty())
    {
        return result;
    }

    const unsigned patternLen = words[0].length() * words.size();

    if (patternLen > s.length())
    {
        return result;
    }

    // Here we use a hash table to store for each word, how many times it occurs.
    unordered_map<string, int> wordCount;
    for (const string& word : words)
    {
        ++wordCount[word];
    }

    for (unsigned i = 0; i <= s.length() - patternLen; ++i)
    {
        unordered_map<string, int> unusedWords(wordCount);

        // Check if all words present in current substring s[i + j..i + j + patternLen - 1]
        // Slice it to words.size() slices, for each slice, use the unusedWords
        // hash table to quick check if it is a word, if yes, decrease value in
        // hash table, means it has been used.
        for (unsigned j = 0; j < patternLen; j += words[0].length())
        {
            const string word = s.substr(i + j, words[0].length());
            if (unusedWords.find(word) == unusedWords.end())
            {
                break;
            }

            if (--unusedWords[word] == 0)
            {
                unusedWords.erase(word);
            }
        }

        if (unusedWords.empty())
        {
            result.push_back(i);
        }
    }

    return result;
}

// 31. Next Permutation
void nextPermutation(vector<int>& nums)
{
    // As we know that the greatest permutation is all numbers are in descending order.
    // So search from end to begin check if all elements are in descending order.
    unsigned i;
    for (i = nums.size() - 1; i >= 1 && nums[i] <= nums[i - 1]; --i);

    // When previous loop ends, it could be (1) i == 0, or (2) i > 0, it doesn't matter, sort this segment first.
    // Please note that starts from i, all elements are in descending order, so sorting is straightforward, just reverse them.
    for (unsigned j = i, k = nums.size() - 1; j < k; ++j, --k)
    {
        const int temp = nums[j];
        nums[j] = nums[k];
        nums[k] = temp;
    }

    // Now lets handle if we really find an i > 0, if yes, swap nums[i-1] with the smallest element that greater than nums[i-1].
    if (i > 0)
    {
        unsigned l;
        for (l = i; l < nums.size() && nums[l] <= nums[i - 1]; ++l);

        const int temp = nums[l];
        nums[l] = nums[i - 1];
        nums[i - 1] = temp;
    }
}

// 32. Longest Valid Parentheses
int longestValidParentheses(string s)
{
    if (s.length() == 0)
    {
        return 0;
    }

    int max = 0;
    vector<int> dp(s.length());

    // lets say dp[i] is the length of longest valid parentheses ends at position i. 
    // If s[0..i] has valid parentheses however it doesn't end at position i, then dp[i] = 0.
    dp[0] = 0;
    for (unsigned i = 1; i < s.length(); ++i)
    {
        // Two key thought: 
        // (1) If a substring s is a valid parentheses string, then "(" + s + ")" is also valid parentheses string.
        // (2) Two valid parentheses string's concatenation is also a valid parentheses string.
        // 
        // It is obviously that a valid parentheses string must end with a ')', We scan s from left to right, if see
        // a ')', because dp[i-1] is the length of valid parentheses string ends at position i-1, So according to (1),
        // we should check s[i-dp[i-1]-1], if it is a '(', then we know s[i-dp[i-1]-1 .. i] is also valid, thus we got:
        //          dp[i] = dp[i-1] + 2
        // According to (2), if there is another valid parentheses string before s[i-dp[i-1]-1 .. i], we also need to add 
        // dp[i-dp[i-1]-1-1] = dp[i-dp[i]] to dp[i], so we have another equation:
        //          dp[i] = dp[i] + dp[i-dp[i]]
        if (s[i] == ')' && s[i - dp[i - 1] - 1] == '(')
        {
            dp[i] = dp[i - 1] + 2;
            dp[i] += dp[i - dp[i]];
        }
        else
        {
            dp[i] = 0;
        }

        if (max < dp[i])
        {
            max = dp[i];
        }
    }

    return max;
}

// 33. Search in Rotated Sorted Array
int search(vector<int>& nums, const int target)
{
    for (int left = 0, right = nums.size() - 1; left <= right;)
    {
        const int middle = (left + right) / 2;

        if (nums[middle] < target)
        {
            // If middle is on right part, we have: nums[left] > nums[right] >= nums[middle].
            // When middle is on right part and target is greater than nums[right] (means not in nums[middle .. right]), should search in left part.
            if (nums[right] >= nums[middle] && target > nums[right])
            {
                right = middle - 1;
            }
            else // When (1) middle is on left part, (2) middle is on right part and target is in nums[middle ... right], continue search in right part.
            {
                left = middle + 1;
            }
        }
        else if (nums[middle] > target)
        {
            // If middle is on left part, we have: nums[middle] >= nums[left] > nums[right]
            // When middle is on left part and target is smaller than nums[left] (means not in nums[left .. middle]), should search in right part.
            if (nums[middle] >= nums[left] && target < nums[left])
            {
                left = middle + 1;
            }
            else // When (1) middle is on the right part, (2) middle is on left part and target is in nums[left .. middle], continue search in left part.
            {
                right = middle - 1;
            }
        }
        else //nums[mid] == target
        {
            return middle;
        }
    }

    return -1;
}
int search1_2(vector<int>& nums, const int target)
{
    for (int left = 0, right = nums.size() - 1; left <= right;)
    {
        const int mid = (left + right) / 2;

        if (nums[mid] == target)
        {
            return mid;
        }

        if (nums[left] <= nums[mid])   // mid is on left part.
        {
            if (target >= nums[left] && target < nums[mid]) // nums[left] <= target < nums[mid]
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        else if (nums[right] >= nums[mid]) // mid is on right part.
        {
            if (target > nums[mid] && target <= nums[right]) // nums[mid] < target <= nums[right]
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
    }

    return -1;
}

// 34. Search for a Range
vector<int> searchRange(vector<int>& nums, const int target)
{
    vector<int> result(2, -1);
    for (int left = 0, right = nums.size() - 1; left <= right;)
    {
        const int middle = (left + right) / 2;
        if (target == nums[middle])
        {
            // searching for the first appearance of target
            for (left = 0, right = middle; left <= right;)
            {
                const int middle2 = (left + right) / 2;
                if (target == nums[middle2])
                {
                    right = middle2 - 1;
                }
                else
                {
                    left = middle2 + 1;
                }
            }

            result[0] = left;

            // searching for the last appearance of target
            for (left = middle, right = nums.size() - 1; left <= right;)
            {
                const int middle2 = (left + right) / 2;
                if (target == nums[middle2])
                {
                    left = middle2 + 1;
                }
                else
                {
                    right = middle2 - 1;
                }
            }

            result[1] = right;

            break;
        }

        if (target < nums[middle])
        {
            right = middle - 1;
        }
        else
        {
            left = middle + 1;
        }
    }

    return result;
}

// 35. Search Insert Position
int searchInsert(vector<int>& nums, const int target)
{
    int left, right;
    for (left = 0, right = nums.size() - 1; left <= right;)
    {
        const int middle = (left + right) / 2;
        if (nums[middle] == target)
        {
            return middle;
        }

        if (nums[middle] > target)
        {
            right = middle - 1;
        }
        else
        {
            left = middle + 1;
        }
    }

    // when loop ends, left > right, left is the first number that larger than target, which is also the insert location.
    return left;
}

// 36. Valid Sudoku
bool isValidSudoku(vector<vector<char>>& board)
{
    unordered_set<string> occurrence;
    for (unsigned i = 0; i < board.size(); ++i)
    {
        for (unsigned j = 0; j < board[0].size(); ++j)
        {
            if (board[i][j] != '.')
            {
                // Instead of checking every row, every column and every block iteratively, do it in one pass scan.
                // For every element board[i][j], we generate 3 strings:
                // "r{i}{board[i][j]}" to check row i,
                // "c{j}{board[i][j]}" to check row j,
                // "{i/3}{board[i][j]}{j/3}" to check tile belonged, each tile is marked using it top left element's indice.
                if (board[i][j] < '1' || board[i][j] > '9' ||
                    !occurrence.insert(string("r") + static_cast<char>(i + '0') + board[i][j]).second ||           // row
                    !occurrence.insert(string("c") + static_cast<char>(j + '0') + board[i][j]).second ||           // column
                    !occurrence.insert(string("") + static_cast<char>(i / 3 + '0') + board[i][j] + static_cast<char>(j / 3 + '0')).second)      // tile
                {
                    return false;
                }
            }
        }
    }

    return true;
}

// 37. Sudoku Solver
bool solveSudokuUse36(vector<vector<char>>& board)
{
    for (unsigned i = 0; i < board.size(); ++i)
    {
        for (unsigned j = 0; j < board[0].size(); ++j)
        {
            if (board[i][j] == '.')
            {
                for (char candidate = '1'; candidate <= '9'; ++candidate)
                {
                    board[i][j] = candidate;
                    if (isValidSudoku(board) && solveSudokuUse36(board)) // No need to check whole board every time, just check the latest location see if it's valid.
                    {
                        return true;
                    }

                    board[i][j] = '.';
                }

                return false;
            }
        }
    }

    return true;
}

bool isValidCandidate(const vector<vector<char>>& board, const unsigned row, const unsigned col)
{
    for (unsigned r = 0; r < board.size(); ++r)
    {
        if (r != row && board[r][col] == board[row][col])
        {
            return false;
        }
    }

    for (unsigned c = 0; c < board[0].size(); ++c)
    {
        if (c != col && board[row][c] == board[row][col])
        {
            return false;
        }
    }

    for (unsigned r = 3 * (row / 3); r < 3 * (row / 3 + 1); ++r)
    {
        for (unsigned c = 3 * (col / 3); c < 3 * (col / 3 + 1); ++c)
        {
            if (r != row && c != col && board[r][c] == board[row][col])
            {
                return false;
            }
        }
    }

    return true;
}
bool dfsSearchSudoku(vector<vector<char>>& board, unsigned row, unsigned col)
{
    for (; row < board.size(); ++row, col = 0)
    {
        for (; col < board[0].size(); ++col)
        {
            if (board[row][col] == '.')
            {
                for (char candidate = '1'; candidate <= '9'; ++candidate)
                {
                    board[row][col] = candidate;
                    if (isValidCandidate(board, row, col) && dfsSearchSudoku(board, row, col + 1)) // valid
                    {
                        return true;
                    }
                }

                board[row][col] = '.';
                return false;
            }
        }
    }

    return true;
}
void solveSudoku(vector<vector<char>>& board)
{
    solveSudokuUse36(board);
}

// 38. Count and Say
string countAndSay(int n)
{
    string result = "1";
    for (; n > 1; --n)
    {
        string temp;
        for (unsigned i = 0, j; i < result.length(); i = j)
        {
            for (j = i; j < result.length() && result[j] == result[i]; ++j);
            temp += string("") + static_cast<char>(j - i + '0') + result[i];
        }

        result = temp;
    }

    return result;
}

// 39. Combination Sum
void dfsCombinationSum(vector<int>& candidates, const unsigned currentIndex, const int target, vector<int>& path, vector<vector<int>>& solutions)
{
    // In this problem, because allow choosing element multiple times, so whether checking target == 0 
    // first or checking currentIndex out of boundary first doesn't matter, won't miss valid combination.
    // Thinking about this test case: [10,1,2,7,6,5] and target == 8. There is a valid combination [1,2,5],
    // when currentIndex == 5, next recursion function call, we can still have currentIndex == 5 then we 
    // know we find a valid combination.
    if (target == 0)
    {
        solutions.push_back(path);
        return;
    }

    if (target < 0 || currentIndex >= candidates.size())
    {
        return;
    }

    // At each position we have 2 choices: choose the number on current position and keep going; or skip it.
    // Because it is allowed to choose an element multiple times so don't update the currentIndex.
    path.push_back(candidates[currentIndex]);
    dfsCombinationSum(candidates, currentIndex, target - candidates[currentIndex], path, solutions);
    path.pop_back();
    dfsCombinationSum(candidates, currentIndex + 1, target, path, solutions);
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target)
{
    vector<vector<int>> solutions;
    vector<int> path;
    dfsCombinationSum(candidates, 0, target, path, solutions);
    return solutions;
}

// 40. Combination Sum II
void dfsCombinationSum2(vector<int>& candidates, unsigned currentIndex, const int target, vector<int>& path, vector<vector<int>>& solutions)
{
    // Doesn't like problem 39, we must check target == 0 first, otherwise we could miss valid combination.
    // Still using test case: [10,1,2,7,6,5] and target == 8, when currentIndex == 5, next recursion function
    // call has currentIndex == 6 (one element can only be chosen one time), which is out of array boundary,
    // if we check boundary first, we don't have a chance to examine target == 0 so [1,2,5] will be missed.
    if (target == 0)
    {
        solutions.push_back(path);
        return;
    }

    if (target < 0 || currentIndex >= candidates.size())
    {
        return;
    }

    path.push_back(candidates[currentIndex]);
    dfsCombinationSum2(candidates, currentIndex + 1, target - candidates[currentIndex], path, solutions);
    path.pop_back();

    // Because in last call, we choose candidates[currentIndex] and may already found a solution. So if 
    // candidates[currentIndex+1] == candidates[currentIndex], need to skip, otherwise will get duplicate solutions.
    for (; currentIndex < candidates.size() - 1 && candidates[currentIndex + 1] == candidates[currentIndex]; ++currentIndex);
    dfsCombinationSum2(candidates, currentIndex + 1, target, path, solutions);
}
vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> solutions;
    vector<int> path;
    dfsCombinationSum2(candidates, 0, target, path, solutions);
    return solutions;
}

// 41. First Missing Positive
int firstMissingPositive(vector<int>& nums)
{
    // For each positive numbers nums[i] that greater than 0 and smaller than nums.size(), put it on nums[nums[i]-1].
    // For numbers not in that range, can be left on their current position.
    for (unsigned i = 0; i < nums.size();)
    {
        // Put 1 in nums[0], put 2 in num[1], etc.
        if (nums[i] > 0 && nums[i] < static_cast<int>(nums.size()) && nums[i] != nums[nums[i] - 1])
        {
            const int temp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        }
        else
        {
            ++i;
        }
    }

    // Now scan the array again, each index i, its element should be i + 1, if not, then we find the missing positive.
    for (unsigned i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != static_cast<int>(i + 1))
        {
            return i + 1;
        }
    }

    return nums.size() + 1;
}

// 42. Trapping Rain Water
// Brute force is not the best solution but it is very important to show that how to think.
// For this problem, DO NOT try to start thinking each hollow, that makes this problem looks complicated. Thinking each
// bar individually instead. Let's assume a single bar can trap some water (of course physically it is impossible), if 
// we can get how much water each bar can trap, and add them together, we get the total trapped water. So given a bar i,
// whose height is height[i], how much water it can trap? Drawing a chart helps to show that:
// water bar i trap = min(height of the highest bar on its left, height of the highest bar on its right) - height[i]
//            ___
//            | |               ___
//            | |      ___      | |
//            | |      | |      | |
//     _______|_|______|_|______|_|____________________
//                      i
// So scan height array from left to right, for each bar, scan its left highest bar and right highest bar and get the amount
// of water it can trap, add to total, after all bars are calculated, total is the answer.
int trapBruteForce(vector<int>& height)
{
    int total = 0;
    for (int i = 0; i < height.size(); ++i)
    {
        // for bar i, find its left highest and right highest bar
        // We need to include bar i itself when searching, so we know if all left/right bars are lower than bar i, in such a case
        // it means bar i cannot trap any water.
        int leftHighestIndex = 0;
        for (int j = 0; j <= i; ++j)
        {
            if (height[j] > height[leftHighestIndex])
            {
                leftHighestIndex = j;
            }
        }

        int rightHeightIndex = height.size() - 1;
        for (int j = height.size() - 1; j >= i; --j)
        {
            if (height[j] > height[rightHeightIndex])
            {
                rightHeightIndex = j;
            }
        }

        const int minHeight = height[leftHighestIndex] > height[rightHeightIndex] ? height[rightHeightIndex] : height[leftHighestIndex];
        total += minHeight - height[i];
    }

    return total;
}
// In brute force solution, for each bar we search its left highest bar and right highest bar, how much water it can trap is
// determined by the lower bar between its left highest bar and right highest bar, so time complexity is O(n^2).
// To optimize it, let's think in a reversed way. Given two bars i and j (i <= j), we first compare their heights, without
// loss of generality, let's assume lower bar is the left bar[i], we search its right side, until find a bar[k] that is higher
// than bar[i], for all bars between bar[i] and bar[k], water every bar can trap is height[i] - height[m] (i < m < k).
// Then we compare bar[k] and bar[j], choose the lower one, and repeat above process, until all bars are calculated.
//                           ___
//                           | |
//                           | |                ___
//                           | |                | |
//            ___         ___| |                | |
//            | |   ______| || |                | |
//            | |___| || || || |                | |
//            | || || || || || |                | |
//     _______|_||_||_||_||_||_|________________|_|_________
//             i              k                  j
int trapOptimized(vector<int>& height)
{
    int total = 0;
    for (int left = 0, right = height.size() - 1, probe; left <= right; )
    {
        if (height[left] < height[right]) // if left bar is lower, search its right
        {
            for (probe = left; probe <= right && height[probe] <= height[left]; ++probe)
            {
                total += height[left] - height[probe];
            }

            left = probe; // now bar[probe] is the first bar on bar[left]'s right that higher than bar[left]
        }
        else // if right bar is lower, search its left.
        {
            for (probe = right; probe >= left && height[probe] <= height[right]; --probe)
            {
                total += height[right] - height[probe];
            }

            right = probe; // now bar[probe] is the first bar on bar[right]'s left that higher than bar[right]
        }
    }

    return total;
}
int trap(vector<int>& height)
{
    return trapOptimized(height);
}

// 43. Multiply Strings
// O(n1 *n2), O(1)
string addStrings(const string& num1, const string& num2);
string multiplyUseAddStrings(string num1, string num2)
{
    if (num1 == "0" || num2 == "0")
    {
        return "0";
    }

    string product = "0";
    for (int i = num2.length() - 1; i >= 0; --i)
    {
        // calculate num1 * num2[i]
        string intermediateResult;
        char carry = '0';
        for (int j = num1.length() - 1; j >= 0; --j)
        {
            int r = (num1[j] - '0') * (num2[i] - '0') + (carry - '0');
            carry = r / 10 + '0';
            intermediateResult.insert(0, 1, static_cast<char>(r % 10 + '0'));
        }

        if (carry - '0' > 0)
        {
            intermediateResult.insert(0, 1, carry);
        }

        for (int padding = num2.length() - i - 1; padding > 0; --padding)
        {
            intermediateResult += "0";
        }

        product = addStrings(product, intermediateResult);
    }

    return product;
}

// O(n1 *n2), O(n1 + n2)
//        1      2     3
// x             5     6
//-----------------------
//         6    12    18
// + 5    10    15
//-----------------------
//   5    16    27    18
//-----------------------
//   5    16    28     8
//-----------------------
//   5    18     8     8
//-----------------------
//   6     8     8     8
string multiplyOptimized(string num1, string num2)
{
    if (num1 == "0" || num2 == "0")
    {
        return "0";
    }

    vector<int> result(num1.length() + num2.length(), 0);
    for (int i = num1.length() - 1; i >= 0; --i)
    {
        for (int j = num2.length() - 1; j >= 0; --j)
        {
            result[i + j + 1] += (num1[i] - '0') * (num2[j] - '0');
        }
    }

    for (int i = result.size() - 1; i > 0; --i)
    {
        if (result[i] >= 10)
        {
            result[i - 1] += result[i] / 10;
            result[i] = result[i] % 10;
        }
    }

    string product;
    for (int i = result[0] == 0 ? 1 : 0; i < result.size(); ++i)
    {
        product += result[i] + '0';
    }

    return product;
}
string multiply(string num1, string num2)
{
    return multiplyOptimized(num1, num2);
}

// 44. Wild card Matching
bool isMatch_Wildcard_Recurisve(string text, string pattern)
{
    if (pattern.length() == 0)
    {
        return text.length() == 0;
    }

    if (pattern[0] == '*')
    {
        int j;
        for (j = 1; j < pattern.length() && pattern[j] == '*'; ++j);

        for (int i = 0; i <= text.length(); ++i)
        {
            if (isMatch_Wildcard_Recurisve(text.substr(i), pattern.substr(j)))
            {
                return true;
            }
        }
    }
    else
    {
        if (pattern[0] == '?' || pattern[0] == text[0])
        {
            return text.length() > 0 && isMatch_Wildcard_Recurisve(text.substr(1), pattern.substr(1));
        }
    }

    return false;
}
/*
(1) Why recursive solution timeout.
Consider this case:
         0        i                       k                       m-1
    text ..........................................................
    pattern ......*.......................*...*.....*..
            0     j                       l           n-1
Let's use two * as example, assuming pattern[0 .. j] matches text[i-j .. i], now encounter a '*' at pattern[j], in recursive solution, a '*'
means branching, because a '*' can be empty, or 1 character, 2 characters, etc., so recursive algorithm needs to try all possible cases, like
this:
it first tries to match pattern[j+1 .. n-1] with text[i .. m-1],
if fails, it tries to match pattern[j+1 .. n-1] with text[i+1 .. m-1],
if fails, then tries to match pattern[j+1 .. n-1] with text[i + 2.. m-1], etc.
Let's say there is another '*' in pattern[l], recursive algorithm branches at pattern[l] again, repeating above process.
If we draw this process in tree, piece of the tree looks like this:

|- attempting to match pattern[j+1 .. n-1] with text[i .. m-1]
|---...
|------- attempting to match pattern[l+1 .. n-1] with text[k .. m-1]
|- attempting to match pattern[j+1 .. n-1] with text[i+1 .. m-1]
|---...
|------- attempting to match pattern[l+1 .. n-1] with text[k+1 .. m-1]
|- attempting to match pattern[j+1 .. n-1] with text[i+2 .. m-1]
|---...
|------- attempting to match pattern[l+1 .. n-1] with text[k+2 .. m-1]

We found that, when matching pattern[l+1 .. n-1] with text[k .. m-1], the process includes matching pattern[l+1 .. n-1] with text[k+1 .. m-1],
matching pattern[l+1 .. n-1] with text[k+2 .. m-1], etc., if pattern[l+1 .. n-1] cannot match with text[k .. m-1], rolling back to first '*' at
pattern[j] is totally useless, so we can get the conclusion: every time when mismatch happens, only need to roll back to nearest '*', if we tried
all possible branches of nearest '*' but cannot match, we can say the whole pattern won't match.
 */
bool isMatch_Wildcard_Iterative(string text, string pattern)
{
    // keep tracking index of nearest '*' in pattern and the index of its initial corresponding element in text.
    unsigned j = 0;
    for (int i = 0, matchStartPosition = -1, lastAsteriskPosition = -1; i < text.length();)
    {
        // Every time we see a '*', update indices and keep moving forward. This also solve the continuous '*' case.
        if (pattern[j] == '*')
        {
            matchStartPosition = i;
            lastAsteriskPosition = j++;
        }
        else if (pattern[j] == text[i] || pattern[j] == '?')
        {
            ++i;
            ++j;
        }
        else // mismatch
        {
            if (lastAsteriskPosition >= 0) // rollback to nearest '*' if exists.
            {
                // Here we try to match the nearest '*' to 1 character, 2 characters, so every time we shift matchStartPosition right by 1.
                // Why don't match '*' with 0 character? Since we have tried and failed otherwise we will not roll back to nearest '*'.
                // Don't worry that matchStartPosition will overflow, next iteration will catch it.
                i = ++matchStartPosition;
                j = lastAsteriskPosition + 1; // pattern always starts from next character.
            }
            else // if there is no nearest '*', means pattern cannot match text.
            {
                return false;
            }
        }
    }

    // as we know, when text is all processed, pattern may still not finish, there is only one case, pattern now only has * remaining.
    for (; j < pattern.length() && pattern[j] == '*'; ++j);
    return j == pattern.length();
}
bool isMatch_Wildcard(string text, string pattern)
{
    return isMatch_Wildcard_Iterative(text, pattern);
}

// 45. Jump Game II
// BFS would be easier to understand than greedy. Starting from first element(root), mark all unvisited elements that current
// level can reach as next level, keep finding the start and end of each level, until current level covers the last element.
int jump(vector<int>& nums)
{
    // Next level's range is nums[current level's end + 1 .. farthest location that current level elements can reach]
    // Please note, Jump Game II assumes that you can always reach the destination, here I made a modification that it is possible 
    // that you cannot reach the destination, to deal test case [3,2,1,0,4] from Jump Game I. 0 here likes a hole, when jump on it
    // you will never get out. In this case we need to do an additional check start <= end, since according to our logic, when jump
    // on the hole, in next iteration, start = end + 1 but end = farthest = end, so start > end, without that check loop will not end.
    for (unsigned start = 0, end = 0, level = 0, farthest = 0; start < nums.size() && start <= end; start = end + 1, end = farthest, ++level)
    {
        // Current level covers the last element, means we've reached the destination in current level, so level is the shortest step count.
        if (end >= nums.size() - 1)
        {
            return level;
        }

        // find the farthest reachable location that current level can reach.
        for (unsigned i = start; i <= end; ++i)
        {
            // for element nums[i], its right most reachable location is nums[i] + i
            if (nums[i] + i > farthest)
            {
                farthest = nums[i] + i;
            }
        }
    }

    return -1;
}

// 46. Permutations
// Solution 1, use recursion
vector<vector<int>> permuteRecursiveInsertion(vector<int>& nums)
{
    vector<vector<int>> results;

    if (nums.size() == 1)
    {
        results.emplace_back(nums);
        return results;
    }

    // First we calculate permutations of subarray nums[i+1, size-1], then for each permutation of nums[i+1, size-1], inserting nums[i] to its
    // every possible location, every insertion gets a new permutation, all  of those new permutations are the permutations of nums[i, size-1].
    vector<int> subarray(nums.begin() + 1, nums.end());
    for (vector<int>& permutation : permuteRecursiveInsertion(subarray))
    {
        // here we don't do insertion directly, we put nums[i] at the beginning of permutation, then keep swapping ever two elements.
        permutation.insert(permutation.begin(), nums[0]);
        results.push_back(permutation);
        for (int i = 0; i < static_cast<int>(permutation.size()) - 1; ++i)
        {
            //swap permutation[i] and permutation[i+1]
            const int exchange = permutation[i];
            permutation[i] = permutation[i + 1];
            permutation[i + 1] = exchange;

            results.push_back(permutation);
        }
    }

    return results;
}
// Solution 2, use DFS
// Given array nums[0, size-1], permutation is, every time we choose an unused element until all elements have been chosen, the
// order how we choose elements is a permutation. To reuse storage, we use a variable currentIndex to indicates the beginning of
// unused elements, its left, nums[0 .. currentIndex-1] are the elements we currently have chosen. Every time we choose an element
// in nums[currentIndex .. size-1] and exchange it with nums[currentIndex], treated this process as 'choose an unused element',
// update currentIndex to currentIndex+1, and keep repeating this process until currentIndex == size, we get one permutation.
void dfsPermuteImpl(vector<int>& nums, const unsigned currentIndex, vector<vector<int>>& results)
{
    if (currentIndex == nums.size())
    {
        results.push_back(nums);
        return;
    }

    // currentIndex split whole nums array into 2 parts, nums[0..curentIndex-1] contains the numbers we have chosen and also that range
    // forms a partial permutation. At every step (currentIndex), we have N-currentIndex choices, we can choose any one of nums[currentIndex .. N-1].
    // So here respectively choose nums[currentIndex], nums[currentIndex + 1], ... nums[N-1], exchange it with nums[currentIndex] which
    // means we choose it, then continue the process for nums[currentIndex+1 .. N-1].
    for (unsigned i = currentIndex; i < nums.size(); ++i)
    {
        swap(nums[currentIndex], nums[i]);
        dfsPermuteImpl(nums, currentIndex + 1, results);
        swap(nums[currentIndex], nums[i]);
    }
}
vector<vector<int>> permuteUseDFS(vector<int>& nums)
{
    // no need to define a vector<int> to record current path since nums itself records current path.
    vector<vector<int>> results;
    dfsPermuteImpl(nums, 0, results);
    return results;
}
vector<vector<int>> permute(vector<int>& nums)
{
    return permuteUseDFS(nums);
}

// 47. Permutations II
void dfsPermuteUniqueImpl(vector<int>& nums, unsigned currentIndex, vector<vector<int>>& results)
{
    if (currentIndex == nums.size() - 1)
    {
        results.push_back(nums);
        return;
    }

    for (unsigned i = currentIndex, j; i < nums.size(); ++i)
    {
        // As we did in problem 46, we are going to exchange nums[i] with nums[currentIndex], but we need to make sure that we 
        // haven't chosen nums[i] before, we check if there is a duplication of nums[i] within range num[currentIndex .. i-1], 
        // if true, then it means we have chosen nums[i] before so we should not choose it again.
        //
        // PLEASE NOTE that this check is for problem 47 which states nums array has duplications. For problem 46, no need to
        // have this check, just do swapping.
        for (j = currentIndex; j < i && nums[j] != nums[i]; ++j);
        if (j == i)
        {
            swap(nums[currentIndex], nums[i]);
            dfsPermuteUniqueImpl(nums, currentIndex + 1, results);
            swap(nums[currentIndex], nums[i]);
        }
    }
}
vector<vector<int>> permuteUnique(vector<int>& nums)
{
    sort(nums.begin(), nums.end());
    vector<vector<int>> results;
    dfsPermuteUniqueImpl(nums, 0, results);
    return results;
}

// 48. Rotate Image
// this solution rotate coil by coil, start from out most coil.
void clockwiseRotateByCoil(vector<vector<int>>& matrix)
{
    // Generally, for element matrix[i][j], it will be moved to matrix[j][n-i-1] after rotation, oppositely, for location
    // matrix[i][j], after rotation, the element on it will be matrix[n-j-1][i].
    for (unsigned i = 0, n = matrix[i].size(); i < matrix.size() / 2; ++i)
    {
        for (unsigned j = i; j < n - 1 - i; ++j)
        {
            const int temp = matrix[i][j];
            matrix[i][j] = matrix[n - j - 1][i];
            matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
            matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
            matrix[j][n - i - 1] = temp;
        }
    }
}
// This solution rotates array as whole, above solution is very easy to get messed in interview.
// first step, mirror on diagonal (swap matrix[i][j] with matrix[j][i]). Then if it is clockwise, 
// flip matrix's each column (swap each column matrix[*][j] with matrix[*][n-1-j])
// 1 2 3     1 4 7    7 4 1
// 4 5 6  => 2 5 8 => 8 5 2
// 7 8 9     3 6 9    9 6 3
// if it is counterclockwise, flip matrix's each row (swap each row matrix[i] with row matrix[n-1-i])
// 1 2 3     1 4 7    3 6 9
// 4 5 6  => 2 5 8 => 2 5 8
// 7 8 9     3 6 9    1 4 7
void clockwiseRotateByFlipping(vector<vector<int>>& matrix)
{
    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        for (unsigned j = i + 1; j < matrix[i].size(); ++j) // Do swap for top half only!
        {
            const int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }

    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        for (unsigned j = 0; j < matrix[i].size() / 2; ++j)
        {
            const int temp = matrix[i][j];
            matrix[i][j] = matrix[i][matrix[i].size() - 1 - j];
            matrix[i][matrix[i].size() - 1 - j] = temp;
        }
    }
}
void rotate(vector<vector<int>>& matrix)
{
    return clockwiseRotateByFlipping(matrix);
}

// 49. Group Anagrams
vector<vector<string>> groupAnagrams(vector<string>& strs)
{
    unordered_map<string, vector<string>> charset2Strings;
    for (const string& str : strs)
    {
        string key = str;
        sort(key.begin(), key.end());
        charset2Strings[key].push_back(str);
    }

    vector<vector<string>> results(charset2Strings.size());
    transform(charset2Strings.begin(), charset2Strings.end(), results.begin(), [](const auto& pair) { return pair.second; });
    return results;
}

// 50. Pow(x, n)
double myPowRecusive(const double x, const int n)
{
    if (n == 0)
    {
        return 1;
    }

    double result = myPowRecusive(x, n / 2);
    result *= result; // don't recursively calculate pow(x, n-n/2), that will exceed time limit.
    if (n % 2)        // still need to do multiplication 1 more time.
    {
        result *= n > 0 ? x : 1 / x;
    }

    return result;
}
double myPowIterativeLogN(double x, int n)
{
    double result = 1.0;
    for (x = n > 0 ? x : 1 / x; n != 0; n /= 2, x *= x)
    {
        if (n & 1)
        {
            result *= x;
        }
    }

    return result;
}
double myPow(double x, int n)
{
    double result = 1.0;
    for (unsigned i = 0, absn = abs(n); absn > 0 && i <= 31; ++i, result *= absn & 1 ? x : 1, absn >>= 1, x *= x);
    return n < 0 ? 1 / result : result;
}

// 51. N-Queens
void dpsSearchNQueen(vector<vector<unsigned>>& results, vector<unsigned>& path, unsigned currentRow)
{
    if (currentRow == path.size())
    {
        results.push_back(path);
        return;
    }

    // Try every possible column in current row. Note: for path[i], it means a queue is put on [i, path[i]].
    for (unsigned j = 0; j < path.size(); ++j)
    {
        // check all previous rows, see if current location [currentRow, j] is allowed.
        bool invalid = false;
        for (unsigned i = 0; i < currentRow && !invalid; ++i)
        {
            invalid = path[i] == j || i + path[i] == currentRow + j || static_cast<int>(i - path[i]) == static_cast<int>(currentRow - j);
        }

        if (!invalid)
        {
            path[currentRow] = j;
            dpsSearchNQueen(results, path, currentRow + 1);
        }
    }
}
vector<vector<string>> solveNQueens(const int n)
{
    vector<vector<unsigned>> solutions;
    vector<unsigned> path(n, 0);
    dpsSearchNQueen(solutions, path, 0);

    vector<vector<string>> results;
    for (const vector<unsigned>& solution : solutions)
    {
        vector<string> matrix;
        for (unsigned column : solution)
        {
            string row(n, '.');
            row[column] = 'Q';
            matrix.push_back(row);
        }

        results.push_back(matrix);
    }

    return results;
}

// 52. N-Queens II
void dpsCountNQueen(unsigned& solutionCount, vector<unsigned>& path, unsigned currentRow)
{
    if (currentRow == path.size())
    {
        ++solutionCount;
        return;
    }

    // Try every possible column in current row. Note: for path[i], it means a queue is put on [i, path[i]].
    for (unsigned j = 0; j < path.size(); ++j)
    {
        // check all previous rows, see if current location [currentRow, j] is allowed.
        bool invalid = false;
        for (unsigned i = 0; i < currentRow && !invalid; ++i)
        {
            invalid = path[i] == j || i + path[i] == currentRow + j || (int)(i - path[i]) == (int)(currentRow - j);
        }

        if (!invalid)
        {
            path[currentRow] = j;
            dpsCountNQueen(solutionCount, path, currentRow + 1);
        }
    }
}
int totalNQueens(int n)
{
    unsigned solutionCount = 0;
    vector<unsigned> path(n, 0);
    dpsCountNQueen(solutionCount, path, 0);
    return solutionCount;
}

// 53. Maximum Subarray
// Some thoughts:
// This is a DP problem, at first I want to define dp[i] as: the maximum sum of subarray in range nums[0..i], 
// so when i increases from 0 to n-1 (n is the length of nums array), we got the final answer at dp[n-1]. However
// this doesn't work. The reason is given an i, we only know the max subarray's sum in the range nums[0..i-1],
// but we don't know where the previous max subarray ends, it could end at nums[i-1], or before nums[i-1], this
// information is very important because we need to decide whether nums[i] is possible to join previous subarray,
// or become a start of new subarray. It is very like to the question 32 (longest valid parentheses), which
// inspires me to change the definition of dp[i] to:
//      Given range nums[0..i], for all subarraies end at num[i], the maximum sum of those subarraies.
// With this new definition, given a dp[i], we know that the max subarray
// ends at nums[i], and the final answer is the maximum number in dp[0..n-1]. Why? 
// Back to the original question, given an array nums[0..n-1], its max subarray must ends at a certain element
// in nums, given that dp[0..n-1] stores the max subarray's sum ends at nums[0], nums[1], ... nums[n-1], so the 
// max subarray's sum is the maximum element in dp[0..n-1].
// Then next step is how to get the transition function. Let's starts from array has 3 element : 1, -2, 3
// i = 0, max sum is 1, no doubt, dp[0] = 1.
// i = 1, we are facing two choices, whether -2 should join previous max subarray or start a new subarray. If 
//        joining previous subarray, then the subarray becomes [1, -2] and sum is 1-2 = -1; if we decide to start
//        a new subarray, the new subarray's sum is -2, -1 > -2, so -2 should join, and dp[1] = -1.
// i = 2, because dp[1] = -1, if 3 join previous subarray, new sum is 3-1=2, if we start a new subarray from 3,
//        the sum of new subarray is 3, 3>2 so we should start a new subarray.
// From this we can see, actually it doesn't matter nums[i] is positive or negative, as long as dp[i-1] is positive,
// nums[i] should always join previous subarray other than start a new subarray.
// Why? If nums[i] is positive, obviously joining the previous subarray can make its sum greater;
// if nums[i] is negative, joining previous subarray is better than starting a new subarray from it if dp[i-1] is
// positive. But if dp[i-1] is negative, we should choose to start a new subarray.
// So we get the state transition function: 
//          dp[i] = dp[i-1] > 0 ? dp[i-1] + nums[i] : nums[i]
// And since every dp[i] only depends on dp[i-1], so we can actually use 1 additional variable to store dp[i-1], 
// reducing space complexity from O(n) to O(1).
int maxSubArrayDP(vector<int>& nums)
{
    int dp = nums[0];
    int maxSum = dp;
    for (unsigned i = 1; i < nums.size(); ++i)
    {
        dp = dp > 0 ? dp + nums[i] : nums[i];

        if (dp > maxSum)
        {
            maxSum = dp;
        }
    }

    return maxSum;
}
// This is another way of thinking this problem. Let's start from nums[0], so for each num[i], we have two choices,
// adding it to current subarray (sum = sum + nums[i]), or starting a new subarray (sum = nums[i]), we go the way 
// can make sum greater by comparing this two sums.
int maxSubArrayStraightforward(vector<int>& nums)
{
    int sum = 0; // sum will be the maximum sum of subarries in nums[0..i] and ending at nums[i].
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
int maxSubArray(vector<int>& nums)
{
    return maxSubArrayStraightforward(nums);
}

// 54. Spiral Matrix
vector<int> spiralOrder(vector<vector<int>>& matrix)
{
    vector<int> result;
    if (!matrix.empty())
    {
        for (int rowStart = 0, colStart = 0, rowEnd = matrix.size() - 1, colEnd = matrix[0].size() - 1;
            rowStart <= rowEnd && colStart <= colEnd;
            ++rowStart, ++colStart, --colEnd, --rowEnd)
        {
            for (int j = colStart; j <= colEnd; ++j)
            {
                result.push_back(matrix[rowStart][j]);
            }

            for (int i = rowStart + 1; i <= rowEnd; ++i)
            {
                result.push_back(matrix[i][colEnd]);
            }

            if (rowStart < rowEnd && colStart < colEnd)
            {
                for (int j = colEnd - 1; j > colStart; --j)
                {
                    result.push_back(matrix[rowEnd][j]);
                }

                for (int i = rowEnd; i > rowStart; --i)
                {
                    result.push_back(matrix[i][colStart]);
                }
            }
        }
    }

    return result;
}

// 55. Jump Game
bool canJumpUse45(vector<int>& nums)
{
    return jump(nums) != -1;
}
// Starting from nums[0], keep updating farthest which is the left most position we can reach so far.
// When loop ends, check whether farthest beyond the last elements of array.
bool canJump(vector<int>& nums)
{
    unsigned farthest = 0;
    for (unsigned i = 0; i <= farthest && farthest < nums.size(); ++i)
    {
        if (nums[i] + i > farthest)
        {
            farthest = nums[i] + i;
        }

        if (farthest >= nums.size() - 1)
        {
            return true;
        }
    }

    return farthest >= nums.size() - 1;
}

// 56. Merge Intervals
vector<Interval> mergeUseSortAndNewVector(vector<Interval>& intervals)
{
    vector<Interval> result;

    if (!intervals.empty())
    {
        std::sort(intervals.begin(), intervals.end(), [](const Interval a, const Interval b) { return a.start < b.start; });
        result.push_back(intervals[0]);
        for (unsigned i = 1; i < intervals.size(); ++i)
        {
            if (result.back().end >= intervals[i].start)
            {
                result.back().end = max(result.back().end, intervals[i].end);
            }
            else
            {
                result.push_back(intervals[i]);
            }
        }
    }

    return result;
}
vector<Interval> mergeInPlace(vector<Interval>& intervals)
{
    int last = intervals.size() - 1;

    // intervals[0 .. unmerged-1] are sorted and merged intervals, intervals[unmerged .. last] are unmerged intervals.
    for (int unmerged = 1, i, j; unmerged <= last; ++unmerged)
    {
        const Interval current = intervals[unmerged];
        for (i = 0, j = unmerged - 1; i <= j;)
        {
            const int mid = (i + j) / 2;
            if (intervals[mid].start < current.start)
            {
                i = mid + 1;
            }
            else
            {
                j = mid - 1;
            }
        }

        // When above loop ends, i > j, and i is the insert location.
        // Check if current interval overlaps with its left element interval[i-1] first, merge current interval
        // with intervals[i-1] instead of inserting it.
        if (i > 0 && intervals[i - 1].end >= current.start)
        {
            intervals[i - 1].end = max(intervals[i - 1].end, current.end);
            --i; // Update i so we treat merged intervals[i-1] as the newly inserted interval, later we merge it with its right intervals.

            // Erase intervals[unmerged] by move intervals[last] to it, and shrink the last.
            // Please note that in current iteration we need to maintain that unmerged is the end of current merged segment.
            intervals[unmerged--] = intervals[last--];
        }
        else
        {
            for (j = unmerged - 1; j >= i; --j) // shift intervals[i .. unmerged-1] right by 1
            {
                intervals[j + 1] = intervals[j];
            }

            intervals[i] = current;
        }

        // check if we need to merge intervals[i] with its right intervals.
        for (j = i + 1; j <= unmerged && intervals[i].end >= intervals[j].start; ++j)
        {
            intervals[i].end = max(intervals[i].end, intervals[j].end);
        }

        // now erase intervals[i+1 .. j-1] by shifting intervals[j .. last] left.
        for (int k = i + 1, p = j; p <= last; ++k, ++p)
        {
            intervals[k] = intervals[p];
        }

        // last step is updating unmerged and last.
        const int offset = j - i - 1;
        unmerged -= offset;
        last -= offset;
    }

    // finally only keep intervals[0 .. last] and return it.
    intervals.resize(last + 1);

    return intervals;
}
vector<Interval> merge(vector<Interval>& intervals)
{
    return mergeUseSortAndNewVector(intervals);
}

// 57. Insert Interval
vector<Interval> insertUseBinarySearch(vector<Interval>& intervals, const Interval newInterval)
{
    // First find the insert location based on start
    int i = 0;
    int j = intervals.size() - 1;
    while (i <= j)
    {
        const int mid = (i + j) / 2;
        if (intervals[mid].start < newInterval.start)
        {
            i = mid + 1;
        }
        else
        {
            j = mid - 1;
        }
    }

    // When above loop ends, i > j, and i is the insert location.
    // Check if current interval overlaps with its left element interval[i-1] first, merge current interval
    // with intervals[i-1] instead of inserting.
    if (i > 0 && intervals[i - 1].end >= newInterval.start)
    {
        intervals[i - 1].end = max(intervals[i - 1].end, newInterval.end);
        --i; // Update i so we treat merged intervals[i-1] as the newly inserted interval, later we merge it with its right intervals.
    }
    else // insert newInterval to i
    {
        intervals.resize(intervals.size() + 1);
        for (j = intervals.size() - 1; j >= i; --j) // shift intervals[i .. unmerged-1] right by 1
        {
            intervals[j + 1] = intervals[j];
        }

        intervals[i] = newInterval;
    }

    // check if we need to merge intervals[i] with its right intervals.
    for (j = i + 1; j < intervals.size() && intervals[i].end >= intervals[j].start; ++j)
    {
        intervals[i].end = max(intervals[i].end, intervals[j].end);
    }

    // now erase intervals[i+1 .. j-1] by shifting intervals[j .. last] left.
    for (int k = i + 1, p = j; p < intervals.size(); ++k, ++p)
    {
        intervals[k] = intervals[p];
    }

    // finally only keep intervals[0 .. last] and return it.
    intervals.resize(intervals.size() - j + i + 1);

    return intervals;
}
vector<Interval> insertUseSTL(vector<Interval>& intervals, const Interval newInterval)
{
    // Given two intervals a and b, define a < b when a is on the left of b and there is no overlap between a and b. Since
    // initially intervals array is sorted and no overlaps between any two elements, means for any i and j, i < j implies 
    // intervals[i] < intervals[j].
    // The logic of equal_range is given an object x and a sorted object array, it returns 2 iterators, *(first-1) is the
    // first element that smaller than x, *last is the first element that x smaller than (*last), given the way how define
    // order of intervals, elements in range intervals[first .. last-1] all have overlap with newInterval.
    const pair<vector<Interval>::iterator, vector<Interval>::iterator> range = equal_range(intervals.begin(), intervals.end(), newInterval, [](const Interval& a, const Interval& b) { return a.end < b.start; });
    const vector<Interval>::iterator first = range.first;
    vector<Interval>::iterator last = range.second;
    if (first == last) // no elements in array has overlap with newInterval so just insert it.
    {
        intervals.insert(first, newInterval);
    }
    else
    {
        // intervals[last-1] is the last element has overlap with newInterval.
        --last;
        last->start = min(newInterval.start, first->start);
        last->end = max(newInterval.end, last->end);
        intervals.erase(first, last); // note that last will not be erased.
    }

    return intervals;
}
vector<Interval> insert(vector<Interval>& intervals, const Interval newInterval)
{
    return insertUseSTL(intervals, newInterval);
}

// 58. Length of Last Word
int lengthOfLastWord(string s)
{
    int i = s.length() - 1;
    for (; i >= 0 && s[i] == ' '; --i); // skip trailing spaces

    // no need to check if i < 0 before start next loop, if i == -1, j = i so j == -1, next loop won't execute thus j - i == 0
    int j = i;
    for (; j >= 0 && s[j] != ' '; --j);
    return i - j;
}

// 59. Spiral Matrix II
vector<vector<int>> generateMatrix(const int n)
{
    vector<vector<int>> matrix(n, vector<int>(n));
    for (int start = 0, end = n - 1, number = 1, i, j; start <= end; ++start, --end)
    {
        // pre initialize the matrix[start][start] since when start == end, following loops won't execute.
        i = start;
        j = start;
        matrix[i][j] = number;

        for (; j < end; ++j) // i is row no., j is col no., when ends, i == start and j == end.
        {
            matrix[i][j] = number++;
        }

        for (; i < end; ++i) // when ends, i == end and j == end
        {
            matrix[i][j] = number++;
        }

        for (; j > start; --j) // when ends, i == end and j == start
        {
            matrix[i][j] = number++;
        }

        for (; i > start; --i) // when ends, i == start and j == start
        {
            matrix[i][j] = number++;
        }
    }

    return matrix;
}

// 60. Permutation Sequence
// We can keep calling nextPermutation() k-1 times to get the answer but that is not the best solution. Thinking in 
// this way, assume n = 4, choosing 1, 2, 3, 4 respectively as first number, we get 4 groups: {1, x, x, x}, 
// {2, x, x, x}, {3, x, x, x} and {4, x, x, x}. First it is obvious that each group has 3! = 6 elements; second, all
// elements of group {1, x, x, x} are smaller than any element of other 3 groups, accordingly, all elements in group
// {2, x, x, x} are smaller than any elements of group {3, x, x, x} and {4, x, x, x}. So, we can divided all 4! = 24 
// elements into 4 groups: 
//              {1, x, x, x} | {2, x, x, x} | {3, x, x, x} | {4, x, x, x}
//                 0 - 5     |    6 - 11    |    12 - 18   |    19 - 23
//.Let's say k = 8, first decrease k by 1 since index starts from 0, the k-th element cannot from first group since it 
// only has 6 elements, it falls into second group {2, x, x, x}'s range, here we get 2 things: (a)the kth permutation's
// first number is 2; (b)k % 6 = 1, the kth permutation is the 2nd element inside second group {2, x, x, x}.
// Now we only have 3 candidates: 1, 3, 4, likewise there are 3 groups: {1, x, x}, {3, x, x} and {4, x, x}, each group
// has 2! = 2 elements, since now k = 1 so we know k is from first group, and the k-th permutation's second number is 1,
// new k = 1 % 2 = 1.
// Now remaining candidates are 3 and 4 and k = 1, there are only 2 groups, {3, 4} and {4, 3}, each group has 1 element,
// so the kth permutation is in second group {4, 3}, the kth permutation's third number is 4, and new k = 1 % 1 = 0.
// Now we only have 1 number remaining, which is 3, so there is no need to do grouping, just return this only number.
// Finally we get all 4 numbers of kth permutation, it is {2, 1, 4, 3}.
string getPermutation(int n, int k)
{
    int factorial = 1;
    vector<int> numbers(n);
    for (int i = 1; i <= n; ++i)
    {
        factorial *= i;
        numbers[i - 1] = i;
    }

    string result;

    if (k-- <= factorial) // Decreasing k by 1 because index starts from 0, this can make counting easier.
    {
        // n is count of unused numbers in numbers array.
        for (int groupSize = factorial / n; n > 1; k %= groupSize, groupSize /= --n) // groupSize == (n-1)!
        {
            // current choose number is numbers[k/groupSize].
            result += numbers[k / groupSize] + '0';

            // then erase number[k/groupSize] by shifting its right elements to left by 1, and decrease n by 1.
            for (int i = k / groupSize; i < n - 1; ++i)
            {
                numbers[i] = numbers[i + 1];
            }
        }

        // when n == 1, no grouping.
        result += numbers[0] + '0';
    }

    return result;
}

// 61. Rotate List
ListNode* rotateRightUseFastSlowPointer(ListNode* head, int k)
{
    ListNode* fast = head;
    int length = 0;
    for (; fast != nullptr; ++length, fast = fast->next);
    if (length > 0 && k % length > 0)
    {
        for (k %= length, fast = head; k > 0 && fast != nullptr; --k, fast = fast->next);

        ListNode* slow = head;
        for (; fast->next != nullptr; fast = fast->next, slow = slow->next);

        fast->next = head;
        head = slow->next;
        slow->next = nullptr;
    }

    return head;
}
ListNode* rotateRightUseSinglePointer(ListNode* head, int k)
{
    if (head != nullptr)
    {
        ListNode* p = head;
        int length = 1;
        for (; p->next != nullptr; ++length, p = p->next);
        p->next = head; // circle the list.
        for (k = length - k % length; k > 0; --k, p = p->next);
        head = p->next;
        p->next = nullptr;
    }

    return head;
}
ListNode* rotateRight(ListNode* head, int k)
{
    return rotateRightUseSinglePointer(head, k);
}

// 62. Unique Paths
// This can use DFS but it will get time exceed and DFS will actually get every route. DP is the correct way.
int uniquePaths2D(const int m, const int n) // n is row count, m is column count.
{
    // Given a cell [i, j], robot can only get to it from above cell [i-1,j] or left cell [i, j-1]. So 
    // we define dp[i,j] as the count of unique paths from [0,0] => [i,j], so we can get this equation:
    // dp[i][j] = dp[i-1][j] + dp[i][j-1]
    vector<vector<int>> dp(n, vector<int>(m, 0));

    // initialization, it actually says if the maze is 1 dimension array, then there is only 1 unique path to get to the destination.
    for (int i = 0; i < m; ++i)
    {
        dp[0][i] = 1;
    }

    for (int i = 0; i < n; ++i)
    {
        dp[i][0] = 1;
    }

    for (int i = 1; i < n; ++i)
    {
        for (int j = 1; j < m; ++j)
        {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    return dp[n - 1][m - 1];
}
// The above solution uses a 2 dimensions array so space complexity is O(n^2). Let's got back to the equation, we can find 
// that to get value of dp[i,j], it actually only needs its left element's value and above element's value.
// So do we need two one dimension arrays? No, actually only need one one dimension array. Initially initialize it to all 1, 
// then starts from dp[1], adding dp[i] with its left element dp[i-1] and store back to dp[i] (note before adding, dp[i] stores 
// the above element's value of new dp[i]), keep doing this n-1 times, last element in dp is the answer.
int uniquePaths(int m, int n) // n is row count, m is column count.
{
    // So this doesn't affect the correctness of solution but is to save more space. Considering a 
    // matrix that m > n, if we swap m and n, which is actually roate matrix by 90 degree, the count
    // of unique paths from top left to right bottom won't change.
    if (m > n)
    {
        swap(m, n);
    }

    vector<int> dp(m, 1);
    for (int i = 1; i < n; ++i)
    {
        for (int j = 1; j < m; ++j)
        {
            dp[j] += dp[j - 1];
        }
    }

    return dp[m - 1];
}

// 63. Unique Paths II
// the algorithm is similar to problem 62, however the dp calculation rule needs some updates. In 62, we keep adding
// dp[j] and dp[j-1] to get new dp[j], here we need to first check if grid[i][j] is obstacle or not. If yes, set dp[j]
// to 0 (remember the definition of dp[i][j] is the count of unique paths from grid[0][0] to grid[i][j], so set dp[i][j]
// to 0 means the obstacle cell is unreachable) instead of adding dp[j] to dp[j-1].
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid)
{
    vector<int> dp(obstacleGrid[0].size(), 0);

    // initialize dp[] to 1 until encounter an obstacle, then set dp[] value to 0 (means unreachable) after obstacle.
    for (unsigned i = 0; i < obstacleGrid[0].size() && obstacleGrid[0][i] == 0; ++i)
    {
        dp[i] = 1;
    }

    for (unsigned i = 1; i < obstacleGrid.size(); ++i)
    {
        // first element could be obstacle.
        if (obstacleGrid[i][0] == 1)
        {
            dp[0] = 0;
        }

        for (unsigned j = 1; j < obstacleGrid[i].size(); ++j)
        {
            dp[j] = obstacleGrid[i][j] == 0 ? dp[j] + dp[j - 1] : 0;
        }
    }

    return dp.back();
}

// 64. Minimum Path Sum
// Note that it asks for the smallest path sum, not the path itself, if we use DFS, we do unnecessary calculation which will 
// get time exceeded, so it is still a DP problem. Thinking in this way, given a cell grid[i][j] (i >= 1 and j >= 1) , it can
// only be entered either from top (grid[i-1][j]) or left (grid[i][j-1]), so the minimum path from grid[0][0] to grid[i][j] is
// min (min path from grid[0][0] to grid[i-1][j], min path from grid[0][0] to grid[i][j-1]). Define dp[i][j] as the minimum path
// sum that from grid[0][0] to grid[i][j], after all cells in grid have been iterated, last element of dp is the answer.
int minPathSum(vector<vector<int>>& grid)
{
    vector<vector<int>> dp(grid.size(), vector<int>(grid[0].size(), 0));

    dp[0][0] = grid[0][0];
    for (unsigned i = 1; i < dp[0].size(); ++i)
    {
        dp[0][i] = grid[0][i] + dp[0][i - 1];
    }

    for (unsigned i = 1; i < dp.size(); ++i)
    {
        dp[i][0] = grid[i][0] + dp[i - 1][0];
    }

    for (unsigned i = 1; i < dp.size(); ++i)
    {
        for (unsigned j = 1; j < dp[i].size(); ++j)
        {
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    return dp.back().back();
}

// 65. Valid Number
// Valid input character: sign(+/-), dot(.) digit(0-9), e/E, whitespace, others are invalid characters.
// Manually build a state transition table:
// +-------------------------------------+-----+-----+-----+----+------------+-------+
// |         State(Example)\Input        | 0-9 | e/E | +/- |  . | whitespace | other |
// +-------------------------------------+-----+-----+-----+----+------------+-------+
// |  0 |                                |  1  |  -1 |  2  |  3 |      0     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  1 |            0, 1, 123           |  1  |  4  |  -1 |  5 |      6     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  2 |              +, -              |  1  |  -1 |  -1 |  3 |     -1     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  3 |                .               |  5  |  -1 |  -1 | -1 |     -1     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  4 |          0e, 1e, 123e          |  7  |  -1 |  8  | -1 |     -1     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  5 |          0., 1., 123.          |  5  |  4  |  -1 | -1 |      6     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  6 | 0{space}, 1{space}, 123{space} |  -1 |  -1 |  -1 | -1 |      6     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  7 |        0e0, 1e0, 123e456       |  7  |  -1 |  -1 | -1 |      6     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
// |  8 |         0e+, 1e-, 123e+        |  7  |  -1 |  -1 | -1 |     -1     |   -1  |
// +----+--------------------------------+-----+-----+-----+----+------------+-------+
bool isNumber(string s)
{
    enum InputType { digit = 0, eE, sign, dot, whitespace, other };
    vector<vector<int>> transitionTable({
        { 1, -1, 2, 3, 0, -1},
        { 1, 4, -1, 5, 6, -1 },
        { 1, -1, -1, 3, -1, -1 },
        { 5, -1, -1, -1, -1, -1 },
        { 7, -1, 8, -1, -1, -1 },
        { 5, 4, -1, -1, 6, -1},
        { -1, -1, -1, -1, 6, -1 },
        { 7, -1, -1, -1, 6, -1 },
        { 7, -1, -1, -1, -1, -1 }
        });

    int state = 0; // start state
    for (char c : s)
    {
        InputType input;

        if (isdigit(c))
        {
            input = digit;
        }
        else if (c == 'e' || c == 'E')
        {
            input = eE;
        }
        else if (c == '+' || c == '-')
        {
            input = sign;
        }
        else if (c == '.')
        {
            input = dot;
        }
        else if (c == ' ')
        {
            input = whitespace;
        }
        else
        {
            input = other;
        }

        state = transitionTable[state][input];
        if (state == -1)
        {
            return false;
        }
    }

    // back to transition table, only state 1, 5, 6, 7 means a completely identified number, other states are not.
    return state == 1 || state == 5 || state == 6 || state == 7;
}

// 66. Plus One
vector<int> plusOne(vector<int>& digits)
{
    unsigned carry = 1;
    for (int i = digits.size() - 1; i >= 0 && carry > 0; --i)
    {
        const int val = digits[i] + carry;
        digits[i] = val % 10;
        carry = val / 10;
    }

    if (carry > 0)
    {
        digits.insert(digits.begin(), carry);
    }

    return digits;
}

// 67. Add Binary
string addBinary(string a, string b)
{
    string result;
    unsigned char carry = 0;
    for (string::const_reverse_iterator p = a.rbegin(), q = b.rbegin(); p != a.rend() || q != b.rend() || carry != 0; )
    {
        const unsigned char val = (p == a.rend() ? 0 : *p++ - '0') + (q == b.rend() ? 0 : *q++ - '0') + carry;
        result.insert(0, 1, (val & 1) + '0');
        carry = val >> 1;
    }

    return result;
}
string addBinary2(string a, string b)
{
    string result;
    unsigned char carry = 0;
    for (int i = a.size() - 1, j = b.size() - 1, val; i >= 0 || j >= 0 || carry != 0; carry = val >> 1)
    {
        val = (i >= 0 ? a[i--] - '0' : 0) + (j >= 0 ? b[j--] - '0' : 0) + carry;
        result.insert(0, 1, (val & 1) + '0');
    }

    return result;
}

// 68. Text Justification
vector<string> fullJustify(const vector<string>& words, const int maxWidth)
{
    vector<string> lines;

    for (unsigned i = 0, j; i < words.size(); i = j)
    {
        // Step 1: find the start and end index of current line. Starting from word[i], add as many words as possible to current line,
        // when loop ends, word[j] is the first word of next line.
        int wordsLength = 0; // wordsLength doesn't count whitespace, only total length of all words in current line.
        for (j = i; j < words.size() && wordsLength + words[j].size() + (j - i) <= maxWidth; wordsLength += words[j++].size());

        // Step 2: Now justify current line, which includes words[i..j), with a total of (maxWidth-wordsLength) whitespace and j-i-1 padding
        // slots. There are 2 corner cases:
        // If current line is the last line, every two words has 1 whitespace (note all unused whitespace are appended at the end of line).
        // If current line is not the last line, we need to consider when (maxWidth-wordsLength) is not divisible by (j-i-1), according to
        // the problem, the empty slots on the left will be assigned more spaces than the slots on the right. For example, let's assume that
        // there are 27 whitespace and 6 slots, 27/6=4..3, every slot has 4 whitespace initially, for the 3 unused whitespace, assign them
        // to all slots from left to right, so the correct padding sizes are [4+1,4+1,4+1,4,4,4]=[5,5,5,4,4,4].
        vector<int> paddings(j - i - 1, 1);
        if (j < words.size() && j - i - 1 > 0) // Not last line, and current line contains more than 1 word.
        {
            const int avg = (maxWidth - wordsLength) / (j - i - 1);
            int reminder = (maxWidth - wordsLength) % (j - i - 1);
            for (int& padding : paddings)
            {
                padding = avg;
                if (reminder > 0)
                {
                    padding++;
                    reminder--;
                }
            }
        }

        // Step 3: Concatenate all words using paddings[] calculated in step 2. We start from word[i+1] so we can handle the case that 
        // current line only has 1 word.
        string line = words[i];
        for (unsigned k = i + 1; k < j; ++k)
        {
            line.append(paddings[k - i - 1], ' ').append(words[k]);
        }

        // Step 4: Check if current line length == maxWidth, if not it means there are unused whitespace, append all unused
        // whitespace to the end of line, it happens when:
        // (1) We are justifying the last line.
        // (2) We are not justifying the last line but current line only contains one word (thus above loop are skipped).
        line.append(maxWidth - line.size(), ' ');
        lines.push_back(line);
    }

    return lines;
}

// 69. Sqrt(x)
int mySqrtBinarySearch(int x)
{
    for (unsigned left = 1, right = x; left <= right; )
    {
        const unsigned mid = (left + right) / 2;
        if (mid > x / mid)
        {
            right = mid - 1;
        }
        else if (mid < x / mid)
        {
            if ((mid + 1) > x / (mid + 1))
            {
                return mid;
            }

            left = mid + 1;
        }
        else
        {
            return mid;
        }
    }

    return 0;
}
int mySqrtNewton(int x)
{
    unsigned r = x;
    for (; r > 0 && x > 0 && r > x / r; r = (r + x / r) / 2);
    return r;
}
int mySqrt(const int x)
{
    // No matter binary search or Newton iterative, they all start from an initial value that is greater than
    // sqrt(x), then keep iterating to get more and more close to answer. So choosing a initial value that as
    // much close to answer as possible can improve performance. The easiest is starting from x, since always
    // have 0 <= sqrt(x) <= x (x >= 0), Can we easily get an initial value smaller than x?
    // Let's assume (x/2^k) > sqrt(x), so we can get x > 2^(2k), it means, if we know x is greater than a power
    // of 2, say 2^n, we can start from x / 2^(n/2), which is proved to be greater than sqrt(x).
    //
    // To find the maximum 2^n < x, we search for the most signicant bit of 1, and count of 1s. The reason is
    // if there is only one 1, it means x is a power of 2, e.g. 8 = 2^3, so the maximum power of 2 that smaller
    // than x, is 2^(3-1) = 2^2 = 4.
    int msb = -1;
    int countOfOne = 0;
    for (unsigned i = 0, y = x; i < 31 && y > 0; ++i, y >>= 1)
    {
        if (y & 1)
        {
            msb = i;
            ++countOfOne;
        }
    }

    unsigned startValue = x;
    for (unsigned half = (countOfOne == 1 ? msb - 1 : msb) / 2; half > 0; startValue >>= 1, --half);

    // we can pass the startValue into sub functions or just paste the code into sub functions to get startValue.
    return mySqrtNewton(x);
}

// 70. Climbing Stairs
// Let dp[n] is the different ways to reach stair n, then dp[n] = dp[n-1] + dp[n-2]. Since for each dp[i] only needs
// dp[i-1] and dp[i-2] so no need to allocate an array.
int climbStairs(int n)
{
    int dp0 = 1;
    int dp = 1;
    for (; n > 1; --n) // dp[1] == 1, we have manually set it, so iteration ends when n == 1.
    {
        const int next = dp0 + dp;
        dp0 = dp;
        dp = next;
    }

    return dp;
}

// 71. Simplify Path
string simplifyPathWithStack(string& path)
{
    vector<string> stack; // slash, no matter it is root or not, will not be pushed to stack. Note std::stack doesn't support iteration.
    for (unsigned i = 0, j; i < path.length(); i = j)
    {
        if (path[i] == '/')
        {
            for (j = i; j < path.length() && path[j] == '/'; ++j);  // consecutive slashes are treated as one slash.
        }
        else
        {
            for (j = i; j < path.length() && path[j] != '/'; ++j);
            const string fraction = path.substr(i, j - i);
            if (fraction == "..")
            {
                if (!stack.empty())
                {
                    stack.pop_back();
                }
            }
            else if (fraction != ".")
            {
                stack.push_back(fraction);
            }
        }
    }

    string result;
    for (const string fraction : stack)
    {
        result.append("/").append(fraction);
    }

    return result.empty() ? "/" : result;
}
string simplifyPathWithoutStack(string& path)
{
    string result;
    for (int i = 0, j, k; i < path.length(); i = j)
    {
        if (path[i] == '/')
        {
            for (j = i; j < path.length() && path[j] == '/'; ++j);
        }
        else
        {
            for (j = i; j < path.length() && path[j] != '/'; ++j);
            const string fraction = path.substr(i, j - i);
            if (fraction == "..")
            {
                for (k = result.length() - 1; k >= 0 && result[k] != '/'; --k);
                if (k >= 0)
                {
                    result.resize(k);
                }
            }
            else if (fraction != ".")
            {
                result.append("/").append(fraction);
            }
        }
    }

    return result.empty() ? "/" : result;
}
string simplifyPath(string& path)
{
    // Note that Unix like system accepts >= 3 dots as folder name while Windows doesn't.
    return simplifyPathWithoutStack(path);
}

// 72. Edit Distance
int minDistanceRecursive(string word1, string word2)
{
    if (word1.empty())
    {
        return word2.length();
    }

    if (word2.empty())
    {
        return word1.length();
    }

    if (word1.front() == word2.front())
    {
        return minDistanceRecursive(word1.substr(1), word2.substr(1));
    }

    // delete first character from word1, word1 moves forward by 1 character
    const int d = 1 + minDistanceRecursive(word1.substr(1), word2);

    // replace word1's first character with word2's first character, both move forward by 1 character
    const int r = 1 + minDistanceRecursive(word1.substr(1), word2.substr(1));

    // insert word2's first character to word1's front (which is equal to move word2 forward by 1 character).
    const int i = 1 + minDistanceRecursive(word1, word2.substr(1));

    return min(d, min(r, i));
}
int minDistanceDP2D(string& word1, string& word2)
{
    // Defines DP[i][j] as the minimum edit distance between word1[0..i-1] and word2[0..j-1], allocate DP array with one additional row and column.
    vector<vector<unsigned>> dp(word1.length() + 1, vector<unsigned>(word2.length() + 1));

    // it means, when word1 is empty, then minimum edit distance between word1 and word2, is depends on length of word2.
    for (unsigned j = 0; j < dp[0].size(); ++j)
    {
        dp[0][j] = j;
    }

    // it means, when word2 is empty, then minimum edit distance between word1 and word2, is depends on length of word1.
    for (unsigned i = 0; i < dp.size(); ++i)
    {
        dp[i][0] = i;
    }

    for (unsigned i = 1; i < dp.size(); ++i)
    {
        for (unsigned j = 1; j < dp[i].size(); ++j)
        {
            if (word1[i - 1] == word2[j - 1])
            {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else
            {
                // delete word[i-1]
                const unsigned d = dp[i - 1][j] + 1;

                // replace word1[i-1] with word2[j-1]
                const unsigned r = dp[i - 1][j - 1] + 1;

                // insert word[j-1] to word1[i-1]
                const unsigned s = dp[i][j - 1] + 1;

                dp[i][j] = min(d, min(r, s));
            }
        }
    }

    return dp.back().back();
}
int minDistanceDP(string& word1, string& word2)
{
    // previous solution use a 2 dimensions array, for dp[i][j], actually only use its top element, top left element and left element,
    // so we can use a 1 dimension array to store previous row plus an integer to store dp[i][j-1], thus we can reduce space complexity.
    vector<unsigned> dp(word2.length() + 1);

    for (unsigned i = 0; i < dp.size(); ++i)
    {
        dp[i] = i;
    }

    for (unsigned i = 1; i <= word1.length(); ++i)
    {
        dp[0] = i;
        for (unsigned j = 1, topleft = i - 1; j < dp.size(); ++j)
        {
            // when a cell dp[j] hasn't been changed, itself is dp[i-1][j], its left cell's current value is dp[i][j-1], left cell's previous
            // value (saved in topleft) is dp[i-1][j-1].
            const unsigned temp = dp[j]; //dp[j]'s before-change value is dp[j+1]'s topleft dp[i-1][j-1].
            if (word1[i - 1] == word2[j - 1])
            {
                dp[j] = topleft;
            }
            else
            {
                // delete word[i-1]
                const unsigned d = dp[j] + 1;

                // replace word1[i-1] with word2[j-1]
                const unsigned r = topleft + 1;

                // insert word[j-1] to word1[i-1]
                const unsigned s = dp[j - 1] + 1;

                dp[j] = min(d, min(r, s));
            }

            topleft = temp;
        }
    }

    return dp.back();
}
int minDistance(string word1, string word2)
{
    return minDistanceDP(word1, word2);
}

// 73. Set Matrix Zeros
void setZeroesUseExtraSpace(vector<vector<int>>& matrix)
{
    vector<unsigned> rows(matrix.size(), 0);
    vector<unsigned> cols(matrix[0].size(), 0);

    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        for (unsigned j = 0; j < matrix[i].size(); ++j)
        {
            if (matrix[i][j] == 0)
            {
                rows[i] = 1;
                cols[j] = 1;
            }
        }
    }

    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        for (unsigned j = 0; j < matrix[i].size(); ++j)
        {
            if (rows[i] == 1 || cols[j] == 1)
            {
                matrix[i][j] = 0;
            }
        }
    }
}
void setZeroesNoExtraSpace(vector<vector<int>>& matrix)
{
    // Idea is to use first column to mark rows that need to be cleared.
    bool clearFirstColumn = false;
    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        if (matrix[i][0] == 0)
        {
            clearFirstColumn = true;
            break;
        }
    }

    // Go through column 1 -> column n
    for (unsigned j = 1, i; j < matrix[0].size(); ++j)
    {
        bool clearCurrentColumn = false;
        for (i = 0; i < matrix.size(); ++i)
        {
            // When matrix[i][j] == 0, row i and column j will be cleared. For row i, set matrix[i][0] to 0, row i will be cleared later.
            // For column j, it will be cleared now.
            if (matrix[i][j] == 0)
            {
                clearCurrentColumn = true;
                matrix[i][0] = 0;
            }
        }

        // Clear current column
        if (clearCurrentColumn)
        {
            for (i = 0; i < matrix.size(); ++i)
            {
                matrix[i][j] = 0;
            }
        }
    }

    // Now clear rows. Go through column 0, for matrix[i][0] == 0, set row i to 0
    for (unsigned i = 0; i < matrix.size(); ++i)
    {
        if (matrix[i][0] == 0)
        {
            for (unsigned j = 1; j < matrix[i].size(); ++j)
            {
                matrix[i][j] = 0;
            }
        }
    }

    // finally check if column 0 should be cleared also.
    if (clearFirstColumn)
    {
        for (unsigned i = 0; i < matrix.size(); ++i)
        {
            matrix[i][0] = 0;
        }
    }
}
void setZeroes(vector<vector<int>>& matrix)
{
    return setZeroesNoExtraSpace(matrix);
}

// 74. Search a 2D Matrix
bool searchMatrix(vector<vector<int>>& matrix, int target)
{
    if (matrix.empty() || matrix[0].empty())
    {
        return false;
    }

    int low;
    int high;

    // first search in first column, use binary search
    for (low = 0, high = matrix.size() - 1; low <= high;)
    {
        const int middle = (low + high) / 2;
        if (matrix[middle][0] < target)
        {
            low = middle + 1;
        }
        else if (matrix[middle][0] > target)
        {
            high = middle - 1;
        }
        else
        {
            return true;
        }
    }

    // when above loop ends, row[high] is the row
    const int row = high;
    if (row >= 0)
    {
        for (low = 0, high = matrix[row].size() - 1; low <= high;)
        {
            const int middle = (low + high) / 2;
            if (matrix[row][middle] < target)
            {
                low = middle + 1;
            }
            else if (matrix[row][middle] > target)
            {
                high = middle - 1;
            }
            else
            {
                return true;
            }
        }
    }

    return false;
}

// 75. Sort Colors
// partitionByPivot is a generic helper function, given a pivot value, partitioning nums[] into 3 partitions: <pivot, ==pivot, >pivot.
void partitionByPivot(vector<int>& nums, const int pivot)
{
    // Our goal is to partition nums[] into 3 partitions, less, equal and greater. Initially they are all empty except a temporary 
    // partition called unprocessed which starts from 0 and ends at nums.size()-1. We keep taking element from unprocessed partition
    // and inserting into corresponding partition until unprocessed partition becomes empty. Here are the start and end of each
    // partitions:
    // [0 .. less] is less partition
    // [less+1 .. unprocessed-1] is equal partition
    // [unprocessed .. greater-1] is unprocessed partition
    // [greater .. nums.size()-1] is greater partition
    for (int less = -1, unprocessed = 0, greater = nums.size(); unprocessed < greater;)
    {
        if (nums[unprocessed] < pivot)
        {
            // When inserting into less partition, actually swapping nums[unprocessed] with first element of equal partition.
            swap(nums[unprocessed++], nums[++less]);
        }
        else if (nums[unprocessed] > pivot)
        {
            swap(nums[unprocessed], nums[--greater]);
        }
        else // remain in pivot partition, just update unprocessed
        {
            ++unprocessed;
        }
    }
}
void sortColors(vector<int>& nums)
{
    return partitionByPivot(nums, 1);
}

// 76. Minimum Window Substring
// Use a dictionary to record occurrence of each unique character in string t, because order of characters in t is not considered.
// Starting from beginning of string s, for each character that appears in t, decrease the occurrence in dictionary, when the dictionary
// values are all 0, it means we found a window in s that contains all characters in t.
// Next step is try to shrink window's size by increase its left boundary, during the time, the dictionary must keep all 0, if not, 
// current window cannot be smaller, it is time to move right boundary to search for next window.
string minWindow(string s, string t)
{
    unordered_map<char, int> charMap;
    for (char c : t)
    {
        ++charMap[c];
    }

    string result;
    for (unsigned left = 0, right = 0, countOfUnusedChars = t.size(); right < s.length(); ++right)
    {
        // Expand window's right boundary until get a window s[left..right] that contains all characters in t.
        if (charMap.find(s[right]) != charMap.end())
        {
            if (charMap[s[right]]-- > 0)
            {
                --countOfUnusedChars;
            }

            // Found a valid window, try to shrink current window by retracting its left boundary, should keep countOfUnusedChars == 0
            while (countOfUnusedChars == 0)
            {
                string currentWindow = s.substr(left, right - left + 1);
                if (result.length() == 0 || currentWindow.length() < result.length())
                {
                    result = currentWindow;
                }

                if (charMap.find(s[left]) == charMap.end()) // s[left] is not in t, it is safe to exclude it from current window.
                {
                    ++left;
                }
                else if (charMap[s[left++]]++ == 0)
                {
                    // if s[left] exists in t, and if we are going to exclude it from current window, we need to know if current
                    // window still contains all characters in t. 
                    // If charMap[s[left]] < 0, it means window s[left..right] contains one or more additional s[left], it's safe
                    // to exclude current s[left], s[left+1 .. right] still contains all characters in s.
                    // If charMap[s[left]] == 0, it means if we exclude s[left], window s[left+1..right] no longer contains all
                    // characters in s, so this inner loop will end and outer loop starts to expand right boundary to search for
                    // next valid window.
                    ++countOfUnusedChars;
                }
            }
        }
    }

    return result;
}

// 77. Combinations
// Solution 1, use DFS search
void dfsCombineImpl(const int n, const int k, vector<vector<int>>& result, vector<int>& path, int currentNumber)
{
    if (path.size() == k)
    {
        result.push_back(path);
        return;
    }

    // Note here we always keep trying from small number to big number. The reason is combination doesn't consider order.
    // The easiest way to achieve this is for a group of numbers, we only pick up the sorted sequence.
    for (unsigned i = currentNumber; i <= n; ++i)
    {
        path.push_back(i);
        dfsCombineImpl(n, k, result, path, i + 1); // i + 1 means no duplicate numbers.
        path.pop_back();
    }
}
vector<vector<int>> combineUseDFS(int n, int k)
{
    vector<vector<int>> results;
    vector<int> path;

    dfsCombineImpl(n, k, results, path, 1);
    return results;
}
// Solution 2, use recursion
// This is a DP-like thought. Given N numbers, consider Nth number, if choose it, then problem becomes:
// get all possible combinations of K-1 numbers out of 1, 2, ... N-1; if DON'T choose the Nth number,
// problem becomes: get all possible combinations of K numbers out of 1, 2, ... N-1. So C(N, K) = C(N-1, K-1) + C(N-1, K)
vector<vector<int>> combineUseRecursion(int n, int k)
{
    if (k > n || k < 0)
    {
        return {};
    }

    if (k == 0)
    {
        return { {} };
    }

    // This we get all possible combinations of K-1 numbers out of 1, 2, ... N-1, back to the analysis at the beginning,
    // this implies that the number N is chosen, so for each combination here, append number N. Why appending here, since
    // we need to keep whole sequence sorted order, to make no duplication.
    vector<vector<int>> results = combineUseRecursion(n - 1, k - 1);
    for (vector<int>& c : results)
    {
        c.push_back(n);
    }

    for (vector<int>& c : combineUseRecursion(n - 1, k))
    {
        results.push_back(c);
    }

    return results;
}
vector<vector<int>> combine(int n, int k)
{
    return combineUseDFS(n, k);
}

// 78. Subsets
void dfsSubsets(vector<vector<int>>& results, vector<int>& path, vector<int>& nums, unsigned currentIndex)
{
    if (currentIndex == nums.size()) // all numbers have been used (choose or not choose)
    {
        results.push_back(path);
        return;
    }

    // Not like DFS permutation, at every step(index), we only have 2 choices, choose current number or don't choose.
    dfsSubsets(results, path, nums, currentIndex + 1); // don't choose nums[currentIndex]

    path.push_back(nums[currentIndex]);
    dfsSubsets(results, path, nums, currentIndex + 1); // choose nums[currentIndex]
    path.pop_back();
}
vector<vector<int>> subsets(vector<int>& nums)
{
    vector<vector<int>> results;
    vector<int> path;

    dfsSubsets(results, path, nums, 0);
    return results;
}

// 79. Word Search
bool dfsSearchWord(vector<vector<char>>& board, const string& word, unsigned currentIndex, int i, int j)
{
    // visited, or not a valid character in word.
    if (board[i][j] == '*' || board[i][j] != word[currentIndex])
    {
        return false;
    }

    // All characters in word have been found
    if (currentIndex == word.length() - 1)
    {
        return true;
    }

    const char current = board[i][j];
    board[i][j] = '*';

    if ((i > 0 && dfsSearchWord(board, word, currentIndex + 1, i - 1, j)) ||                    //top
        (j < board[i].size() - 1 && dfsSearchWord(board, word, currentIndex + 1, i, j + 1)) ||  //right
        (i < board.size() - 1 && dfsSearchWord(board, word, currentIndex + 1, i + 1, j)) ||     //bottom
        (j > 0 && dfsSearchWord(board, word, currentIndex + 1, i, j - 1)))                      //left
    {
        return true;
    }

    // When current location cannot proceed after having tried all 4 directions, we must restore current board[i][j]
    // before roll back, this is very important! It is because current cell may be visited and used later
    // we only change cell to * for the cells on current path.
    board[i][j] = current;
    return false;
}
bool exist(vector<vector<char>>& board, string word)
{
    for (unsigned i = 0; i < board.size(); ++i)
    {
        for (unsigned j = 0; j < board[i].size(); ++j)
        {
            if (board[i][j] == word[0] && dfsSearchWord(board, word, 0, i, j)) // start to search
            {
                return true;
            }
        }
    }

    return false;
}

// 80. Remove Duplicates from Sorted Array II
int removeDuplicates2(vector<int>& nums)
{
    return removeDuplicatesK(nums, 2);
}

// 81. Search in Rotated Sorted Array II
// In problem 33, we use nums[mid] to compare with nums[left] or nums[right] to determine whether mid is fall in left part or right part.
// Now when array contains duplication, it no longer works, since it is possible that nums[mid] == nums[left] == nums[right], e.g.:
// {1, 3, 1, 1, 1}, or {2, 2, 2, 1, 2}, in this case, we shrink left and right simultaneously.
bool search2(vector<int>& nums, int target)
{
    for (int left = 0, right = nums.size() - 1; left <= right; )
    {
        const int mid = (left + right) / 2;

        if (nums[mid] < target)
        {
            if (nums[mid] == nums[left] && nums[mid] == nums[right])
            {
                ++left;
                --right;
            }
            else if (nums[left] > nums[mid] && target > nums[right])
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        else if (nums[mid] > target)
        {
            if (nums[mid] == nums[left] && nums[mid] == nums[right])
            {
                ++left;
                --right;
            }
            else if (nums[mid] > nums[right] && target < nums[left])
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
        else // if (nums[mid] == target)
        {
            return true;
        }
    }

    return false;
}
bool search2_2(vector<int>& nums, int target)
{
    for (int left = 0, right = nums.size() - 1; left <= right;)
    {
        const int mid = (left + right) / 2;
        if (nums[mid] == target)
        {
            return true;
        }

        if (nums[left] == nums[mid] && nums[mid] == nums[right])
        {
            ++left;
            --right;
        }
        else if (nums[left] <= nums[mid])   // mid is on left part.
        {
            if (target >= nums[left] && target < nums[mid]) // nums[left] <= target < nums[mid]
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        else if (nums[right] >= nums[mid]) // mid is on right part.
        {
            if (target > nums[mid] && target <= nums[right]) // nums[mid] < target <= nums[right]
            {
                left = mid + 1;
            }
            else
            {
                right = mid - 1;
            }
        }
    }

    return false;
}

// 82. Remove Duplicates from Sorted List II
ListNode* deleteDuplicates2(ListNode* head)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    for (ListNode* current = &dummyHead, *runner; current != nullptr;)
    {
        for (runner = current->next; runner != nullptr && runner->val == current->next->val; runner = runner->next);

        // If current->next->next == runner, then current->next only appears 1 times, no need to delete it, otherwise
        // we need to remove nodes from current->next to q->prev.
        if (current->next != nullptr && current->next->next != runner)
        {
            // If no need to care about memory leak, we can just do current->next = runner;
            // Otherwise, we need to delete nodes from current->next to q->prev.
            // NOTE: In this algorithm, we always check if current->next has duplication, so don't 
            // move current forward after following loop ends.
            while (current->next != runner)
            {
                ListNode* toBeDeleted = current->next;
                current->next = toBeDeleted->next;
                delete toBeDeleted;
            }
        }
        else
        {
            current = current->next;
        }
    }

    return dummyHead.next;
}
ListNode* deleteDuplicates2_2(ListNode* head)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    // use two pointers, prev - track the node before the dup nodes; runner - to find the last node of duplications.
    for (ListNode* prev = &dummyHead, *runner = head; runner != nullptr;)
    {
        for (; runner->next != nullptr && runner->val == runner->next->val; runner = runner->next);

        if (prev->next != runner)
        {
            while (prev->next != runner->next)
            {
                ListNode* toBeDeleted = prev->next;
                prev->next = toBeDeleted->next;
                delete toBeDeleted;
            }

            runner = prev->next;
        }
        else
        {
            prev = prev->next;
            runner = runner->next;
        }
    }

    return dummyHead.next;
}

// 83. Remove Duplicates from Sorted List
ListNode* deleteDuplicates(ListNode* head)
{
    for (ListNode* p = head; p != nullptr;)
    {
        if (p->next != nullptr && p->next->val == p->val) // delete p->next
        {
            ListNode* toBeDeleted = p->next;
            p->next = toBeDeleted->next;
            delete toBeDeleted;
        }
        else
        {
            p = p->next;
        }
    }

    return head;
}

// 84. Largest Rectangle in Histogram
// Given a bar, the max rectangle it can build is determined by the its left and right closest bars that lower than it.
// For each bar, scan its left and right until encounter a bar that lower than it, then the max rectangle current bar
// can build is (right - left - 1) * heights[i], wo do this for each bar, then we get the max rectangle. 
// This is a brute force solution and its time complexity is O(n^2).
int largestRectangleAreaUseBruteForce(vector<int>& heights)
{
    int maxRect = 0;
    for (int i = 0, left, right; i < heights.size(); ++i)
    {
        // for each i, search its left and right until we saw a lower than i
        for (left = i; left >= 0 && heights[left] >= heights[i]; --left);
        for (right = i; right < heights.size() && heights[right] >= heights[i]; ++right);
        const int currentRect = (right - left - 1) * heights[i];
        if (currentRect > maxRect)
        {
            maxRect = currentRect;
        }
    }

    return maxRect;
}
// As we analyzed, for each bar, its max rectangle is determined by left and right closest bars that lower than it (let's call
// them left and right boundaries). The brute force solution is slow because for each bar we need to search for left and right
// boundary, if for each bar we can get left and right boundaries' index in O(1) time then we can improve time complexity from
// O(n^2) to O(n), here comes the increasing stack solution:
// We use an increasing stack, in anytime, the elements in stack are in ascending order, from bottom to top. If current bar is
// higher than stack top, push its index to stack, until we encounter a bar that lower than current stack top, we start to pop
// bars until stack top is no longer higher than current bar. During popping we calculate the rectangle that every popped bar
// can build, and update the max rectangle so far.
// When we encounter a bar lower than stack's current top, there are 2 facts:
// (1) the current bar which is lower than stack top, is stack top bar's right boundary.
// (2) the next stack top is the stack top bar's left boundary, if stack becomes empty after pop then the left boundary is -1.
int largestRectangleAreaUseStack(vector<int>& heights)
{
    int maxRect = 0;
    stack<int> stack;
    for (unsigned i = 0; i <= heights.size(); )
    {
        // This is all conditions that do pushing, either stack is empty or current bar is no shorter than stack top.
        if (stack.empty() || i < heights.size() && heights[i] >= heights[stack.top()])
        {
            stack.push(i++);
        }
        else // If it is not the push condition, then we do pop. No need to add an inner loop, just don't update i after pop.
        {
            const int currentHeight = heights[stack.top()];
            stack.pop();

            // right boundary is i, left boundary is stack.top()
            const int rect = currentHeight * (stack.empty() ? i : i - stack.top() - 1);
            if (rect > maxRect)
            {
                maxRect = rect;
            }
        }
    }

    return maxRect;
}
int largestRectangleArea(vector<int>& heights)
{
    return largestRectangleAreaUseStack(heights);
}

// 85. Maximal Rectangle
int maximalRectangleUse84(vector<vector<char>>& matrix)
{
    int maxRect = 0;
    vector<int> packed(matrix.empty() ? 0 : matrix[0].size(), 0);
    for (vector<char>& row : matrix) // given row[i], packed[] = row[0] + row[1] + row[2] + ... + row[i]
    {
        // pack row[i] to packed[] array, if in current row we encounter a '0', then reset packed[j] to 0.
        for (unsigned j = 0; j < row.size(); packed[j] = row[j] == '1' ? packed[j] + 1 : 0, ++j);

        const int currentMaxRect = largestRectangleArea(packed);
        if (currentMaxRect > maxRect)
        {
            maxRect = currentMaxRect;
        }
    }

    return maxRect;
}
int maximalRectangleUseDP(vector<vector<char>>& matrix)
{
    int maxRect = 0;

    if (!matrix.empty())
    {
        vector<vector<int>> dp(matrix.size(), vector<int>(matrix[0].size(), 0));
        for (unsigned i = 0; i < matrix.size(); ++i)
        {
            for (unsigned j = 0; j < matrix[i].size(); ++j)
            {
                if (matrix[i][j] == '1')
                {
                    dp[i][j] = 1 + (j > 0 ? dp[i][j - 1] : 0);

                    for (int width = dp[i][j], k = i; k >= 0 && dp[k][j] > 0; --k)
                    {
                        if (dp[k][j] < width)
                        {
                            width = dp[k][j];
                        }

                        const int rect = (i - k + 1) * width;
                        if (rect > maxRect)
                        {
                            maxRect = rect;
                        }
                    }
                }
                else
                {
                    dp[i][j] = 0;
                }
            }
        }
    }

    return maxRect;
}
int maximalRectangle(vector<vector<char>>& matrix)
{
    return maximalRectangleUseDP(matrix);
}

// 86. Partition List
ListNode* partition(ListNode* head, int x)
{
    ListNode small(-1);
    ListNode great(-1);
    ListNode* p = &small;
    ListNode* q = &great;
    for (; head != nullptr; head = head->next)
    {
        if (head->val < x)
        {
            p->next = head;
            p = p->next;
        }
        else
        {
            q->next = head;
            q = q->next;
        }
    }

    p->next = great.next; // splice small and great
    q->next = nullptr; // seal the whole list.

    return small.next;
}

// 87. Scramble String
bool isScramble(const string& s1, const string& s2)
{
    if (s1 == s2)
    {
        return true;
    }

    // make sure s1 and s2 have same character set and each char appears same times.
    // this is an important optimization to avoid unnecessary recursion.
    unordered_map<char, int> occurance;
    for (unsigned i = 0; i < s2.length(); ++i)
    {
        occurance[s1[i]]++;
        occurance[s2[i]]--;
    }

    for (const pair<char, int> kvp : occurance)
    {
        if (kvp.second != 0)
        {
            return false;
        }
    }

    // split s2 into 2 parts, left part: s2[0..leftlen-1], right part: s2[leftlen..s2.length()-1].
    for (unsigned leftlen = 1; leftlen < s2.length(); ++leftlen)
    {
        // there are 2 cases, left part and right part are not swapped, or swapped.
        if (isScramble(s1.substr(0, leftlen), s2.substr(0, leftlen)) &&
            isScramble(s1.substr(leftlen, s1.length() - leftlen), s2.substr(leftlen, s2.length() - leftlen)) ||
            isScramble(s1.substr(s1.length() - leftlen, leftlen), s2.substr(0, leftlen)) &&
            isScramble(s1.substr(0, s1.length() - leftlen), s2.substr(leftlen, s2.length() - leftlen)))
        {
            return true;
        }
    }

    return false;
}

// 88. Merge Sorted Array
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n)
{
    // to avoid copying/moving elements in nums1, let's start from location nums1[m+n-1].
    for (int last = m-- + n-- - 1; last >= 0 && (m >= 0 || n >= 0); --last)
    {
        if (m >= 0 && n >= 0)
        {
            if (nums1[m] > nums2[n])
            {
                nums1[last] = nums1[m--];
            }
            else
            {
                nums1[last] = nums2[n--];
            }
        }
        else if (m >= 0)
        {
            nums1[last] = nums1[m--];
        }
        else if (n >= 0)
        {
            nums1[last] = nums2[n--];
        }
    }
}

// 89. Gray Code
// There are 2 easy ways to generate gray code. 
// First is using recursion, based on grayCode(n-1) we can quick get grayCode(n). Looking at the gray codes when n == 3:
// 000
// 001
// 011
// 010
// 110
// 111
// 101
// 100
// first 4 codes are 0 + {00, 01, 11, 10}, that in bracket are the gray codes when n == 2.
// last 4 codes are 1 + {10, 11, 01, 00), that in bracket are the reverse order of gray codes when n == 2.
// So, here is the rule, first we get gray codes of n-1, then traverse the vector, prefix with 0; then reverse traverse the
// vector, prefix with 1. 
// In practical, because if we prefix a number with 0, its value will not change, so we just need to do a reverse traverse
// and for each number we prefix it with 1.
vector<int> grayCodeUseRecursion(const int n)
{
    if (n == 0)
    {
        return { 0 };
    }

    vector<int> previousGrayCodes = grayCodeUseRecursion(n - 1);
    for (int i = previousGrayCodes.size() - 1; i >= 0; --i)
    {
        previousGrayCodes.push_back(previousGrayCodes[i] | 1 << (n - 1));
    }

    return previousGrayCodes;
}
// The other solution is to calculate gray code directly, let's say a integer is represented in binary as b(n-1)b(n-2)...b(2)b(1)b(0),
// its corresponding gray code is g(n-1)g(n-2)...g(2)g(1)g(0), the process is:
// b(n-1) becomes g(n-1), then g(i) = b(i+1) XOR b(i).
//         b(n-1) b(n-2) ... b(2) b(1) b(0)
//   (XOR)   0    b(n-1)     b(3) b(2) b(1)
//  ----------------------------------------
//         g(n-1) g(n-2)     g(2) g(1) g(0)
// So, given an unsigned integer i, we just let i XOR i>>1, then we get the corresponding gray code.
vector<int> grayCodeUseXOR(const int n)
{
    vector<int> codes;
    codes.reserve(1 << n);
    for (int i = 0; i < 1 << n; ++i)
    {
        codes.push_back(i ^ i >> 1);
    }

    return codes;
}
vector<int> grayCode(const int n)
{
    return grayCodeUseXOR(n);
}

// 90. Subsets II
// All subsets of nums[0 .. nums.size()-1] can be grouped into:
// subsets that nums[0] is the first element.
// subsets that nums[1] is the first element.
// ...
// subsets that nums[i] (0 <= i <= nums.size()-1) is the first element.
// ...
// subsets that nums[nums.size()-1] is the first element.
// For the group that has nums[i] is the first element, it can be generated in this way: nums[i] + { subsets of nums[i+1 .. nums.size()-1] }.
// So it is a recursion problem, here this helper function recursively finds all subsets of nums[currentIndex .. nums.size()-1].
void dfsSearchSubsetWithDup(vector<vector<int>>& subsets, vector<int>& currentSubset, vector<int>& nums, const unsigned currentIndex)
{
    subsets.push_back(currentSubset);
    for (unsigned i = currentIndex; i < nums.size(); ++i) // Select nums[i] as the first element respectively
    {
        if (i == currentIndex || nums[i] != nums[i - 1])
        {
            currentSubset.push_back(nums[i]);
            dfsSearchSubsetWithDup(subsets, currentSubset, nums, i + 1);
            currentSubset.pop_back();
        }
    }
}
vector<vector<int>> subsetsWithDup(vector<int>& nums)
{
    vector<vector<int>> subsets;
    vector<int> currentSubset;

    sort(nums.begin(), nums.end());
    dfsSearchSubsetWithDup(subsets, currentSubset, nums, 0);

    return subsets;
}

// 91. Decode Ways
int numDecodingsUseRecursion(const string& s)
{
    if (s.empty()) // Whole string has been decoded.
    {
        return 1;
    }

    if (s[0] == '0') // Means decoding cannot proceed.
    {
        return 0;
    }

    int ways = numDecodingsUseRecursion(s.substr(1));

    if (s.length() >= 2 && (s[0] - '0') * 10 + (s[1] - '0') <= 26)
    {
        ways += numDecodingsUseRecursion(s.substr(2));
    }

    return ways;
}
// We define DP[i] as decoding ways of substring starts from index i (which is s[i .. s.length()-1]), so dp[0] is final answer.
// DP[i] = s[i] == '0' ? 0
//                       DP[i+1] + s[i,i+2] <= 26 ? 0
//                                                  DP[i+2]
// Since anytime we just need DP[i+1] and DP[i+2] so the space complexity can be optimized to O(1), we use 2 integers to store
// DP[i+1] and DP[i+2].
int numDecodingsUseDP(const string& s)
{
    vector<int> dp(2, 0);
    if (!s.empty())
    {
        dp[1] = 1; // store dp[i+2]
        dp[0] = s[s.length() - 1] == '0' ? 0 : 1; // store dp[i+1]

        for (int i = s.length() - 2; i >= 0; --i)
        {
            if (s[i] == '0')
            {
                dp[1] = dp[0];
                dp[0] = 0;
            }
            else
            {
                // For single digit, dp[i] = dp[i+1], so no need to update dp[0]'s value. Just when 2 digits are possible,
                // we need to add dp[1] to dp[0] to get new dp[0].
                const int dp0 = dp[0];
                if ((s[i] - '0') * 10 + (s[i + 1] - '0') <= 26)
                {
                    dp[0] += dp[1];
                }

                dp[1] = dp0;
            }
        }
    }

    return dp[0];
}
int numDecodings(const string& s)
{
    return numDecodingsUseDP(s);
}

// 92. Reverse Linked List II
// Note, since it is guaranteed that 1 <= m <= n < list length, so no needs to check m and n, node m and node n are always valid.
ListNode* reverseBetween(ListNode* head, int m, int n)
{
    ListNode dummyHead(0);
    dummyHead.next = head;

    ListNode* p = &dummyHead;
    for (n -= m; m > 1; --m, p = p->next); // find the predecessor of node m and update n relative to m.

    ListNode* q = p->next; // q points to node m.
    ListNode* r = q->next;
    for (ListNode* t; n > 0; --n, t = r->next, r->next = q, q = r, r = t); // when this loop ends, n==0, so it executes n+1 times, q points to node n, r points to node n's next node.

    p->next->next = r;
    p->next = q;

    return dummyHead.next;
}

// 93. Restore IP Addresses
void dfsSearchIPAddresses(vector<string>& ips, vector<string>& currentOctets, const string& s, unsigned currentIndex)
{
    // the longest IP address like 255.255.255.255 contains 12 digits.
    if (s.length() > 12)
    {
        return;
    }

    if (s.length() == currentIndex && currentOctets.size() == 4)
    {
        ips.push_back(currentOctets[0] + '.' + currentOctets[1] + '.' + currentOctets[2] + '.' + currentOctets[3]);
        return;
    }

    // There are 3 cases, current ip octet can be 1, 2 or 3 digits (<= 255), and when octet is 2 or 3 digits, it cannot start with 0
    if (currentIndex + 1 <= s.length())
    {
        currentOctets.push_back(s.substr(currentIndex, 1));
        dfsSearchIPAddresses(ips, currentOctets, s, currentIndex + 1);
        currentOctets.pop_back();
    }

    if (s[currentIndex] != '0')
    {
        if (currentIndex + 2 <= s.length())
        {
            currentOctets.push_back(s.substr(currentIndex, 2));
            dfsSearchIPAddresses(ips, currentOctets, s, currentIndex + 2);
            currentOctets.pop_back();
        }

        if (currentIndex + 3 <= s.length() && (s[currentIndex] - '0') * 100 + (s[currentIndex + 1] - '0') * 10 + (s[currentIndex + 2] - '0') <= 255)
        {
            currentOctets.push_back(s.substr(currentIndex, 3));
            dfsSearchIPAddresses(ips, currentOctets, s, currentIndex + 3);
            currentOctets.pop_back();
        }
    }
}
vector<string> restoreIpAddresses(const string& s)
{
    vector<string> ips;
    vector<string> currentOctets;
    dfsSearchIPAddresses(ips, currentOctets, s, 0);
    return ips;
}

// 94. Binary Tree Inorder Traversal
vector<int> inorderTraversalUseStack(TreeNode* root)
{
    vector<int> result;
    stack<TreeNode*> stack;

    for (; root != nullptr; stack.push(root), root = root->left);

    while (!stack.empty())
    {
        root = stack.top();
        result.push_back(root->val);
        stack.pop();
        if (root->right != nullptr)
        {
            root = root->right;
            for (; root != nullptr; stack.push(root), root = root->left);
        }
    }

    return result;
}
vector<int> inorderMorrisTraversal(TreeNode* root)
{
    vector<int> result;
    for (TreeNode* current = root; current != nullptr;)
    {
        if (current->left != nullptr)
        {
            // Before exploring into the left child tree, find the in-order predecessor of current node and borrow its unused right field
            // to point to current node, so we can return to current node after left child tree has been visited. The predecessor is the
            // right most node in the left child tree.
            TreeNode* predecessor = current->left;
            for (; predecessor->right != nullptr && predecessor->right != current; predecessor = predecessor->right);

            if (predecessor->right == nullptr) // create a thread, then proceed to left child tree.
            {
                predecessor->right = current;
                current = current->left;
            }
            else
            {
                // We found a already connected thread, it means current node's left child tree has been visited, so we need to do:
                // (1) revert the thread, (2) visit current node (according to in-order, after left tree has been visited, we should 
                // visit the root which is current node, (3) proceed to visit current node's right child tree.
                predecessor->right = nullptr;
                result.push_back(current->val);
                current = current->right;
            }
        }
        else
        {
            result.push_back(current->val);
            current = current->right; // here right could point to real right child, or it is used as thread, so don't revert thread here.
        }
    }

    return result;
}
vector<int> inorderTraversal(TreeNode* root)
{
    return inorderMorrisTraversal(root);
}

// 95. Unique Binary Search Trees II
vector<TreeNode*> generateTreesUseRecursion(const int start, const int end)
{
    if (start > end)
    {
        return { nullptr };
    }

    vector<TreeNode*> trees;
    for (int i = start; i <= end; ++i)
    {
        for (TreeNode* leftRoot : generateTreesUseRecursion(start, i - 1)) // left child, low..i - 1
        {
            for (TreeNode* rightRoot : generateTreesUseRecursion(i + 1, end)) // right child i+1..high
            {
                TreeNode* root = new TreeNode(i);
                root->left = leftRoot;
                root->right = rightRoot;
                trees.push_back(root);
            }
        }
    }

    return trees;
}
vector<TreeNode*> generateTrees(const int n)
{
    return generateTreesUseRecursion(1, n);
}

// 96. Unique Binary Search Trees
// Thinking in this way, let's say there are i nodes, since one node is used as root so there are i-1 nodes remaining. Left subtree 
// can has 0, 1, 2, ... , i-1 nodes, accordingly right subtree can has i-1, i-2, ..., 1, 0 nodes. Assume F(i) is the count  of unique
// binary trees that have i nodes, then F(i) = F(0)*F(i-1) + F(1)*F(i-2) + F(2)*F(i-3) + ... + F(i-1)*F(0).
int numTrees(int n)
{
    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; ++i)
    {
        for (int j = 0; j <= i - 1; ++j)
        {
            dp[i] += dp[j] * dp[i - 1 - j];
        }
    }

    return dp.back();
}

// 97. Interleaving String
// How to think this problem? It makes it seems complicated if we start from s1 and s2 and think the combinations can get from s1
// and s2, let's thinking in a reversed way. Assume s3 is formed by interleaving s1 and s2, then s3's substring : s3.substr(0, k)
// should also be formed by interleaving s1's substring and s2's substring, let's say they are s1.substr(0, i) and s2.substr(0, j)
// (obviously i+j=k), then let's move one step further, think about s3.substr(0, k+1), actually we only need to think about the 
// new added element s3[k], if s3[k] == s1[i], we can say s3.substr(0, k+1) is formed by s1.substr(0, i+1) and s2.substr(0, j); if
// s[3] == s2[j], we can say s3.substr(0, k+1) is formed by s1.substr(0, i) and s2(0, j+1). Now it is more clear, let's define the
// DP[i][j] == whether s3.substr(0, i+j) is formed by interleaving of s1.substr(0, i) and s2.substr(0, j), so, 
// DP[i][j] = (whether s3.substr(0, i+j-1) is formed by interleaving of s1.substr(0, i) and s2.substr(0, j-1)) && s3[i+j] == s1[i]
//            ||
//            (whether s3.substr(0, i+j-1) is formed by interleaving of s1.substr(0, i-1) and s2.substr(0, j)) && s3[i+j] == s2[j]
//          = (DP[i][j-1] && s3[i+j] == s1[i]) || (DP[i-1][j] && s3[i+j] == s2[j])
bool isInterleaveUseDP(const string& s1, const string& s2, const string& s3)
{
    if (s1.length() + s2.length() != s3.length())
    {
        return false;
    }

    // Note we start from substr(0,0), so DP[][] has additional column and row, when i==0 and j==0, substrings are empty string, 
    // so DP[0][0] means empty string is formed by interleaving empty strings, obviously it is true.
    vector<vector<bool>> dp(s1.length() + 1, vector<bool>(s2.length() + 1, false));
    dp[0][0] = true;

    // Initialize first row and first column. For first row and first column, they mean, when s1 is empty or s2 is empty, whether
    // s3.substr(0,i) is formed by interleaving of the other string, but since s1 or s2 is empty, so actually no interleaving, just
    // string compare, DP[i][0] == whether s3.substr(0,i) == s1.substr(0,i); DP[0][j] == whether s3.substr(0,j)==s2.substr(0,j).
    for (unsigned i = 1; i <= s1.length(); ++i) // remember, i and j here means the substring length, so last element index is i-1 and j-1.
    {
        dp[i][0] = dp[i - 1][0] && s1[i - 1] == s3[i - 1];
    }

    for (unsigned j = 1; j <= s2.length(); ++j)
    {
        dp[0][j] = dp[0][j - 1] && s2[j - 1] == s3[j - 1];
    }

    for (unsigned i = 1; i <= s1.length(); ++i)
    {
        for (unsigned j = 1; j <= s2.length(); ++j)
        {
            dp[i][j] = (dp[i - 1][j] && s3[i + j - 1] == s1[i - 1]) || (dp[i][j - 1] && s3[i + j - 1] == s2[j - 1]);
        }
    }

    return dp.back().back();
}
bool bfsSearchInterleavingString(const string& s1, const string& s2, const string& s3)
{
    if (s1.length() + s2.length() != s3.length())
    {
        return false;
    }

    unordered_set<string> visited;
    queue<pair<unsigned, unsigned>> queue;
    queue.push(make_pair(0, 0));
    for (unsigned k = 0; !queue.empty(); ++k)
    {
        if (k == s3.length())
        {
            return true;
        }

        for (unsigned l = 0, currenQueueLength = queue.size(), i, j; l < currenQueueLength; ++l, queue.pop())
        {
            const pair<unsigned, unsigned> current = queue.front();
            i = current.first;
            j = current.second;

            if (i < s1.length() && s1[i] == s3[k] && visited.find(to_string(i + 1) + "," + to_string(j)) == visited.end())
            {
                visited.insert(to_string(i + 1) + "," + to_string(j));
                queue.push(make_pair(i + 1, j));
            }

            if (j < s2.length() && s2[j] == s3[k] && visited.find(to_string(i) + "," + to_string(j + 1)) == visited.end())
            {
                visited.insert(to_string(i) + "," + to_string(j + 1));
                queue.push(make_pair(i, j + 1));
            }
        }
    }

    return false;
}
// Imaging that in each step, we pick up a character from s1 or s2, after all characters in s1 and s2 have been used, we get an 
// interleaving string, if s3 is formed by interleaving s1 and s2, we should be able to construct s3 in this recursive way, this
// process is recursive so can be solved by recursion.
// The only problem is pure DFS will timeout, because it tries too many branches and most of branches we already know that cannot
// success, the basic idea is caching the pair (i,j) when we know that s1[i..s1.length()-1] and s2[j..s2.length()-1] cannot form 
// s3[k..s3.length()-1] via interleaving. This solution we call it cached DFS.
bool dfsSearchInterleavingString(unordered_set<string>& knownFailures, const string& s1, const unsigned i, const string& s2, const unsigned j, const string& s3, const unsigned k)
{
    if (knownFailures.find(to_string(i) + "," + to_string(j)) != knownFailures.end())
    {
        return false;
    }

    if (i == s1.length())
    {
        return s2.substr(j) == s3.substr(k);
    }

    if (j == s2.length())
    {
        return s1.substr(i) == s3.substr(k);
    }

    if (s1[i] == s3[k] && dfsSearchInterleavingString(knownFailures, s1, i + 1, s2, j, s3, k + 1))
    {
        return true;
    }

    if (s2[j] == s3[k] && dfsSearchInterleavingString(knownFailures, s1, i, s2, j + 1, s3, k + 1))
    {
        return true;
    }

    knownFailures.insert(to_string(i) + "," + to_string(j));
    return false;
}
bool isInterleave(const string& s1, const string& s2, const string& s3)
{
    if (s1.length() + s2.length() != s3.length())
    {
        return false;
    }

    unordered_set<string> knownFailures;
    return dfsSearchInterleavingString(knownFailures, s1, 0, s2, 0, s3, 0);
}

// 98. Validate Binary Search Tree
bool isValidBSTHelper(TreeNode* root, TreeNode* lowerBound, TreeNode* upperBound)
{
    if (root == nullptr)
    {
        return true;
    }

    if ((lowerBound == nullptr || root->val > lowerBound->val) && (upperBound == nullptr || root->val < upperBound->val))
    {
        return isValidBSTHelper(root->left, lowerBound, root) && isValidBSTHelper(root->right, root, upperBound);
    }

    return false;
}
bool isValidBST(TreeNode* root)
{
    return isValidBSTHelper(root, nullptr, nullptr);
}

// 99. Recover Binary Search Tree
// This problem doesn't need to consider the possible location of swapped nodes, we know that for BST its inorder is sorted,
// so if there are 2 nodes are swapped, that will reflect in its inorder, there will be a pair of values that are also swapped.
// So first O(n) solution is, we get inorder series first, then search for pairs that are not in ascending order. Note that
// the swapped pair could be adjacent or not, e.g.: let's say the inorder is [1, 3, 2, 4, 5, 6], the swapped pair are [3, 2],
// but for inorder [1, 5, 3, 4, 2, 6], the swapped pair is [5, 2], not the first swapped pair [5, 3]. So, to deal with such a
// case, we let swapped pair = the first swapped pair encounter, which is [5, 3] here, if we encounter another reversed order
// pair [4, 2], we just update the second value in pair, that is [5, 2].
void getInorder(TreeNode* root, vector<TreeNode*>& inorder)
{
    if (root != nullptr)
    {
        getInorder(root->left, inorder);
        inorder.push_back(root);
        getInorder(root->right, inorder);
    }
}
void recoverTreeNSpace(TreeNode* root)
{
    vector<TreeNode*> inorder;
    getInorder(root, inorder);
    TreeNode* a = nullptr;
    TreeNode* b = nullptr;
    for (unsigned i = 0; i < inorder.size() - 1; ++i)
    {
        if (inorder[i]->val > inorder[i + 1]->val)
        {
            if (a == nullptr)
            {
                a = inorder[i];
                b = inorder[i + 1];
            }
            else
            {
                b = inorder[i + 1];
            }
        }
    }

    swap(a->val, b->val);
}
// The O(1) time complexity solution is an optimization of O(n) solution, it is obviously that to check the swapped pair
// in inorder series, it is unnecessary to save whole inorder series, we can do this while traversing inorder. If we use
// stack, it is still O(n) space complexity (worst case, BST is a linked list), so Morris inorder traverse is the answer.
void recoverTree(TreeNode* root)
{
    TreeNode* a = nullptr;
    TreeNode* b = nullptr;
    for (TreeNode* previous = nullptr; root != nullptr;) // reuse root
    {
        bool visitCurrent = true;
        if (root->left != nullptr)
        {
            TreeNode* predecessor = root->left;
            for (; predecessor->right != nullptr && predecessor->right != root; predecessor = predecessor->right);

            if (predecessor->right == nullptr) // setup thread
            {
                predecessor->right = root;
                root = root->left;
                visitCurrent = false;
            }
            else if (predecessor->right == root) // left child tree has been visited, disconnect thread
            {
                predecessor->right = nullptr;
            }
        }

        if (visitCurrent)
        {
            if (previous != nullptr && previous->val > root->val)
            {
                if (a == nullptr)
                {
                    a = previous;
                    b = root;
                }
                else
                {
                    b = root;
                }
            }

            previous = root;
            root = root->right;
        }
    }

    swap(a->val, b->val);
}

// 100. Same Tree
bool isSameTree(TreeNode* p, TreeNode* q)
{
    if (p == nullptr || q == nullptr)
    {
        return p == q;
    }

    return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}

// 101. Symmetric Tree
bool isSymmetricUseRecursion(TreeNode* p, TreeNode* q)
{
    if (p == nullptr || q == nullptr)
    {
        return p == q;
    }

    return p->val == q->val && isSymmetricUseRecursion(p->left, q->right) && isSymmetricUseRecursion(p->right, q->left);
}
bool isSymmetric(TreeNode* root)
{
    if (root == nullptr)
    {
        return true;
    }

    return isSymmetricUseRecursion(root->left, root->right);
}
// The iterative solution is based on BFS. DFS will not work since we compare nodes from top to bottom, level by level.
// Instead of checking whether current level is a palindrome, we change the enqueue order to make two symmetric nodes
// are neighbors in queue. And since we always push a node's left child and right child to queue no matter it is null or
// not, so queue length keeps even, no need to check queue length parity.
bool isSymmetricIterative(TreeNode* root)
{
    if (root == nullptr)
    {
        return true;
    }

    queue<TreeNode*> queue;
    queue.push(root->left);
    queue.push(root->right);
    while (!queue.empty())
    {
        TreeNode* left = queue.front();
        queue.pop();
        TreeNode* right = queue.front();
        queue.pop();

        if (left == nullptr && right == nullptr)
        {
            continue;
        }

        if (left == nullptr || right == nullptr || left->val != right->val)
        {
            return false;
        }

        queue.push(left->left);
        queue.push(right->right);

        queue.push(left->right);
        queue.push(right->left);
    }

    return true;
}

// 102. Binary Tree Level Order Traversal
vector<vector<int>> levelOrderUseQueue(TreeNode* root)
{
    vector<vector<int>> results;
    queue<TreeNode*> queue;
    queue.push(root);
    while (!queue.empty())
    {
        vector<int> currentLevel;
        for (unsigned i = 0, len = queue.size(); i < len; ++i, queue.pop())
        {
            if (queue.front() != nullptr)
            {
                currentLevel.push_back(queue.front()->val);
                queue.push(queue.front()->left);
                queue.push(queue.front()->right);
            }
        }

        if (!currentLevel.empty())
        {
            results.push_back(currentLevel);
        }
    }

    return results;
}
// The recursive solution is based on per order traverse, we pass level as a parameter, when encounter level i, push
// current node value to results[i]. Since preorder is left to right, so we get each level in left to right order
void preOrderWithLevel(vector<vector<int>>& results, TreeNode* root, const unsigned level)
{
    if (root == nullptr)
    {
        return;
    }

    if (level == results.size())
    {
        results.emplace_back();
    }

    results[level].push_back(root->val);

    preOrderWithLevel(results, root->left, level + 1);
    preOrderWithLevel(results, root->right, level + 1);
}
vector<vector<int>> levelOrderUseRecursion(TreeNode* root)
{
    vector<vector<int>> results;
    preOrderWithLevel(results, root, 0);
    return results;
}
vector<vector<int>> levelOrder(TreeNode* root)
{
    return levelOrderUseQueue(root);
}

// 103. Binary Tree Zigzag Level Order Traversal
vector<vector<int>> zigzagLevelOrderUseQueue(TreeNode* root)
{
    vector<vector<int>> results;
    if (root != nullptr)
    {
        queue<TreeNode*> queue;
        queue.push(root);
        for (bool leftToRight = true; !queue.empty(); leftToRight = !leftToRight)
        {
            const unsigned len = queue.size();
            vector<int> currentLevel(len);
            for (unsigned i = 0; i < len; ++i, queue.pop())
            {
                currentLevel[leftToRight ? i : len - i - 1] = queue.front()->val;

                if (queue.front()->left != nullptr)
                {
                    queue.push(queue.front()->left);
                }

                if (queue.front()->right != nullptr)
                {
                    queue.push(queue.front()->right);
                }
            }

            results.push_back(currentLevel);
        }
    }

    return results;
}
void preOrderWithLevelAndOrder(vector<vector<int>>& results, TreeNode* root, const unsigned level, const bool leftToRight)
{
    if (root == nullptr)
    {
        return;
    }

    if (level == results.size())
    {
        results.emplace_back();
    }

    results[level].insert(leftToRight ? results[level].end() : results[level].begin(), root->val);
    preOrderWithLevelAndOrder(results, root->left, level + 1, !leftToRight);
    preOrderWithLevelAndOrder(results, root->right, level + 1, !leftToRight);
}
vector<vector<int>> zigzagLevelOrderUseRecursion(TreeNode* root)
{
    vector<vector<int>> results;
    preOrderWithLevelAndOrder(results, root, 0, true);
    return results;
}
vector<vector<int>> zigzagLevelOrder(TreeNode* root)
{
    return zigzagLevelOrderUseQueue(root);
}

// 104. Maximum Depth of Binary Tree
int maxDepth(TreeNode* root)
{
    return root == nullptr ? 0 : max(maxDepth(root->left), maxDepth(root->right)) + 1;
}

// 105. Construct Binary Tree from PreOrder and InOrder Traversal
// this helper function's purpose is to avoid sub array allocation and copy.
TreeNode* buildTree105Helper(vector<int>& preorder, const int ps, const int pe, vector<int>& inorder, const int is, const int ie)
{
    if (pe < ps || ie < is)
    {
        return nullptr;
    }

    TreeNode* root = new TreeNode(preorder[ps]); // first element in preorder is the root.
    int r;
    for (r = is; r <= ie && inorder[r] != preorder[ps]; ++r);

    // left child tree: inorder[is .. r-1], preorder[ps+1..ps+r-is]
    root->left = buildTree105Helper(preorder, ps + 1, ps + r - is, inorder, is, r - 1);

    // right child tree: inorder[r+1 .. ie], preorder[ps+r-is+1 .. pe]
    root->right = buildTree105Helper(preorder, ps + r - is + 1, pe, inorder, r + 1, ie);

    return root;
}
TreeNode* buildTree105(vector<int>& preorder, vector<int>& inorder)
{
    return buildTree105Helper(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
}

// 106. Construct Binary Tree from Inorder and Postorder Traversal
TreeNode* buildTree106Helper(vector<int>& inorder, const int is, const int ie, vector<int>& postorder, const int ps, const int pe)
{
    if (is > ie || ps > pe)
    {
        return nullptr;
    }

    TreeNode* root = new TreeNode(postorder[pe]); // last element in postorder is the root
    int r;
    for (r = is; r <= ie && inorder[r] != postorder[pe]; ++r);

    // left child tree: inorder[is, r-1], postorder[ps, ps+r-is-1]
    root->left = buildTree106Helper(inorder, is, r - 1, postorder, ps, ps + r - is - 1);

    // right child tree: inorder[r+1, ie], postorder[pe-ie+r, pe-1]
    root->right = buildTree106Helper(inorder, r + 1, ie, postorder, pe - ie + r, pe - 1);

    return root;
}
TreeNode* buildTree106(vector<int>& inorder, vector<int>& postorder)
{
    return buildTree106Helper(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
}

// 107. Binary Tree Level Order Traversal II
void levelOrderBottomDFS(vector<vector<int>>& results, TreeNode* root, const unsigned level)
{
    if (root == nullptr)
    {
        return;
    }

    if (level == results.size())
    {
        // this is a tricky part, we should not use emplace_back but always add new vector to the
        // beginning, the reason is when value is being pushed to vector, our code always assumes
        // that results.size() implies tree height, which is not correct. e.g.: for this tree:
        //               1
        //              / \
        //             2   3
        //                / \
        //               4   5
        // when we reach 2, since it is a leaf, so it needs to be pushed to results vector, but 
        // current results.size() is 2, so it will be pushed to results[2-1-1] = results[0], that
        // is not correct, in final results [[4,5], [2,3], [1]], 2 actually should be pushed to 
        // results[1]. How to fix it?, when we go to right child tree, we will see a new height 2
        // it means we need to add a new vector for this new level, if we append new vector to end
        // of results, then previous pushed 2 will stay in results[0], if we insert the new vector
        // to the beginning of results, then 2 get corrected, it now in results[1].
        results.insert(results.begin(), vector<int>());
    }

    // results.size() is the tree's current height (not the whole tree's height).
    results[results.size() - 1 - level].push_back(root->val);

    levelOrderBottomDFS(results, root->left, level + 1);
    levelOrderBottomDFS(results, root->right, level + 1);
}
vector<vector<int>> levelOrderBottom(TreeNode* root)
{
    vector<vector<int>> results;
    levelOrderBottomDFS(results, root, 0);
    return results;
}
vector<vector<int>> levelOrderBottomBFS(TreeNode* root)
{
    vector<vector<int>> results;
    queue<TreeNode*> queue;
    queue.push(root);

    while (!queue.empty())
    {
        vector<int> level;
        for (unsigned i = 0, len = queue.size(); i < len; ++i, queue.pop())
        {
            if (queue.front() != nullptr)
            {
                level.push_back(queue.front()->val);
                queue.push(queue.front()->left);
                queue.push(queue.front()->right);
            }
        }

        if (!level.empty()) // every time we insert current level to begin of result array so we get a bottom up level order.
        {
            results.insert(results.begin(), level);
        }
    }

    return results;
}

// 108. Convert Sorted Array to Binary Search Tree
TreeNode* sortedArrayToBSTHelper(vector<int>& nums, const int start, const int end)
{
    if (start > end)
    {
        return nullptr;
    }

    const int mid = (start + end) / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = sortedArrayToBSTHelper(nums, start, mid - 1);
    root->right = sortedArrayToBSTHelper(nums, mid + 1, end);
    return root;
}
TreeNode* sortedArrayToBST(vector<int>& nums)
{
    return sortedArrayToBSTHelper(nums, 0, nums.size() - 1);
}

// 109. Convert Sorted List to Binary Search Tree
TreeNode* sortedListToBSTHelper(ListNode* start, ListNode* end) // Note [start .. end) is a right open interval, means end is the node should not be used.
{
    if (start == end)
    {
        return nullptr;
    }

    ListNode* slow = start;
    for (ListNode* fast = start; fast != end && fast->next != end; slow = slow->next, fast = fast->next->next);

    TreeNode* root = new TreeNode(slow->val);
    root->left = sortedListToBSTHelper(start, slow);
    root->right = sortedListToBSTHelper(slow->next, end);
    return root;
}
TreeNode* sortedListToBST(ListNode* head)
{
    return sortedListToBSTHelper(head, nullptr);
}

// 110. Balanced Binary Tree
bool isBalanced(TreeNode* root)
{
    if (root == nullptr)
    {
        return true;
    }

    return isBalanced(root->left) && isBalanced(root->right) && abs(maxDepth(root->left) - maxDepth(root->right)) <= 1;
}

// 111. Minimum Depth of Binary Tree
int minDepth(TreeNode* root)
{
    if (root == nullptr)
    {
        return 0;
    }

    const int leftHeight = minDepth(root->left);
    const int rightHeight = minDepth(root->right);

    // note that the min depth is root to nearest leaf, so if root only has left child or right child, min depth is not 0.
    return 1 + (leftHeight > 0 && rightHeight > 0 ? min(leftHeight, rightHeight) : leftHeight + rightHeight);
}

// 112. Path Sum
bool hasPathSum(TreeNode* root, int sum)
{
    if (root == nullptr)
    {
        return false;
    }

    if (root->left == nullptr && root->right == nullptr)
    {
        return root->val == sum;
    }

    return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
}

// 113. Path Sum II
void dfsSearchPathSum(vector<vector<int>>& paths, vector<int>& path, TreeNode* root, const int sum)
{
    if (root == nullptr)
    {
        return;
    }

    // add current root to path
    path.push_back(root->val);

    // problem says path is from root to leaf, so we only check when encounter a leaf.
    if (sum == root->val && root->left == nullptr && root->right == nullptr)
    {
        paths.push_back(path);

        // remember to pop the current leaf. If not doing it here, we can delete this line and the return statement,
        // since it is a leaf so next 2 recursive calls do nothing, finally pop current root, same effect.
        path.pop_back();
        return;
    }

    dfsSearchPathSum(paths, path, root->left, sum - root->val);
    dfsSearchPathSum(paths, path, root->right, sum - root->val);
    path.pop_back();
}
vector<vector<int>> pathSum(TreeNode* root, int sum)
{
    vector<vector<int>> paths;
    vector<int> path;
    dfsSearchPathSum(paths, path, root, sum);
    return paths;
}

// 114. Flatten Binary Tree to Linked List
void flattenUseStack(TreeNode* root)
{
    TreeNode dummy(0);
    stack<TreeNode*> stack;

    stack.push(root);
    for (TreeNode* last = &dummy; !stack.empty(); )
    {
        TreeNode* current = stack.top();
        stack.pop();

        if (current != nullptr)
        {
            stack.push(current->right);
            stack.push(current->left);

            last->right = current;
            current->left = nullptr; // since we are building single linked list, so only use right as next, set left to null.
            last = current;
        }
    }
}
void flattenUseRecursion(TreeNode* root)
{
    if (root != nullptr)
    {
        flattenUseRecursion(root->left);
        flattenUseRecursion(root->right);

        TreeNode* right = root->right;
        root->right = root->left;
        root->left = nullptr;

        TreeNode* last;
        for (last = root; last->right != nullptr; last = last->right);
        last->right = right;
    }
}
void flattenUseIteration(TreeNode* root)
{
    // As hint said, the linked list is in preorder, so we just need to keep preorder unchanged during conversion.
    for (TreeNode* current = root; current != nullptr; current = current->right)
    {
        //                r                     r
        //              /   \                    \
        //             x     y   =>               x
        //            / \   / \                  / \
        //           z   v a   b                z   v
        //          / \                        / \   \
        //         j   k                      j   k   y
        //                                           / \
        //                                          a   b
        // preorder remains unchanged after conversion : r x z j k v y a b
        // if current has left child tree, make current right child tree become the right child tree of right most leaf node
        // in left child tree, then change left child tree to right child tree, this will not change the preorder.
        if (current->left != nullptr)
        {
            TreeNode* temp;
            for (temp = current->left; temp->right != nullptr; temp = temp->right);
            temp->right = current->right;
            current->right = current->left;
            current->left = nullptr;
        }
    }
}
void flatten(TreeNode* root)
{
    flattenUseIteration(root);
}

// 115. Distinct Subsequences
void dfsSearchDistinct(int& num, string& path, unsigned currentIndex, const string& s, const string& t)
{
    if (path.length() == t.length())
    {
        num += path == t ? 1 : 0;
        return;
    }

    // From currentIndex, let's try to pick up one character one time.
    // We may have a question, for every character in s, there are always two choices, choosing it, or not choosing it,
    // seems in this solution we always choose a character? That is because not choosing a character is implicit here,
    // when currentIndex = currentIndex + 1, that is actually means s[currentIndex] is not chosen.
    // If prefer the instinct solution, please refer to function dfsSearchDistinct2
    for (; currentIndex < s.length(); ++currentIndex)
    {
        path += s[currentIndex];
        dfsSearchDistinct(num, path, currentIndex + 1, s, t);
        path.pop_back();
    }
}
void dfsSearchDistinct2(int& num, string& path, const unsigned currentIndex, const string& s, const string& t)
{
    if (path.length() == t.length())
    {
        num += path == t ? 1 : 0;
        return;
    }

    if (currentIndex < s.length())
    {
        // Not choosing s[currentIndex]
        dfsSearchDistinct2(num, path, currentIndex + 1, s, t);

        // Choosing s[currentIndex].
        path += s[currentIndex];
        dfsSearchDistinct2(num, path, currentIndex + 1, s, t);
        path.pop_back();
    }
}
int numDistinctUseDP(const string& s, const string& t)
{
    // Define dp[i][j] as the count of distinct sub-sequences of s[0..i-1] (first i characters from s) equals
    // to t[0..j-1] (first j characters from t)
    vector<vector<int>> dp(s.length() + 1, vector<int>(t.length() + 1));

    // First column dp[][0] means, when t is empty string, s will only has 1 distinct subsequence equals to t.
    for (unsigned i = 0; i < dp.size(); ++i)
    {
        dp[i][0] = 1;
    }

    // First row dp[0][] means, when s is empty string and t is not empty string, s has 0 distinct subsequence equals to t.
    for (unsigned j = 1; j < dp[0].size(); ++j)
    {
        dp[0][j] = 0;
    }

    for (unsigned i = 1; i < dp.size(); ++i)
    {
        for (unsigned j = 1; j < dp[i].size(); ++j)
        {
            if (s[i - 1] != t[j - 1])
            {
                dp[i][j] = dp[i - 1][j];
            }
            else
            {
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
            }
        }
    }

    return dp.back().back();
}
int numDistinct(const string& s, const string& t)
{
    return numDistinctUseDP(s, t);
}

// 116. Populating Next Right Pointers in Each Node
// 117. Populating Next Right Pointers in Each Node II
void connect(TreeLinkNode* root)
{
    for (TreeLinkNode dummyHead(0); root != nullptr; root = dummyHead.next, dummyHead.next = nullptr)
    {
        for (TreeLinkNode* last = &dummyHead; root != nullptr; root = root->next)
        {
            if (root->left != nullptr)
            {
                last->next = root->left;
                last = last->next;
            }

            if (root->right != nullptr)
            {
                last->next = root->right;
                last = last->next;
            }
        }
    }
}
// Following are the solutions only for 116. 117 is more generic case.
void connectUseRecursionOnlyFor116(TreeLinkNode* root)
{
    if (root == nullptr)
    {
        return;
    }

    // Setup next links between left child tree and right child tree, including left child and right child.
    for (TreeLinkNode* p = root->left, *q = root->right; p != nullptr && q != nullptr; p->next = q, p = p->right, q = q->left);
    connectUseRecursionOnlyFor116(root->left);
    connectUseRecursionOnlyFor116(root->right);
}
// Reuse the next pointer setup in above level is the key here.
// The solution is we go through the left most branch, from root to last non-leaf; in every iteration we go through
// every level, connect all nodes on same level.
void connectUseIterationOnlyFor116(TreeLinkNode* root)
{
    for (; root != nullptr && root->left != nullptr && root->right != nullptr; root = root->left) // From root to last non-leaf
    {
        for (TreeLinkNode* level = root; level != nullptr; level = level->next)
        {
            level->left->next = level->right;
            level->right->next = level->next != nullptr ? level->next->left : nullptr;
        }
    }
}

// 118. Pascal's Triangle
vector<vector<int>> generate(const int numRows)
{
    vector<vector<int>> triangle;
    for (int i = 0; i < numRows; ++i)
    {
        // for triangle[i] it has i+1 elements.
        vector<int> row(i + 1, 1);

        // only calculate row[1 .. i-1], first and last elements are always 1, no need to calculate.
        for (int j = 1; j < i; ++j)
        {
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
        }

        triangle.push_back(row);
    }

    return triangle;
}

// 119. Pascal's Triangle II
vector<int> getRow(const int rowIndex)
{
    vector<int> row(rowIndex + 1, 1); // k-th row has k+1 elements
    for (int i = 1; i <= rowIndex; ++i)
    {
        // Note: calculate from tail to head, so no need to save original value on each index.
        for (int j = i - 1; j >= 1; --j)
        {
            row[j] += row[j - 1];
        }
    }

    return row;
}

// 120. Triangle
int minimumTotalTopDown(vector<vector<int>>& triangle)
{
    vector<int> pathSum(triangle.back().size());
    pathSum[0] = triangle[0][0]; //root to root, min path is root itself.

    // calculate the min path from root to every element on i-th level, store in dp[].
    for (unsigned i = 1; i < triangle.size(); ++i)
    {
        // We calculate from last element on i-th level to the first element, so no need to introduce temporary variable.
        // the last element can only be reached from the last element of above level.
        int j = triangle[i].size() - 1; // Actually we can let j = i, since in triangle, i-th level has i+1 elements, so i == triangle[i].size() - 1, the last element on i-th level is triangle[i][i].
        pathSum[j] = pathSum[j - 1] + triangle[i][j];

        for (j = triangle[i].size() - 2; j > 0; --j) // compare and get min path for triangle[i][i-1] to triangle[i][1] (i-th level elements exclude the first and last)
        {
            pathSum[j] = min(pathSum[j - 1] + triangle[i][j], pathSum[j] + triangle[i][j]);
        }

        // the first element can only be reached from the first element of above level.
        pathSum[0] += triangle[i][0];
    }

    // After we reached the last level, we have computed the min path from root to every element on last level, the smallest one is the
    // minimum path sum of whole triangle.
    return *min_element(pathSum.begin(), pathSum.end());
}
int minimumTotalBottomUp(vector<vector<int>>& triangle)
{
    vector<int> pathSum(triangle.back());
    for (int i = triangle.size() - 2; i >= 0; --i)
    {
        for (unsigned j = 0; j < triangle[i].size(); ++j) // Note triangle[i].size() == i+1
        {
            pathSum[j] = min(pathSum[j], pathSum[j + 1]) + triangle[i][j];
        }
    }

    return pathSum[0];
}
int minimumTotal(vector<vector<int>>& triangle)
{
    return minimumTotalBottomUp(triangle);
}

// 121. Best Time to Buy and Sell Stock
int maxProfit(vector<int>& prices)
{
    int maxProfit = 0;
    int lowestPrice = INT_MAX;
    for (int price : prices)
    {
        // search prices array from begin to end, if we find a price lower than the lowest price we know so far,
        // update the lowest price; if we find a price greater than lowest price, recalculate the profit and if
        // it is greater than max profit so far, update the max profit. 
        // we use the truth that stock must be bought before sell, so only need to care the high price that appears
        // after lowest price.
        if (price < lowestPrice)
        {
            lowestPrice = price;
        }
        else if (price > lowestPrice)
        {
            const int curProfit = price - lowestPrice;
            if (curProfit > maxProfit)
            {
                maxProfit = curProfit;
            }
        }
    }

    return maxProfit;
}
int maxProfitUseDP(vector<int>& prices)
{
    // Define profits[i] = the maximum profit can get if sell stock at prices[i].
    // This array can be optimized to one variable, only need to save previous day's profit.
    vector<int> profits(prices.size(), 0);
    int maxProfit = 0;
    for (unsigned i = 1; i < prices.size(); ++i)
    {
        // For every prices[i] we have two choices, sell or not sell.
        // if sell at prices[i], profits[i] = profits[i - 1] + (prices[i] - prices[i - 1]), note prices[i] - prices[i - 1] can be negative.
        // if profits[i] becomes negative, we should not sell.
        profits[i] = max(profits[i - 1] + prices[i] - prices[i - 1], 0);

        if (profits[i] > maxProfit)
        {
            maxProfit = profits[i];
        }
    }

    return maxProfit;
}

// 122. Best Time to Buy and Sell Stock II
int maxProfit2(vector<int>& prices)
{
    unsigned maxProfit = 0;
    for (unsigned i = 1; i < prices.size(); ++i)
    {
        // as long as today's price is higher than yesterday, add the difference to profit. it is a greedy algorithms.
        if (prices[i] > prices[i - 1])
        {
            maxProfit += prices[i] - prices[i - 1];
        }
    }

    return maxProfit;
}

// 123. Best Time to Buy and Sell Stock III
// Since we must sell then buy again, so it means there is an index i ��[0, prices.size()-1] can splits prices[]
// into 2 ranges: prices[0..i] and prices[i..prices.size()-1], the sum of maximum profits in each range is also
// the maximum (among all possible values of i). So it can be done in 3 steps:
// 1. Scan prices[] from begin to end, calculate and store the maximum profit of prices[0..i] in profits[i].
// 2. Scan prices[] from end to begin, calculate the maximum profit of prices[i..prices.end()-1] and update 
//    profits[i].
// 3. Get the maximum element of profits[].
// In practical, we can combine 2 and 3 in one loop, no need to update the profits[].
int maxProfit3UseTwoPassScan(vector<int>& prices)
{
    vector<int> profits(prices.size(), 0);

    if (prices.size() <= 1)
    {
        return 0;
    }

    // First scan from begin to end, for every i calculate profits[i] = maximum profit of prices[0..i], i ��[0, prices.size()-1].
    // Just reuse problem 121's solution.
    for (int i = 1, lowest = prices[0]; i < prices.size(); ++i)
    {
        if (prices[i] < lowest)
        {
            lowest = prices[i];
        }

        profits[i] = max(prices[i] - lowest, profits[i - 1]);
    }

    // Next scan from end to begin, for every i calculate maximum profit of prices[i..prices.size()-1], and maintain the max profit.
    int maxProfit = profits.back();
    for (int i = prices.size() - 2, highest = prices.back(), previousProfit = 0; i >= 0; --i)
    {
        if (prices[i] > highest)
        {
            highest = prices[i];
        }

        previousProfit = max(highest - prices[i], previousProfit);
        maxProfit = max(maxProfit, profits[i] + previousProfit);
    }

    return maxProfit;
}
int maxProfit4(int k, vector<int>& prices);
int maxProfit3(vector<int>& prices)
{
    return maxProfit4(2, prices);
}

// 124. Binary Tree Maximum Path Sum
// Drawing some examples can show that the max path always has a root, in other words, all nodes on the path
// has a lowest comment ancestor. Given a node p, how to get the max path with p as root? If we know the max
// paths end at its left and right child, we get its max path:
// max path of p = max(p->val,
//                     p->val + max path ends at p->left,
//                     p->val + max path ends at p->right,
//                     p->val + max path ends at p->left + max path ends at p->right)
// But don't forget that p's parent node also  needs to know the max path ends at p, so it can calculate the
// max path with it as root. So we need a helper function, it visits every node and return the max path ends
// at current node, also it maintains the max path so far.
int dfsSearchMaxPath(TreeNode* root, int& maxPath)
{
    if (root == nullptr)
    {
        return 0;
    }

    const int leftMaxPath = dfsSearchMaxPath(root->left, maxPath);
    const int rightMaxPath = dfsSearchMaxPath(root->right, maxPath);

    int curMaxPath = max(root->val, leftMaxPath + root->val);
    curMaxPath = max(curMaxPath, rightMaxPath + root->val);
    curMaxPath = max(curMaxPath, leftMaxPath + rightMaxPath + root->val);
    maxPath = max(maxPath, curMaxPath);

    return max(root->val, root->val + max(leftMaxPath, rightMaxPath)); // don't forget that max path ends at root could be root itself.
}
int maxPathSum(TreeNode* root)
{
    int maxPath = INT_MIN;
    dfsSearchMaxPath(root, maxPath);
    return maxPath;
}

// 125. Valid Palindrome
// No need to care about the non alphanumeric characters in string.
bool isPalindrome(string s)
{
    for (int i = 0, j = s.length() - 1; i <= j; )
    {
        if (!isalnum(s[i]))
        {
            ++i;
        }
        else if (!isalnum(s[j]))
        {
            --j;
        }
        else
        {
            if (tolower(s[i++]) != tolower(s[j--]))
            {
                return false;
            }
        }
    }

    return true;
}

// 126. Word Ladder II
vector<vector<string>> findLaddersUseBFS(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    vector<vector<string>> result;
    unordered_set<string> unusedWords(wordList.begin(), wordList.end()); // use hash set as it supports deleting element by value, makes code simple.
    queue<vector<string>> paths({ {beginWord} });

    // because in nested for loop, we always process all paths belong to one level, so every time when back to outer loop,
    // all paths in queue have the same length, so here paths.front().size() actually means, if current level > minLength,
    // then no need to continue searching.
    for (unsigned minLength = UINT_MAX; !paths.empty() && paths.front().size() < minLength; )
    {
        unordered_set<string> usedWords; // save words that will be selected when processing current level.

        for (unsigned i = 0, queueLength = paths.size(); i < queueLength; ++i, paths.pop()) // process one level of BFS.
        {
            const vector<string>& path = paths.front();
            const string& lastWord = path.back();

            // Finding next word, this is faster than checking every unused word. Assume word length = m, there are n
            // unused words, it needs m * 26 compares, checking every unused word needs (n-1) * m compares, for large
            // test cases this is much faster.
            for (unsigned j = 0; j < lastWord.size(); ++j)
            {
                string nextWord = lastWord;
                for (char ch = 'a'; ch <= 'z'; ++ch)
                {
                    nextWord[j] = ch;
                    if (unusedWords.count(nextWord) == 0)
                    {
                        continue;
                    }

                    usedWords.insert(nextWord);

                    vector<string> nextPath = path;
                    nextPath.push_back(nextWord);
                    if (nextWord == endWord)
                    {
                        result.push_back(nextPath);
                        minLength = nextPath.size();
                    }
                    else
                    {
                        paths.push(nextPath);
                    }
                }
            }
        }

        // After current level has been processed, now it is safe to remove all new used words from unused word list,
        // these words cannot be selected again. Why removing them after whole level has been processed? Because it is
        // possible that one word can be used in multiple paths, e.g.: tax-tex-ted and tax-tad-ted, ted is used in both
        // paths, if removing ted immediately after it is selected into a path, other paths may not be able to use it
        // so can get a incomplete path set.
        for (string usedWord : usedWords)
        {
            unusedWords.erase(usedWord);
        }
    }

    return result;
}
vector<vector<string>> findLaddersUseBidirectionalBFS(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    return { {} };
}
vector<vector<string>> findLadders(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    return findLaddersUseBFS(beginWord, endWord, wordList);
}

// 127. Word Ladder
int ladderLengthUseBFS(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    unordered_set<string> unusedWords(wordList.begin(), wordList.end());
    queue<string> queue({ beginWord });
    for (int level = 1; !queue.empty(); ++level)
    {
        unordered_set<string> usedWords;
        for (unsigned i = 0, queueLength = queue.size(); i < queueLength; ++i, queue.pop())
        {
            string currentWord = queue.front();
            for (unsigned j = 0; j < currentWord.size(); ++j)
            {
                string nextWord = currentWord;
                for (char ch = 'a'; ch != 'z'; ++ch)
                {
                    nextWord[j] = ch;
                    if (unusedWords.count(nextWord) == 0)
                    {
                        continue;
                    }

                    if (nextWord == endWord)
                    {
                        return level + 1;
                    }

                    queue.push(nextWord);
                    usedWords.insert(nextWord);
                }
            }
        }

        for (string usedWord : usedWords)
        {
            unusedWords.erase(usedWord);
        }
    }

    return 0;
}
int ladderLengthUseBidirectionalBFS(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    unordered_set<string> beginUnusedWords(wordList.begin(), wordList.end());
    unordered_set<string> endUnusedWords(wordList.begin(), wordList.end());
    vector<string> beginQueue({ beginWord });
    vector<string> endQueue({ endWord });

    if (beginUnusedWords.count(endWord) == 0)
    {
        return 0;
    }

    for (unsigned level = 1; !beginQueue.empty() && !endQueue.empty(); ++level)
    {
        // always operates the smaller queue.
        vector<string>& queue = beginQueue.size() > endQueue.size() ? endQueue : beginQueue;
        vector<string>& otherQueue = beginQueue.size() > endQueue.size() ? beginQueue : endQueue;
        unordered_set<string>& unusedWords = beginQueue.size() > endQueue.size() ? endUnusedWords : beginUnusedWords;

        unordered_set<string> usedWords;
        for (unsigned i = 0, queueLength = queue.size(); i < queueLength; ++i)
        {
            string currentWord = *(queue.begin());
            queue.erase(queue.begin()); // pop

            for (unsigned j = 0; j < currentWord.size(); ++j)
            {
                string nextWord = currentWord;
                for (char ch = 'a'; ch <= 'z'; ++ch)
                {
                    nextWord[j] = ch;

                    if (nextWord == currentWord || unusedWords.count(nextWord) == 0)
                    {
                        continue;
                    }

                    for (string word : otherQueue)
                    {
                        if (word == nextWord)
                        {
                            return level + 1;
                        }
                    }

                    usedWords.insert(nextWord);
                    queue.push_back(nextWord);
                }
            }
        }

        for (string usedWord : usedWords)
        {
            unusedWords.erase(usedWord);
        }
    }

    return 0;
}
int ladderLength(const string& beginWord, const string& endWord, const vector<string>& wordList)
{
    return ladderLengthUseBidirectionalBFS(beginWord, endWord, wordList);
}

// 128. Longest Consecutive Sequence
int longestConsecutive(vector<int>& nums)
{
    int maxLength = 0;
    unordered_set<int> set(nums.begin(), nums.end());

    for (int num : set)
    {
        if (set.count(num - 1) == 0)
        {
            int count = 0;
            for (int n = num; set.count(n) != 0; ++n, ++count);
            if (count > maxLength)
            {
                maxLength = count;
            }
        }
    }

    return maxLength;
}

// 129. Sum Root to Leaf Numbers
int dfsSum(TreeNode* root, int path)
{
    if (root == nullptr)
    {
        return 0;
    }

    path = path * 10 + root->val;
    if (root->left == nullptr && root->right == nullptr)
    {
        return path;
    }

    return dfsSum(root->left, path) + dfsSum(root->right, path);
}
int sumNumbers(TreeNode* root)
{
    return dfsSum(root, 0);
}

// 130. Surrounded Regions
void dfsNotSurroundedRegion(vector<vector<char>>& board, const unsigned i, const unsigned j)
{
    if (board[i][j] == 'O')
    {
        board[i][j] = '*';

        if (i > 0)
        {
            dfsNotSurroundedRegion(board, i - 1, j);
        }

        if (j > 0)
        {
            dfsNotSurroundedRegion(board, i, j - 1);
        }

        if (j < board[i].size() - 1)
        {
            dfsNotSurroundedRegion(board, i, j + 1);
        }

        if (i < board.size() - 1)
        {
            dfsNotSurroundedRegion(board, i + 1, j);
        }
    }
}
// Instead of start searching from every not captured location that is also not on boundary,
// we search from all not captured locations *on* boundary, mark reachable locations to '*'.
// After all locations have been explored, those locations are still not captured now can be
// claimed as captured.
void solve(vector<vector<char>>& board)
{
    if (!board.empty())
    {
        // first row and last row
        for (unsigned j = 0; j < board[0].size(); ++j)
        {
            dfsNotSurroundedRegion(board, 0, j);
            dfsNotSurroundedRegion(board, board.size() - 1, j);
        }

        // first column and last column.
        for (unsigned i = 0; i < board.size(); ++i)
        {
            dfsNotSurroundedRegion(board, i, 0);
            dfsNotSurroundedRegion(board, i, board[i].size() - 1);
        }

        // Not surrounded region are all '*', all locations have 'O' are surrounded.
        for (unsigned i = 0; i < board.size(); ++i)
        {
            for (unsigned j = 0; j < board[i].size(); ++j)
            {
                if (board[i][j] == 'O')
                {
                    board[i][j] = 'X';
                }

                if (board[i][j] == '*')
                {
                    board[i][j] = 'O';
                }
            }
        }
    }
}

// 131. Palindrome Partitioning
bool isPalindromeString(string s)
{
    unsigned i, j;
    for (i = 0, j = s.length() - 1; i < j && s[i] == s[j]; ++i, --j);
    return i >= j;
}
vector<vector<string>> partitionUseRecursion(string& s)
{
    vector<vector<string>> results;
    for (unsigned i = 0; i < s.length(); ++i)
    {
        string s1 = s.substr(0, i + 1);
        if (isPalindromeString(s1))
        {
            string s2 = s.substr(i + 1, s.length() - i - 1);
            vector<vector<string>> s2partitions = partitionUseRecursion(s2);
            if (s2partitions.empty())
            {
                vector<string> cur(1, s1);
                results.push_back(cur);
            }
            else
            {
                for (vector<string> s2partition : s2partitions)
                {
                    vector<string> cur(1, s1);
                    cur.insert(cur.end(), s2partition.begin(), s2partition.end());
                    results.push_back(cur);
                }
            }
        }
    }

    return results;
}
void dfsPalindrome(const string& s, const vector<vector<bool>>& isPalindrome, vector<vector<string>>& partitions, vector<string>& currentPartition, unsigned start)
{
    if (start == s.length())
    {
        partitions.push_back(currentPartition);
        return;
    }

    for (unsigned i = start; i < s.length(); ++i)
    {
        if (isPalindrome[start][i]) // s[start .. i] is palindrome
        {
            currentPartition.push_back(s.substr(start, i - start + 1)); // s[start .. i]
            dfsPalindrome(s, isPalindrome, partitions, currentPartition, i + 1); // search s[i+1 .. s.length()-1]
            currentPartition.pop_back();
        }
    }
}
vector<vector<string>> partitionUseDFSWithDPOptimization(string& s) // The idea is to prebuild a DP array to speed up DFS search.
{
    // isPalindrome[i][j] means whether s[i..j] is a palindrome or not.
    vector<vector<bool>> isPalindrome(s.length(), vector<bool>(s.length(), false));

    for (int j = 0; j < s.length(); ++j)
    {
        for (int i = j; i >= 0; --i)
        {
            // Only check isPalindrome[i + 1][j - 1] when s[i] is not adjacent to s[j].
            isPalindrome[i][j] = s[i] == s[j] && (j - i <= 1 || isPalindrome[i + 1][j - 1]);
        }
    }

    vector<vector<string>> results;
    vector<string> currentPartition;
    dfsPalindrome(s, isPalindrome, results, currentPartition, 0);
    return results;
}
vector<vector<string>> partition(string& s)
{
    return partitionUseDFSWithDPOptimization(s);
}

// 132. Palindrome Partitioning II
int minCutUseDP(string& s)
{
    // isPalindrome[i][j] means whether s[i..j] is a palindrome or not.
    vector<vector<bool>> isPalindrome(s.length(), vector<bool>(s.length(), false));

    // minCutDP[i] is the minimum cut for substring s[0..i].
    vector<int> minCutDP(s.length(), 0);

    for (int j = 0; j < s.length(); ++j)
    {
        minCutDP[j] = j;

        for (int i = j; i >= 0; --i)
        {
            isPalindrome[i][j] = s[i] == s[j] && (j - i <= 1 || isPalindrome[i + 1][j - 1]);

            // now we know that s[i .. j] is a palindrome
            if (isPalindrome[i][j])
            {
                if (i == 0) // whole s[0..j] is a palindrome, no cut is needed.
                {
                    minCutDP[j] = 0;
                }
                else
                {
                    minCutDP[j] = min(minCutDP[j], minCutDP[i - 1] + 1);
                }
            }
        }
    }

    return minCutDP.back();
}
int minCutUseSpaceOptimizedDP(string& s)
{

}
int minCut(string s)
{
    return minCutUseSpaceOptimizedDP(s);
}

// 138. Copy List with Random Pointer
RandomListNode* copyRandomList(RandomListNode* head)
{
    if (head == nullptr)
    {
        return nullptr;
    }

    // First, insert a duplicate node after each original node.
    // NOTE: No need to set new node's random pointer now.
    for (RandomListNode* p = head; p != nullptr; )
    {
        RandomListNode* q = p->next;
        p->next = new RandomListNode(p->label);
        p->next->next = q;
        p = q;
    }

    // Second, traverse each original node, and set duplicated node's random link.
    // NOTE: random could be null.
    for (RandomListNode* p = head; p != nullptr;)
    {
        RandomListNode* q = p->next;
        q->random = p->random == nullptr ? nullptr : p->random->next;
        p = q->next;
    }

    // Final, decouple the duplicated list from original list.
    RandomListNode* newHead = head->next;
    for (RandomListNode* p = head; p != nullptr;)
    {
        RandomListNode* q = p->next;
        p->next = q->next;
        p = p->next;
        if (p != nullptr)
        {
            q->next = p->next;
        }
    }

    return newHead;
}

// 144. Binary Tree Preorder Traversal
vector<int> preorderTraversalUseStack(TreeNode* root)
{
    vector<int> result;
    stack<TreeNode*> stack;

    for (stack.push(root); !stack.empty();) // note in this solution we allow pushing nullptr to stack.
    {
        root = stack.top();
        stack.pop();

        if (root != nullptr)
        {
            result.push_back(root->val);
            stack.push(root->right); // since we allow nullptr to be pushed to stack so no need to check left and right.
            stack.push(root->left);
        }
    }

    return result;
}
vector<int> preorderTraversal(TreeNode* root)
{
    return preorderTraversalUseStack(root);
}

// 145. Binary Tree Postorder Traversal
vector<int> postorderTraversal(TreeNode* root)
{
    vector<int> postorder;
    if (root == nullptr)
    {
        return postorder;
    }

    stack<TreeNode*> stack;
    stack.push(root);
    TreeNode* lastVisited = root;
    while (!stack.empty())
    {
        while (stack.top()->left != nullptr)
        {
            stack.push(stack.top()->left);
            lastVisited = stack.top()->left;
        }

        postorder.push_back(stack.top()->val);
        stack.pop();

        if (stack.top()->right != nullptr && lastVisited != stack.top()->right)
        {
            stack.push(stack.top()->right);
            lastVisited = stack.top()->right;
        }
        else
        {
            postorder.push_back(stack.top()->val);
            stack.pop();
        }
    }

    return postorder;
}

// 151. Reverse Words in a String
string reverseWordsUseSplitAndExtraSpace(string s)
{
    stack<string> words;

    // split
    for (unsigned i = 0, j, k; i < s.length(); i = k)
    {
        for (j = i; j < s.length() && s[j] == ' '; ++j);
        for (k = j; k < s.length() && s[k] != ' '; ++k);
        if (k > j) // Note: if k == j, substr() returns an empty string.
        {
            words.push(s.substr(j, k - j));
        }
    }

    // join
    for (s.clear(); !words.empty(); words.pop())
    {
        s += words.top() + " ";
    }

    return s.substr(0, s.length() - 1);
}
void reverseSubstring(string& s, int i, int j)
{
    for (; i < j; ++i, --j)
    {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }
}
string reverseWordsInPlace(string s)
{
    reverseSubstring(s, 0, s.length() - 1); // reverse whole string

    for (int i = 0, j, k; i < s.length(); i = k) // reverse all words
    {
        for (j = i; j < s.length() && s[j] == ' '; ++j);
        for (k = j; k < s.length() && s[k] != ' '; ++k);
        reverseSubstring(s, j, k - 1);
    }

    // trim extra whitespace.
    int last = 0;
    for (int probe = 0; probe < s.length(); )
    {
        for (; probe < s.length() && s[probe] == ' '; ++probe);
        for (; probe < s.length() && s[probe] != ' '; s[last++] = s[probe++]);
        for (; probe < s.length() && s[probe] == ' '; ++probe);
        if (probe != s.length())
        {
            s[last++] = ' ';
        }
    }

    return s.substr(0, last);
}
string reverseWords(string s)
{
    return reverseWordsInPlace(s);
}

// 188. Best Time to Buy and Sell Stock IV
// Let profits[k][j] represent the max profit when we only trade from prices[0] to prices[j] (inclusive) using at most k
// transactions.
// Consider prices[j], we may ignore it, thus downgrading to f[k][j - 1], or sell at prices[j]. If we sell at prices[j],
// we have to buy at some price before prices[j], assume buy at prices[i], i∈[0, j); between prices[i..j] no other trade
// can happen (required by the problem statement).
// We earn prices[j] - prices[i] in this trade, and at most k - 1 trades are allowed at and before prices[i], which is 
// profits[k - 1][i], so the max profit of prices[0..j] is:
// profits[k][j] = max(profits[k][j - 1], prices[j] - prices[i] + profits[k - 1][i]), i∈[0, j).
// Initial conditions:
// first row profits[0][] = 0; 0 trade makes 0 profit
// first column profits[][0] = 0; if there is only one price data point no profit no matter how many times you can trade.
int maxProfit4UseDPWithoutOptimization(const int k, vector<int>& prices)
{
    if (prices.size() <= 1 || k == 0)
    {
        return 0;
    }

    // Here profits[][] array is initialized to 0 to simplify initialization code.
    vector<vector<int>> profits(k + 1, vector<int>(prices.size(), 0));

    for (int t = 1; t <= k; ++t)
    {
        for (unsigned j = 1; j < prices.size(); ++j)
        {
            int maxProfit = profits[t][j - 1];
            for (unsigned i = 0; i < j; ++i)
            {
                maxProfit = max(maxProfit, prices[j] - prices[i] + profits[t - 1][i]);
            }

            profits[t][j] = maxProfit;
        }
    }

    return profits.back().back();
}
int maxProfit4WithSpaceOptimization(int k, vector<int>& prices)
{
    if (prices.size() <= 1 || k == 0)
    {
        return 0;
    }

    // When k >= prices.size(), it degenerates to problem 122 (unlimited transactions).
    if (k > static_cast<int>(prices.size()))
    {
        return maxProfit2(prices);
    }

    // Here is the first optimization. From transition function we know that to calculate current
    // state, only need previous state, so we can optimize profits array from k+1 rows to 2 rows.
    vector<vector<int>> profits(2, vector<int>(prices.size(), 0)); // first row is 0
    for (; k > 0; --k) // loop k times not k+1 times since we are starting from k==0 (initialize profits[0]=={0}).
    {
        for (unsigned j = 1; j < prices.size(); ++j)
        {
            int maxProfit = profits[1][j - 1];
            for (unsigned i = 0; i < j; ++i)
            {
                maxProfit = max(maxProfit, prices[j] - prices[i] + profits[0][i]);
            }

            profits[1][j] = maxProfit;
        }

        // copy profits[1][] to profits[0][], make sure profit[0] always contains previous row.
        profits[0] = profits[1];
    }

    return profits[1].back();
}
int maxProfit4(int k, vector<int>& prices)
{
    if (prices.size() <= 1 || k == 0)
    {
        return 0;
    }

    // When k >= prices.size(), it degenerates to problem 122 (unlimited transactions).
    if (k >= prices.size())
    {
        return maxProfit2(prices); // greedy
    }

    vector<vector<int>> profits(2, vector<int>(prices.size(), 0));
    for (; k > 0; --k) // loop k times not k+1 times since we are starting from k==0 (initialize profits[0]=={0}).
    {
        /* Here is the second optimization. To figure out how to optimize the 3rd loop to make it downgrade from O(n^3) to O(n^2),
           let's write some samples:
        j == 1      profits[1][1] = max(prices[1] + (profits[0][0] - prices[0]))
                                                                                      = prices[1] + max(profits[0][0] - prices[0])
        j == 2      profits[1][2] = max(prices[2] + (profits[0][0] - prices[0]),
                                        prices[2] + (profits[0][1] - prices[1]))
                                                                                      = prices[2] + max(profits[0][0] - prices[0],
                                                                                                        profits[0][1] - prices[1])
        j == 3      profits[1][3] = max(prices[3] + (profits[0][0] - prices[0]),
                                        prices[3] + (profits[0][1] - prices[1]),
                                        prices[3] + (profits[0][2] - prices[2]))
                                                                                      = prices[3] + max(profits[0][0] - prices[0],
                                                                                                        profits[0][1] - prices[1],
                                                                                                        profits[0][2] - prices[2])
        As we can see, for every j, we add prices[j] to max(profits[0][i] - prices[i]), i∈[0, j), the purpose of 3rd loop is
        actually trying to find max(profits[0][i] - prices[i]), we can maintain the maximum value of profits[0][i] - prices[i]
        so far, then we don't need the 3rd loop.
        */
        for (int j = 1, innerMax = profits[0][0] - prices[0]; j < prices.size(); ++j)
        {
            profits[1][j] = max(profits[1][j - 1], prices[j] + innerMax);
            innerMax = max(innerMax, profits[0][j] - prices[j]);
        }

        // copy profits[1][] to profits[0][], make sure profit[0] always contains previous state.
        profits[0] = profits[1];
    }

    return profits[1].back();
}

// 200. Number of Islands
int dfsCheckIsland(vector<vector<char>>& grid, vector<vector<bool>>& visited, unsigned i, unsigned j)
{
    if (i < 0 || i >= grid.size() || j < 0 || j >= grid[i].size() ||
        grid[i][j] != '1' || visited[i][j])
    {
        return 0;
    }

    visited[i][j] = true;

    // top
    dfsCheckIsland(grid, visited, i - 1, j);

    // left
    dfsCheckIsland(grid, visited, i, j - 1);

    // bottom
    dfsCheckIsland(grid, visited, i + 1, j);

    // right
    dfsCheckIsland(grid, visited, i, j + 1);

    return 1;
}
int numIslands(vector<vector<char>>& grid)
{
    if (grid.size() == 0)
    {
        return 0;
    }

    int result = 0;
    vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
    
    for (unsigned i = 0; i < grid.size(); ++i)
    {
        for (unsigned j = 0; j < grid[i].size(); ++j)
        {
            result += dfsCheckIsland(grid, visited, i, j);
        }
    }

    return result;
}

// 202. Happy Number
bool isHappy(int n)
{
    for (int next; n != 1 && n != 4; n = next) // NOTE: when a number is unhappy, loop will contains 4.
    {
        for (next = 0; n > 0; n /= 10)
        {
            next += (n % 10) * (n % 10);
        }
    }

    return n == 1;
}

// 206. Reverse Linked List
ListNode* reverseList(ListNode* head)
{
    if (head == nullptr)
    {
        return head;
    }

    ListNode* p, * q;
    for (p = head, q = head->next; q != nullptr;)
    {
        ListNode* k = q->next;
        q->next = p;
        if (p == head)
        {
            p->next = nullptr;
        }

        p = q;
        q = k;
    }

    return p;
}

// 226. Invert Binary Tree
TreeNode* invertTree(TreeNode* root)
{
    if (root == nullptr)
    {
        return root;
    }

    invertTree(root->left);
    invertTree(root->right);

    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;

    return root;
}

//235. Lowest Common Ancestor of a Binary Search Tree
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    for (; root->val > p->val&& root->val > q->val || root->val < p->val && root->val < q->val; root = p->val < root->val ? root->left : root->right);
    return root;
}

// 297. Serialize and Deserialize Binary Tree
void getBTPreOrderHelper(TreeNode* root, ostringstream& os)
{
    if (root == nullptr)
    {
        os << "#" << " ";
        return;
    }

    os << root->val << " ";
    getBTPreOrderHelper(root->left, os);
    getBTPreOrderHelper(root->right, os);
}
string serializeBTUsePreOrder(TreeNode* root)
{
    ostringstream os;
    getBTPreOrderHelper(root, os);
    return os.str();
}
TreeNode* buildBTFromPreOrderHelper(istringstream& is)
{
    string val;
    if (!(is >> val) || val == "#")
    {
        return nullptr;
    }

    TreeNode* root = new TreeNode(stoi(val));
    root->left = buildBTFromPreOrderHelper(is);
    root->right = buildBTFromPreOrderHelper(is);

    return root;
}
TreeNode* deserializeBTUsePreOrder(string data)
{
    istringstream is(data);
    return buildBTFromPreOrderHelper(is);
}
string serializeBTUseLevelOrder(TreeNode* root) // Encodes a tree to a single string.
{
    ostringstream os;
    queue<TreeNode*> queue;
    for (queue.push(root); !queue.empty();)
    {
        for (unsigned i = 0, queueLength = queue.size(); i < queueLength; ++i, queue.pop())
        {
            TreeNode* current = queue.front();
            if (current != nullptr)
            {
                os << current->val << " ";
                queue.push(current->left);
                queue.push(current->right);
            }
            else
            {
                os << "#" << " ";
            }
        }
    }

    return os.str();
}
TreeNode* deserializeBTFromLevelOrder(string data) // Decodes your encoded data to tree.
{
    istringstream is(data);
    vector<string> nums;
    for (string num; is >> num; nums.push_back(num));

    TreeNode* root = nullptr;
    if (!nums.empty() && nums[0] != "#")
    {
        queue<TreeNode*> queue;
        root = new TreeNode(stoi(nums[0]));
        queue.push(root);

        for (unsigned i = 1; i < nums.size() - 1 && !queue.empty(); i += 2, queue.pop())
        {
            TreeNode* current = queue.front();
            if (nums[i] == "#")
            {
                current->left = nullptr;
            }
            else
            {
                current->left = new TreeNode(stoi(nums[i]));
                queue.push(current->left);
            }

            if (nums[i + 1] == "#")
            {
                current->right = nullptr;
            }
            else
            {
                current->right = new TreeNode(stoi(nums[i + 1]));
                queue.push(current->right);
            }
        }
    }

    return root;
}
string serializeBT(TreeNode* root)
{
    return serializeBTUseLevelOrder(root);
}
TreeNode* deserializeBT(string data)
{
    return deserializeBTFromLevelOrder(data);
}

// 334. Increasing Triplet Subsequence
bool increasingTriplet(vector<int>& nums)
{
    int smallest = INT_MAX;
    int secondSmallest = INT_MAX;
    for (int num : nums)
    {
        if (num <= smallest)
        {
            smallest = num;
        }
        else if (num <= secondSmallest)
        {
            secondSmallest = num;
        }
        else
        {
            return true;
        }
    }

    return false;
}

// 414. Third Maximum Number
int thirdMax(vector<int>& nums)
{
    vector<long long> maxes(3, LLONG_MIN); // 0: max, 1: 2ndmax, 3: 3rdmax
    for (int num : nums)
    {
        if (num > maxes[0])
        {
            maxes[2] = maxes[1];
            maxes[1] = maxes[0];
            maxes[0] = num;
        }
        else if (num != maxes[0] && num > maxes[1])
        {
            maxes[2] = maxes[1];
            maxes[1] = num;
        }
        else if (num != maxes[0] && num != maxes[1] && num > maxes[2])
        {
            maxes[2] = num;
        }
    }

    return maxes[2] != LLONG_MIN ? maxes[2] : maxes[0];
}

// 415. Add Strings
string addStrings(const string& num1, const string& num2)
{
    string sum;
    for (int i = num1.length() - 1, j = num2.length() - 1, r, carry = 0; i >= 0 || j >= 0 || carry != 0; --i, --j, carry = r / 10)
    {
        r = (i >= 0 ? num1[i] - '0' : 0) + (j >= 0 ? num2[j] - '0' : 0) + carry;
        sum.insert(0, 1, static_cast<char>(r % 10 + '0'));
    }

    return sum;
}

// 449. Serialize and Deserialize BST
void getBSTPreOrderHelper(TreeNode* root, ostringstream& os) // Encodes a tree to a single string.
{
    if (root == nullptr)
    {
        return;
    }

    os << root->val << " ";
    getBSTPreOrderHelper(root->left, os);
    getBSTPreOrderHelper(root->right, os);
}
string serializeBST(TreeNode* root)
{
    ostringstream os;
    getBSTPreOrderHelper(root, os);
    return os.str();
}
TreeNode* buildBSTFromPreOrderHelper(vector<int>& nums, const int start, const int end)
{
    if (start > end)
    {
        return nullptr;
    }

    int pivot;
    for (pivot = start; pivot <= end && nums[pivot] <= nums[start]; ++pivot);

    TreeNode* root = new TreeNode(nums[start]);
    root->left = buildBSTFromPreOrderHelper(nums, start + 1, pivot - 1);
    root->right = buildBSTFromPreOrderHelper(nums, pivot, end);

    return root;
}
TreeNode* deserializeBST(string data) // Decodes your encoded data to tree.
{
    vector<int> nums;
    istringstream is(data);
    for (string num; is >> num; nums.push_back(stoi(num)));
    return buildBSTFromPreOrderHelper(nums, 0, nums.size() - 1); // No need to check if string is empty as nums.size()-1 = -1 < 0.
}

// 674. Longest Continuous Increasing Subsequence
int findLengthOfLCIS(vector<int>& nums)
{
    int maxIncreasingLength = 0;
    int currentIncreasingLength = 0;
    for (unsigned i = 0; i < nums.size(); ++i)
    {
        if (i == 0 || nums[i] > nums[i - 1])
        {
            if (++currentIncreasingLength > maxIncreasingLength)
            {
                maxIncreasingLength = currentIncreasingLength;
            }
        }
        else
        {
            currentIncreasingLength = 1;
        }
    }

    return maxIncreasingLength;
}

// 695. Max Area of Island
// NOTE: both DFS and BFS solution are slow because of using additional data structure to maintain coordinates that have been
// visited. To speed up, can directly change visited grid value from 1 to 0 or 2.
bool checkLand(const int i, const int j, const vector<vector<int>>& grid, vector<vector<bool>>& visited, queue<pair<int, int>>& queue)
{
    if (i >= 0 && i < grid.size() && j >= 0 && j < grid[i].size() && grid[i][j] == 1 && !visited[i][j])
    {
        visited[i][j] = true;
        queue.push(make_pair(i, j));

        return true;
    }

    return false;
}
int maxAreaOfIslandUseBFS(const vector<vector<int>>& grid)
{
    int maxArea = 0;
    vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
    queue<pair<int, int>> queue;
    for (unsigned i = 0; i < grid.size(); ++i)
    {
        for (unsigned j = 0; j < grid[i].size(); ++j)
        {
            if (checkLand(i, j, grid, visited, queue)) // Note checkLand will push coordinate (i,j) to queue if it is an unexplored land
            {
                int area = 0; // We are going to counting size when popping, if initialize size to 1, the first element will be counted twice.
                while (!queue.empty()) // Start BFS from grid[i][j]
                {
                    for (unsigned k = 0, len = queue.size(); k < len; ++k, ++area, queue.pop())
                    {
                        checkLand(queue.front().first - 1, queue.front().second, grid, visited, queue); // top
                        checkLand(queue.front().first + 1, queue.front().second, grid, visited, queue); // bottom
                        checkLand(queue.front().first, queue.front().second - 1, grid, visited, queue); // left
                        checkLand(queue.front().first, queue.front().second + 1, grid, visited, queue); // right
                    }
                }

                maxArea = max(maxArea, area);
            }
        }
    }

    return maxArea;
}
void areaOfIsland(const vector<vector<int>>& grid, vector<vector<bool>>& visited, const int i, const int j, int& area)
{
    if (i >= 0 && i < grid.size() && j >= 0 && j < grid[i].size() &&
        grid[i][j] == 1 && !visited[i][j])
    {
        visited[i][j] = true;
        ++area;

        areaOfIsland(grid, visited, i - 1, j, area); // top
        areaOfIsland(grid, visited, i + 1, j, area); // bottom
        areaOfIsland(grid, visited, i, j - 1, area); // left
        areaOfIsland(grid, visited, i, j + 1, area); // right
    }
}
int maxAreaOfIslandUseDFS(const vector<vector<int>>& grid)
{
    int maxArea = 0;
    vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
    for (unsigned i = 0; i < grid.size(); ++i)
    {
        for (unsigned j = 0; j < grid[i].size(); ++j)
        {
            int area = 0;
            areaOfIsland(grid, visited, i, j, area);
            maxArea = max(area, maxArea);
        }
    }

    return maxArea;
}
int maxAreaOfIsland(const vector<vector<int>>& grid)
{
    return maxAreaOfIslandUseDFS(grid);
}

// 973. K Closest Points to Origin
// Max heap is not the most efficient solution, but it can be used in streaming calculation as it does not have to know the size
// of the data previously.
vector<vector<int>> kClosestUseMaxHeap(const vector<vector<int>>& points, const int k)
{
    // can also use priority_queue here, declare as:
    // priority_queue<vector<int>, vector<vector<int>>, decltype(distanceComparer)> pq(distanceComparer);
    // but priority_queue cannot be converted into vector, has to pop all elements out to a vector.

    vector<vector<int>> maxHeap;
    const function<bool(const vector<int>&, const vector<int>&)> distanceComparer = [](const vector<int>& a, const vector<int>& b)
    {
        return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1];
    };

    for (const vector<int>& point : points)
    {
        if (maxHeap.size() < k || distanceComparer(point, maxHeap.front())) // true means point is closer than heap top
        {
            maxHeap.push_back(point);
            push_heap(maxHeap.begin(), maxHeap.end(), distanceComparer); // adjust

            if (maxHeap.size() > k)
            {
                pop_heap(maxHeap.begin(), maxHeap.end(), distanceComparer);
                maxHeap.pop_back();
            }
        }
    }

    return maxHeap;
}
// Similar to quick sort, we can choose a pivot element and partition vector into 2 ranges, all elements in left range are 
// smaller than right range, if left range's size == k, then we found the answer.
vector<vector<int>> kClosestUsePartitioning(vector<vector<int>>& points, const int k)
{
    const function<bool(const vector<int>&, const vector<int>&)> distanceComparer = [](const vector<int>& a, const vector<int>& b)
    {
        return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1];
    };

    for (int start = 0, end = points.size() - 1, left = -1; start <= end && left + 1 != k; )
    {
        left = start - 1;
        for (int i = start; i < end; ++i) // Need to protect start, so use a local variable i.
        {
            if (distanceComparer(points[i], points[end])) // use points[end] as pivot, so loop is start < end.
            {
                swap(points[++left], points[i]);
            }
        }

        swap(points[end], points[++left]); // place the pivot element on correct index.

        if (left + 1 < k) // go right range
        {
            start = left + 1;
        }
        else if (left + 1 > k) //go left range
        {
            end = left - 1;
        }
    }

    return vector<vector<int>>(points.begin(), points.begin() + k);
}
vector<vector<int>> kClosest(vector<vector<int>>& points, const int k)
{
    return kClosestUsePartitioning(points, k);
}
