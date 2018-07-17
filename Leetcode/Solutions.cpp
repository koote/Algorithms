#include <stack>
#include <queue>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include "DataStructure.h"

using namespace std;

// 1. Two Sum
vector<int> twoSum(vector<int>& nums, int target)
{
    vector<int> result;
    unordered_map<int, int> map; // <value, original_index>
    unordered_map<int, int>::iterator iterator;

    for (int i = 0; i < nums.size(); ++i)
    {
        int new_target = target - nums[i];
        if ((iterator = map.find(new_target)) != map.end())
        {
            result.push_back(i);
            result.push_back(iterator->second);

            return result;
        }

        map.insert(std::make_pair(nums[i], i)); // unordered_map search first element of pair, so have to make pair in this way.
    }

    return result;
}

// 2. Add Two Numbers
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
{
    ListNode* a = l1;
    ListNode* b = l2;
    ListNode* dummy = new ListNode(-1);
    ListNode* c = dummy;
    int carry = 0;

    while (a != nullptr || b != nullptr)
    {
        int val = (a == nullptr ? 0 : a->val) + (b == nullptr ? 0 : b->val) + carry;
        carry = val / 10;
        val %= 10;

        c->next = new ListNode(val);
        c = c->next;
        a = a == nullptr ? a : a->next;
        b = b == nullptr ? b : b->next;
    }

    if (carry > 0) // don't forget the carry
    {
        c->next = new ListNode(carry);
        c = c->next;
    }

    c->next = nullptr; //close the result list
    ListNode* result = dummy->next;
    delete dummy;
    return result;
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
double findKth(int a[], int m, int b[], int n, int k)
{
    if (m < n) { return findKth(b, n, a, m, k); }
    if (n == 0) { return a[k - 1]; }
    if (k == 1) { return min(a[0], b[0]); }
    if (k == m + n) { return max(a[m - 1], b[n - 1]); }

    // let k = i+j. Please note that we are sure n <= m here
    int j = min(n, k / 2);
    const int i = k - j;

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
        // When a[i-1] == b[j-1], they have k-2 elements left and m+n-k elements right, so either is the kth.
        return a[i - 1];
    }
}
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
    const int m = nums1.size();
    const int n = nums2.size();

    if ((m + n) & 1) // odd
    {
        return findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2 + 1);
    }
    else //even
    {
        return (findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2) + findKth(nums1.data(), m, nums2.data(), n, (m + n) / 2 + 1)) / 2.0;
    }
}

// 5. Longest Palindromic Substring
string searchPalindrome(string s, int left, int right)
{
    // when loop ends, no matter it is because of out of bounds or s[left] and s[rigth] no longer equal, current palindrome is always s[left+1 .. right-1].
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
string zigzagConvert(string s, int numRows)
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
        int digit = x % 10;
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
    // but not vice versa, because when text is empty, if pattern is a* or .*, they still match.
    if (pattern.length() == 0)
    {
        return text.length() == 0;
    }

    // If there is a * followed, there are 2 cases:
    // (1) the preceding character of * appears zero times, next match would be isRegexMatch(text, pattern.substr(2)).
    // (2) the preceding character of * appears >= 1 times, next match would be isRegexMatch(text.substr(1), pattern.substr(1)) if current character matches pattern.
    return (pattern.length() > 1 && pattern[1] == '*') ?
        isMatch_Regex(text, pattern.substr(2)) || (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch_Regex(text.substr(1), pattern) :
        (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch_Regex(text.substr(1), pattern.substr(1));
}

// 11. Container With Most Water
int maxArea(vector<int>& height)
{
    int max = 0;
    for (int i = 0, j = height.size() - 1; i < j;)
    {
        int current;
        if (height[i] < height[j])
        {
            current = (j - i) * height[i];
            ++i;
        }
        else
        {
            current = (j - i)* height[j];
            --j;
        }

        if (current > max)
        {
            max = current;
        }
    }

    return max;
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
    if (strs.empty())
    {
        return "";
    }

    int i;
    bool flag = true;
    for (i = 0; flag; ++i)
    {
        for (int j = 0; j < strs.size() && flag; ++j)
        {
            flag = i < strs[j].length() && strs[j][i] == strs[0][i];
        }
    }

    return strs[0].substr(0, i - 1);
}

// 15. 3Sum
vector<vector<int>> kSum(vector<int>& nums, unsigned int k, int sum) // nums must be sorted.
{
    vector<vector<int>> result(0);

    if (nums.size() < k || k == 0)
    {
        return result;
    }

    if (k == 2)
    {
        for (size_t i = 0, j = nums.size() - 1; i < j;)
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

    for (int i = 0; i < static_cast<int>(nums.size()) - k + 1;)
    {
        /*
        // Because nums are sorted, so nums[i+1..i+k-1] are smallest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[i+1] + ... + nums[i+k-1]) > sum, we can give up, because
        // we cannot find answer, that sum is the smallest we can produce.
        int s = nums[i];
        for (size_t j = i + 1, p = k - 1; j < nums.size() && p > 0; s += nums[j++], --p);
        if (s > sum)
        {
            for (++i; i < nums.size() && nums[i - 1] == nums[i]; ++i);
            break;
        }

        // Same reason, nums[size-k+1..size-1] are largest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[size-k+1] + ... + nums[size-1]) < sum, we can ignore current loop (jump over nums[i])
        // because that sum is the largest we can produce in current loop.
        s = nums[i];
        for (size_t j = nums.size() - 1, p = k - 1; j > 0 && p > 0; s += nums[j--], --p);
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
    int diff = INT_MAX; // Don't intialize sum to INT_MAX then calc difference = abs(sum - target), that could overflow.
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
vector<string> letterCombinationsTopDown(string digits)
{
    const string mapping[9] = { "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    vector<string> result;
    if (digits.length() >= 1)
    {
        vector<string> rest = letterCombinationsTopDown(digits.substr(1));
        string first = mapping[digits[0] - '0' - 1];
        for (int i = 0; i < first.length(); ++i)
        {
            string s = first.substr(i, 1);
            if (!rest.empty())
            {
                for (int j = 0; j < rest.size(); ++j)
                {
                    result.push_back(s + rest[i]);
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
vector<string> letterCombinationsBottomUp(string digits)
{
    const string mapping[9] = { "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    vector<vector<string>> dp(digits.length() + 1);
    for (int i = digits.length() - 1; i >= 0; --i)
    {
        string current = mapping[digits[i] - '0' - 1];
        if (dp[i + 1].empty())
        {
            for (int j = 0; j < current.length(); ++j)
            {
                dp[i].push_back(current.substr(j, 1));
            }
        }
        else
        {
            for (int j = 0; j < current.length(); ++j)
            {
                string s = current.substr(j, 1);
                for (int k = 0; k < dp[i + 1].size(); ++k)
                {
                    dp[i].push_back(s + dp[i + 1][k]);
                }
            }
        }
    }

    return dp[0];
}
vector<string> letterCombinations(string digits)
{
    vector<string> r1 = letterCombinationsTopDown(digits);
    vector<string> r2 = letterCombinationsBottomUp(digits);
    return r1;
}

// 18. 4Sum
vector<vector<int>> fourSum(vector<int>& nums, int target)
{
    sort(nums.begin(), nums.end());
    return kSum(nums, 4, target);
}

// 19. Remove Nth Node From End of List
ListNode* removeNthFromEnd(ListNode* head, int n)
{
    ListNode dummy(-1);
    dummy.next = head;

    // The algorithms is to use two pointers and let distance between first and last is n+1, when first reaches 
    // end of list, last->next is the one needs to be deleted.
    // We start counting from dummy other than real head, this can solve the condition that list length is 1 and
    // n == 1, pointer first stops at real head, pointer last points to dummy head, no special handling is needed.
    // If we start counting from real head, then when first loop ends, first is null.
    ListNode* first;
    for (first = &dummy; n > 0 && first != nullptr; first = first->next, --n);

    ListNode* last;
    for (last = &dummy; first->next != nullptr; first = first->next, last = last->next);

    // last->next is what needs to be removed.
    ListNode* target = last->next;
    last->next = target->next;
    delete target;

    return dummy.next;
}

// 20. Valid Parentheses
bool isValid(string s)
{
    stack<char> stk;
    for (char i : s)
    {
        if (s[i] == '{' || s[i] == '[' || s[i] == '(')
        {
            stk.push(s[i]);
        }
        else
        {
            if (!stk.empty())
            {
                const char top = stk.top();
                if ((s[i] == ')' && top == '(') || (s[i] == ']'&&top == '[') || (s[i] == '}' && top == '{'))
                {
                    stk.pop();
                    continue;
                }
            }

            return false;
        }
    }

    return stk.empty();
}

// 21. Merge Two Sorted Lists
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2)
{
    ListNode dummy(-1);
    ListNode* last = &dummy;
    while (l1 != nullptr || l2 != nullptr)
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

    return dummy.next;
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
            // For every string in dp[j] and dp[i-j-1], concat a new string (dp[j])+dp[i-j-1]
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

    // Solution space is a binary tree. But this problem is a little different from other similiar problems (e.g. letterCombinations),
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
vector<string> generateParenthesis(int n)
{
    vector<string> r1 = generateParenthesisBacktracking("", n, n);
    vector<string> r2 = generateParenthesisBottomUp(n);
    assert(r1 == r2);
    return r2;
}

// 23. Merge k Sorted Lists
void minHeapify(vector<ListNode*>& nodes, int root)
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
    ListNode dummy(0);
    ListNode* last = &dummy;
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

    last->next = nullptr;
    return dummy.next;
}

// 24. Swap Nodes in Pairs
ListNode* swapPairs(ListNode* head)
{
    ListNode dummy(0);
    dummy.next = head;

    // What will be swaped are p->next and q->next.
    // check q is null is only for empty list. check q->next is null is really needed for non empty list.
    for (ListNode* p = &dummy, *q = head; q != nullptr && q->next != nullptr;)
    {
        p->next = q->next;
        p = p->next; // Moving p forward makes code looks a little bit clean, otherwise it would be q->next = p->next->next;p->next->next=q;
        q->next = p->next;
        p->next = q;

        p = q;
        q = p->next;
    }

    return dummy.next;
}

//25. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, int k)
{
    ListNode dummy(0);
    dummy.next = head;

    for (ListNode* p = &dummy, *q = head; q != nullptr;)
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

            // reuse q and p for linked list reversing. p is q's succeedor
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

    return dummy.next;
}

// 26. Remove Duplicates from Sorted Array
int removeDuplicates(vector<int>& nums)
{
    int last = -1;
    for (size_t probe = 0; probe < nums.size(); nums.at(++last) = nums.at(probe++))
    {
        for (; probe + 1 < nums.size() && nums.at(probe + 1) == nums.at(probe); ++probe);
    }

    return last + 1;
}

// 27. Remove Element
int removeElement(vector<int>& nums, int val)
{
    size_t last = 0;
    for (size_t probe = 0; probe < nums.size(); ++probe)
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
    for (size_t i = 0, j; text.length() >= pattern.length() && i <= text.length() - pattern.length(); ++i)
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
int divide2(int dividend, int divisor)
{
    unsigned int quotient = 0;

    for (unsigned int a = dividend < 0 ? -dividend : dividend, b = divisor < 0 ? -divisor : divisor, k; a >= b; a -= b << k)
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
    unsigned int quotient = 0;
    for (unsigned int a = dividend < 0 ? -dividend : dividend, b = divisor < 0 ? -divisor : divisor, k, temp; a >= b; a -= temp, quotient += k)
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

    const size_t patternLen = words[0].length() * words.size();

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

    for (size_t i = 0; i <= s.length() - patternLen; ++i)
    {
        unordered_map<string, int> unusedWords(wordCount);

        // Check if all words present in current substring s[i + j..i + j + patternLen - 1]
        // Slice it to words.size() slices, for each slice, use the unusedWords
        // hash table to quick check if it is a word, if yes, decrease value in
        // hash table, means it has been used.
        for (size_t j = 0; j < patternLen; j += words[0].length())
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
    size_t i;
    for (i = nums.size() - 1; i >= 1 && nums[i] <= nums[i - 1]; --i);

    // When previous loop ends, it could be (1) i == 0, or (2) i > 0, it doesn't matter, sort this segment first.
    // Please note that starts from i, all elements are in descending order, so sorting is straightforward, just reverse them.
    for (size_t j = i, k = nums.size() - 1; j < k; ++j, --k)
    {
        const int temp = nums[j];
        nums[j] = nums[k];
        nums[k] = temp;
    }

    // Now lets handle if we really find an i > 0, if yes, swap nums[i-1] with the smallest element that greater than nums[i-1].
    if (i > 0)
    {
        size_t l;
        for (l = i; l < nums.size() && nums[l] < nums[i - 1]; ++l);

        const int temp = nums[l];
        nums[i - 1] = nums[l];
        nums[l] = temp;
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
    for (size_t i = 1; i < s.length(); ++i)
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
int search(vector<int>& nums, int target)
{
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right)
    {
        const size_t mid = (left + right) / 2;
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
int search2(vector<int>& nums, const int target)
{
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right)
    {
        const int middle = (left + right) / 2;
        if (nums[middle] == target)
        {
            return middle;
        }

        if (target > nums[middle]) // Have 2 cases
        {
            // When middle is on right part and target is not in range nums[middle .. right]
            if (nums[right] >= nums[middle] && target > nums[right])
            {
                right = middle - 1;
            }
            else // for other 2 cases (1) middle is on left part, (2) middle is on right part and target is in range nums[middle ... right]
            {
                left = middle + 1;
            }
        }
        else // nums[middle] > target
        {
            // When middle is on left part and target is not in range nums[left .. middle]
            if (nums[left] <= nums[middle] && target < nums[left])
            {
                left = middle + 1;
            }
            else // for other 2 cases (1) middle is on the right part, (2) middle is on left part and target is in range nums[left .. middle]
            {
                right = middle - 1;
            }
        }
    }

    return -1;
}

// 34. Search for a Range
vector<int> searchRange(vector<int>& nums, int target)
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
int searchInsert(vector<int>& nums, int target)
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
    for (size_t i = 0; i < board.size(); ++i)
    {
        for (size_t j = 0; j < board[0].size(); ++j)
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
    for (size_t i = 0; i < board.size(); ++i)
    {
        for (size_t j = 0; j < board[0].size(); ++j)
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

bool isValidCandidate(const vector<vector<char>>& board, size_t row, size_t col)
{
    for (size_t r = 0; r < board.size(); ++r)
    {
        if (r != row && board[r][col] == board[row][col])
        {
            return false;
        }
    }

    for (size_t c = 0; c < board[0].size(); ++c)
    {
        if (c != col && board[row][c] == board[row][col])
        {
            return false;
        }
    }

    for (size_t r = 3 * (row / 3); r < 3 * (row / 3 + 1); ++r)
    {
        for (size_t c = 3 * (col / 3); c < 3 * (col / 3 + 1); ++c)
        {
            if (r != row && c != col && board[r][c] == board[row][col])
            {
                return false;
            }
        }
    }

    return true;
}
bool dfsSearchSudoku(vector<vector<char>>& board, size_t row, size_t col)
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
    dfsSearchSudoku(board, 0, 0);
}

// 38. Count and Say
string countAndSay(int n)
{
    string result = "1";
    for (; n > 1; --n)
    {
        string temp;
        for (size_t i = 0, j; i < result.length(); i = j)
        {
            for (j = i; j < result.length() && result[j] == result[i]; ++j);
            temp += string("") + static_cast<char>(j - i + '0') + result[i];
        }

        result = temp;
    }

    return result;
}

// 39. Combination Sum
void dfsSearchCombinationSumSolution(vector<int>& candidates, size_t currentIndex, int target, vector<int>& path, vector<vector<int>>& solutions)
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
    dfsSearchCombinationSumSolution(candidates, currentIndex, target - candidates[currentIndex], path, solutions);
    path.pop_back();
    dfsSearchCombinationSumSolution(candidates, currentIndex + 1, target, path, solutions);
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target)
{
    vector<vector<int>> solutions;
    vector<int> path;
    dfsSearchCombinationSumSolution(candidates, 0, target, path, solutions);
    return solutions;
}

// 40. Combination Sum II
void dfsSearchCombinationSum2Solution(vector<int>& candidates, size_t currentIndex, int target, vector<int>& path, vector<vector<int>>& solutions)
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
    dfsSearchCombinationSum2Solution(candidates, currentIndex + 1, target - candidates[currentIndex], path, solutions);
    path.pop_back();

    // Because in last call, we choose candidates[currentIndex] and may already found a solution. So if 
    // candidates[currentIndex+1] == candidates[currentIndex], need to skip, otherwise will get duplicate solutions.
    for (; currentIndex < candidates.size() - 1 && candidates[currentIndex + 1] == candidates[currentIndex]; ++currentIndex);
    dfsSearchCombinationSum2Solution(candidates, currentIndex + 1, target, path, solutions);
}
vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> solutions;
    vector<int> path;
    dfsSearchCombinationSum2Solution(candidates, 0, target, path, solutions);
    return solutions;
}

// 41. First Missing Positive
int firstMissingPositive(vector<int>& nums)
{
    // For each positive numbers nums[i] that greater than 0 and smaller than nums.size(), put it on nums[nums[i]-1].
    // For numbers not in that range, can be left on their current position.
    for (size_t i = 0; i < nums.size();)
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
    for (size_t i = 0; i < nums.size(); ++i)
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
// For this problem, DO NOT try to start thinking each hollow, that makes this problem looks complicated. Thinking 
// each bar instead. Let's say every bar can trap some water (physically it is impossble), if we can get how much  
// water each bar can trap, and add them together, we get the toal trapped water.
// Given a bar i, whose height is height[i], how much water it can trap? Drawing a chart 
// helps to show that:
// water bar i can trap = min(height of the highest bar on bar i's left, height of the highest bar on bar i's right) - height[i]
//            ___
//            | |               ___
//            | |      ___      | |
//            | |      | |      | |
//     _______|_|______|_|______|_|____________________
//                      i
// So scan height array from left to right, for each bar, scan its left highest bar and right highest bar.
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
// In brute force solution, for each bar we need to search its left highest bar and right highest bar, and how much water
// it can trap is determined by the lower bar between its left highest bar and right highest bar, so time complexity is 
// O(n2).
// Let's think in a reversed way, given two bars i and j (i <= j), we compare their heights, without loss of generality,
// let's say lower one is the left end i, we search its right side, before encounter a bar k has height[k] > height[i], 
// for all bars on its right who are not higher than height[i], water they can trap is height[i] - height[m] (i < m < k).
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
        if (height[left] < height[right])
        {
            for (probe = left; probe <= right && height[probe] <= height[left]; ++probe)
            {
                total += height[left] - height[probe];
            }

            left = probe;
        }
        else
        {
            for (probe = right; probe >= left && height[probe] <= height[right]; --probe)
            {
                total += height[right] - height[probe];
            }

            right = probe;
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
string addStrings(string num1, string num2);
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

// 44. Wildcard Matching
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
Let's use two * as example, assuming pattern[0 .. j] matches text[i-j .. i], now encounter a '*' at pattern[j], in recursive solution, a '*' means
branching, because a '*' can be empty, or 1 character, 2 characters, etc., so algorithm needs to try all possible cases, like this:
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
pattern[j] is totally useless, so we can get the conlusion: everytime when mismatch happens, only need to roll back to nearest '*', if we tried
all possible branches of nearest '*' but cannot match, we can say the whole pattern won't match.
 */
bool isMatch_Wildcard_Iterative(string text, string pattern)
{
    // keep tracking index of nearest '*' in pattern and the index of its initial corresponding element in text.
    size_t j = 0;
    for (int i = 0, matchStartPosition = -1, lastAsteriskPosition = -1; i < text.length();)
    {
        // Everytime we see a '*', update indices and keep moving forward. This also solve the continous '*' case.
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
        else // mistmatch
        {
            if (lastAsteriskPosition >= 0) // rollback to nearest '*' if exists.
            {
                // Here we try to match the nearest '*' to 1 character, 2 characters, so everytime we shift matchStartPosition right by 1.
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
// BFS would be easier to understand than greedy.Starting from first element(root), mark all unvisited elements that current
// level can reach as next level, keep finding the start and end of each level, until current level covers the last element.
int jump(vector<int>& nums)
{
    // Next level's range is nums[current level's end + 1 .. right most location current level can reach]
    for (size_t levelStart = 0, levelEnd = 0, level = 0, currentLevelRighMostReachableLocation = 0;
        levelStart < nums.size();
        levelStart = levelEnd + 1, levelEnd = currentLevelRighMostReachableLocation, ++level)
    {
        // Current level covers the last element, means we've reached the destination in current level, so level is the shortest step count.
        if (levelEnd >= nums.size() - 1)
        {
            return level;
        }

        // find the right most location that current level can reach.
        for (size_t i = levelStart; i <= levelEnd; ++i)
        {
            // for element nums[i], its right most reachable location is nums[i] + i
            if (nums[i] + i > currentLevelRighMostReachableLocation)
            {
                currentLevelRighMostReachableLocation = nums[i] + i;
            }
        }
    }

    return -1;
}

// 46. Permutations
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
// Given array nums[0, size-1], permutation is, every time we choose an unused element until all elements have been choosen, the
// order how we choose elements is a permutation. To reuse storage, we use a variable currentIndex to indicates the beginning of
// unused elements, its left, nums[0 .. currentIndex-1] are the elements we currently have choosen. Everytime we choose an element
// in nums[currentIndex .. size-1] and exchange it with nums[currentIndex], treated this process as 'choose an unused element',
// update currentIndex to currentIndex+1, and keep repeating this process until currentIndex == size, we get one permutation.
void dfsPermuteImpl(vector<int>& nums, size_t currentIndex, vector<vector<int>>& results)
{
    if (currentIndex == nums.size() - 1)
    {
        results.push_back(nums);
        return;
    }

    // respectively choose nums[currentIndex], nums[currentIndex + 1], ... nums[size-1], exchange it with nums[currentIndex]
    // (first element of range), then continue the process for range [currentIndex+1 .. size-1].
    for (size_t i = currentIndex, j; i < nums.size(); ++i)
    {
        swap(nums[currentIndex], nums[i]);
        dfsPermuteImpl(nums, currentIndex + 1, results);
        swap(nums[currentIndex], nums[i]);
    }
}
vector<vector<int>> dfsPermute(vector<int>& nums)
{
    // no need to define a vector<int> path since nums itself records current path.
    vector<vector<int>> results;
    dfsPermuteImpl(nums, 0, results);
    return results;
}
vector<vector<int>> permute(vector<int>& nums)
{
    return dfsPermute(nums);
}

// 47. Permutations II
void dfsPermuteUniqueImpl(vector<int>& nums, size_t currentIndex, vector<vector<int>>& results)
{
    if (currentIndex == nums.size() - 1)
    {
        results.push_back(nums);
        return;
    }

    for (size_t i = currentIndex, j; i < nums.size(); ++i)
    {
        // As we did in problem 46, we are going to exchange nums[i] with nums[currentIndex], but we need to make sure that we 
        // haven't choosen nums[i] before, we check if there is a duplication of nums[i] within range num[currentIndex .. i-1], 
        // if true, then it means we have choosen nums[i] before so we should not choose it again.
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
// this solution rotate coil by coil, start from outmost coil.
void clockwiseRotateByCoil(vector<vector<int>>& matrix)
{
    // Generally, for element matrix[i][j], it will be moved to matrix[j][n-i-1] after rotation, oppositely, for location
    // matrix[i][j], after rotation, the element on it will be matrix[n-j-1][i].
    for (size_t i = 0, n = matrix[i].size(); i < matrix.size() / 2; ++i)
    {
        for (size_t j = i; j < n - 1 - i; ++j)
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
// first step, mirror on diagnoal (swap matrix[i][j] with matrix[j][i]). Then if it is clockwise, 
// flip matrix's each column (swap each column matrix[*][j] with matrix[*][n-1-j])
// 1 2 3     1 4 7    7 4 1
// 4 5 6  => 2 5 8 => 8 5 2
// 7 8 9     3 6 9    9 6 3
// if it is counterclockwise, flip matrix's each row (swap each row matrix[i] with row matrx[n-1-i])
// 1 2 3     1 4 7    3 6 9
// 4 5 6  => 2 5 8 => 2 5 8
// 7 8 9     3 6 9    1 4 7
void clockwiseRotateByFlipping(vector<vector<int>>& matrix)
{
    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = i + 1; j < matrix[i].size(); ++j) // Do swap for top half only!
        {
            const int temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }

    for (size_t i = 0; i < matrix.size(); ++i)
    {
        for (size_t j = 0; j < matrix[i].size() / 2; ++j)
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
    transform(charset2Strings.begin(), charset2Strings.end(), results.begin(), [](const auto &pair) {return pair.second; });
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
    for (unsigned i = 0, absn = abs(n); i <= 31; ++i, absn >>= 1, x *= x)
    {
        if (absn & 1)
        {
            result *= x;
        }
    }

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
            invalid = path[i] == j || i + path[i] == currentRow + j || (int)(i - path[i]) == (int)(currentRow - j);
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
        for(unsigned column : solution)
        {
            string row(n, '.');
            row[column] = 'Q';
            matrix.push_back(row);
        }

        results.push_back(matrix);
    }

    return results;
}

// 53. Maximum Subarray
// Some throughts:
// This is a DP problem, at first I want to define dp[i] as: the maximum sum of subarray in range nums[0..i], 
// so when i increases from 0 to n-1 (n is the length of nums array), we got the final answer at dp[n-1]. However
// I soon realize this doesn't work. The reason is given an i, we only konw the max subarray's sum in the range 
// nums[0..i-1], but we don't know where the previous max subarray ends, it could end at nums[i-1], or before 
// nums[i-1], this information is very important because we need to decide whether nums[i] is possible to join 
// previous max subarray, or must become a start of new subarray. It is very like to the question 32 (longest
// valid parentheses), which inspires me to change the definition of dp[i] to: given range nums[0..i], for all 
// subarraies end at num[i], the maximum sum of those subarraies. With this new definition, given a dp[i], we 
// know that the max subarray ends at nums[i], and the final answer is the maximum number in dp[0..n-1]. Why? 
// Back to the original question, given an array nums[0..n-1], its max subarray must ends at a certain element
// in nums, given that dp[0..n-1] stores the max subarray's sum ends at nums[0], nums[1], ... nums[n-1], so the 
// max subarray's sum is the maximum element in dp[0..n-1].
// Then next step is how to get the transition function. Let's starts from array has 3 element : 1, -2, 3
// i = 0, max sum is 1, no doubt, dp[0] = 1.
// i = 1, we are facing two choices, whether -2 should join previous max subarry or start a new subarray. If 
//        joining previous max subarray, then the max subarray is [1, -2] and sum is 1-2 = -1; if we decide to 
//        start a new subarray, the new subarray's sum is -2, -1 > -2, so -2 should join, and dp[1] = -1.
// i = 2, because dp[1] = -1, if 3 join previous subarray, new sum is 3-1=2, if we start a new subarray from 3,
//        the sum of new subarray is 3, 3>2 so we should start a new subarray.
// From this we can see, actually it doesn't matter nums[i] is positive or negative, as long as dp[i-1] is positive,
// nums[i] should always join previous subarry.
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
    for (size_t i = 1; i < nums.size(); ++i)
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
    int sum = 0;
    int maxSum = INT_MIN;
    for (int num : nums)
    {
        sum = sum + num > num ? sum + num : num;

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

// 144. Binary Tree Preorder Traversal
vector<int> preorderTraversal(TreeNode* root)
{
    vector<int> result;
    stack<TreeNode*> stk;
    stk.push(root);
    while (!stk.empty())
    {
        root = stk.top(); // reuse root
        stk.pop();

        if (root != nullptr)
        {
            result.push_back(root->val);

            if (root->right != nullptr)
            {
                stk.push(root->right);
            }

            if (root->left != nullptr)
            {
                stk.push(root->left);
            }
        }
    }

    return result;
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
    TreeNode* r = root;
    while (r != nullptr)
    {
        if (r->val > p->val && r->val > q->val)
        {
            r = r->left;
        }
        else if (r->val < p->val && r->val < q->val)
        {
            r = r->right;
        }
        else
        {
            break;
        }
    }

    return r;
}

// 415. Add Strings
string addStrings(string num1, string num2)
{
    string sum;
    char carry = '0';
    for (int i = num1.length() - 1, j = num2.length() - 1, r; i >= 0 || j >= 0; --i, --j)
    {
        if (i >= 0 && j >= 0)
        {
            r = (num1[i] - '0') + (num2[j] - '0') + (carry - '0');
        }
        else if (i >= 0)
        {
            r = (num1[i] - '0') + (carry - '0');
        }
        else if (j >= 0)
        {
            r = (num2[j] - '0') + (carry - '0');
        }

        carry = r / 10 + '0';
        sum.insert(0, 1, static_cast<char>((r % 10) + '0'));
    }

    if (carry - '0' > 0)
    {
        sum.insert(0, 1, carry);
    }

    return sum;
}
