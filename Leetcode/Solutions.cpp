#include <string>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <stack>
#include "DataStructure.h"
using namespace std;

// 1. Two Sum
vector<int> twoSum(vector<int>& nums, int target)
{
    int length = nums.size();
    vector<int> result;
    unordered_map<int, int> map; // <value, original_index>
    unordered_map<int, int>::iterator iterator;

    for (int i = 0; i < length; ++i)
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
        // When a[i-1] == b[j-1], they have k-2 elements left and m+n-k elements right, so either is the kth.
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

// 5. Longest Palindromic Substring
string searchPalindrome(string s, int left, int right)
{
    // when loop ends, no matter it is because of out of bounds or s[left] and s[rigth] no longer equal, current palindrome is always s[left+1 .. right-1].
    for (; left >= 0 && right < s.length() && s[left] == s[right]; left--, right++);
    return s.substr(left + 1, right - 1 - left);
}
string longestPalindrome(string s)
{
    string result("");
    for (int position = 0; position < s.length(); ++position)
    {
        string palindrome1 = searchPalindrome(s, position, position);
        if (palindrome1.length() > result.length())
        {
            result = palindrome1;
        }

        // don't worry that position+1 could out of right bound, since searchPalindrome will check j < s.length()
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

    string result("");
    int distance = 2 * numRows - 2;
    for (int r = 0; r < numRows; ++r)
    {
        int curDistance1 = distance - r * 2;
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
    int sign = x < 0 ? -1 : 1;
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
        int digit = str[i] - '0';

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
    return (numOfDigits & 0x1 ? x / 10 : x) == lowhalf;;
}

// 10. Regular Expression Matching
bool isMatch(string text, string pattern)
{
    // Exit condition of recursion should be: when pattern is processed, text must also be processed, 
    // but not vice versa. Why? because when text is empty, pattern is a* or .*, they still match.
    if (pattern.length() == 0)
    {
        return text.length() == 0;
    }

    // If there is a * followed, there are 2 cases:
    // (1) the preceding character of * appears zero times, next match would be isMatch(text, pattern.substr(2)).
    // (2) the preceding character of * appears >= 1 times, next match would be isMatch(text.substr(1), pattern.substr(1)) if current character matches pattern.
    return (pattern.length() > 1 && pattern[1] == '*') ?
        isMatch(text, pattern.substr(2)) || (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch(text.substr(1), pattern) :
        (text.length() > 0 && (pattern[0] == '.' || text[0] == pattern[0])) && isMatch(text.substr(1), pattern.substr(1));
}

// 11. Container With Most Water
int maxArea(vector<int>& height)
{
    int max = 0;
    for (int i = 0, j = height.size() - 1; i < j;)
    {
        int current = 0;
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

//13. Roman to Integer
int romanToInt(string s)
{
    throw new exception();
}

// 14. Longest Common Prefix
string longestCommonPrefix(vector<string>& strs)
{
    if (strs.size() == 0)
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
    const size_t size = nums.size();
    vector<vector<int>> result(0);

    if (size < k || k == 0)
    {
        return result;
    }

    if (k == 2)
    {
        for (size_t i = 0, j = size - 1; i < j;)
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

    for (size_t i = 0; i < size; ++i)
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
        for (size_t j = i + 1, p = k - 1; j < size && p > 0; s += nums[j++], --p);
        if (s > sum)
        {
            break;
        }

        // Same reason, nums[size-k+1..size-1] are largest in nums[i+1..size-1].
        // if SUM(nums[i] + nums[size-k+1] + ... + nums[size-1]) < sum, we can ignore current loop (jump over nums[i])
        // because that sum is the largest we can produce in current loop.
        s = nums[i];
        for (size_t j = size - 1, p = k - 1; j > 0 && p > 0; s += nums[j--], --p);
        if (s < sum)
        {
            continue;
        }

        vector<int> v(0);
        for (size_t j = i + 1; j < size; v.push_back(nums[j++]));

        vector<vector<int>> cures = kSum(v, k - 1, sum - nums[i]);
        for (size_t j = 0; j < cures.size(); ++j)
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

// 16. 3Sum Closest
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

// 17. Letter Combinations of a Phone Number
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
        for (size_t i = 0; i < val.length(); ++i)
        {
            result.push_back(val.substr(i, 1));
        }

        return result;
    }

    vector<string> cures = letterCombinations(digits.substr(1));
    string val = dict[digits[0] - '0'];
    for (size_t i = 0; i < val.length(); ++i)
    {
        for (size_t j = 0; j < cures.size(); ++j)
        {
            result.push_back(val.substr(i, 1).append(cures[j]));
        }
    }

    return result;
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

// 20. Valid Parentheses
bool isValid(string s)
{
    stack<char> stk;
    for (int i = 0, length = s.length(); i < length; ++i)
    {
        if (s[i] == '{' || s[i] == '[' || s[i] == '(')
        {
            stk.push(s[i]);
        }
        else if ((s[i] == '}' && !stk.empty() && stk.top() == '{') || (s[i] == ']' && !stk.empty() && stk.top() == '[') || (s[i] == ')' && !stk.empty() && stk.top() == '('))
        {
            stk.pop();
        }
        else
        {
            return false;
        }
    }

    return stk.empty();
}

// 21. Merge Two Sorted Lists
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2)
{
    ListNode head(-1);
    ListNode* last = &head;
    while (l1 != nullptr && l2 != nullptr)
    {
        ListNode* r;
        if (l1->val < l2->val)
        {
            r = l1;
            l1 = l1->next;
        }
        else
        {
            r = l2;
            l2 = l2->next;
        }

        r->next = nullptr;
        last->next = r;
        last = r;
    }

    if (l1 != nullptr)
    {
        last->next = l1;
    }
    else if (l2 != nullptr)
    {
        last->next = l2;
    }

    return head.next;
}

// 22. Generate Parentheses
vector<string> generateParenthesisInternal(string trail, int remainingLeftBrackets, int remainingRightBrackets)
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
            vector<string> cures1 = generateParenthesisInternal(trail + '(', remainingLeftBrackets - 1, remainingRightBrackets);
            results.insert(results.end(), cures1.begin(), cures1.end());
        }

        if (remainingRightBrackets > 0)
        {
            vector<string> cures2 = generateParenthesisInternal(trail + ')', remainingLeftBrackets, remainingRightBrackets - 1);
            results.insert(results.end(), cures2.begin(), cures2.end());
        }
    }

    return results;
}
vector<string> generateParenthesis(int n)
{
    return generateParenthesisInternal("", n, n);
}

// 23. Merge k Sorted Lists
void minHeapify(vector<ListNode*>& nodes, int i)
{
    int size = nodes.size();
    int smallest = i;

    if (2 * i + 1 < size && nodes[smallest]->val > nodes[2 * i + 1]->val) // left child
    {
        smallest = i * 2 + 1;
    }

    if (2 * i + 2 < size && nodes[smallest]->val > nodes[2 * i + 2]->val) // right child
    {
        smallest = 2 * i + 2;
    }

    if (smallest != i)
    {
        ListNode* temp = nodes[smallest];
        nodes[smallest] = nodes[i];
        nodes[i] = temp;

        minHeapify(nodes, smallest);
    }
}
void buildMinHeap(vector<ListNode*>& nodes)
{
    for (int i = (nodes.size() - 2) / 2; i >= 0; --i)
    {
        minHeapify(nodes, i);
    }
}
ListNode* mergeKLists(vector<ListNode*>& lists)
{
    ListNode head(-1);
    ListNode* last = &head;

    vector<ListNode*> nodes;
    for (ListNode* node : lists)
    {
        if (node != nullptr)
        {
            nodes.push_back(node);
        }
    }

    // nodes[0] will be the smallest one.
    buildMinHeap(nodes);

    while (!nodes.empty())
    {
        last->next = nodes[0];
        last = last->next;
        nodes[0] = nodes[0]->next;
        last->next = nullptr;

        if (nodes[0] == nullptr) // this list has no remaining nodes.
        {
            // NOTE: After heap is built, never use erase to remove the root of heap, that will break the heap.
            // We should always only exchange node[0] with last leaf node to get the root out of heap.
            // WHY? Beucase after the heap is adjusted, for each element i, its left (2i+1) and right (2i+2) child
            // are indexed by their position, if we erase the root, all elements' indexes shift left 1 position,
            // so element i becomes i-1, and its new left child is 2i-1 (which was 2i) and new right child is 2i (which was 2i+1).
            // The order of each nodes and its children could be broken.
            // If a heap is broken, we cannot just use heapify to make it is a heap again, we need a full rebuild.

            ListNode* temp = nodes[0];
            nodes[0] = nodes[nodes.size() - 1];
            nodes[nodes.size() - 1] = temp;
            nodes.pop_back();
        }

        minHeapify(nodes, 0);
    }

    return head.next;
}

// 24. Swap Nodes in Pairs
ListNode* swapPairs(ListNode* head)
{
    if (head == nullptr || head->next == nullptr) // 0 or 1 node.
    {
        return head;
    }

    ListNode* p = head;
    ListNode* q = head->next;

    ListNode dummy(-1);
    dummy.next = head;
    ListNode* r = &dummy;

    while (p != nullptr && q != nullptr)
    {
        r->next = q;
        p->next = q->next;
        q->next = p;

        r = p;
        p = p->next;
        q = p != nullptr ? p->next : nullptr;
    }

    return dummy.next;
}

//25. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, int k)
{
    if (head == nullptr || head->next == nullptr || k <= 1)
    {
        return head;
    }

    ListNode dummy(-1);
    dummy.next = head;
    ListNode* r = &dummy;

    ListNode* p = head;
    ListNode* q = head;

    while (p != nullptr && q != nullptr)
    {
        int n = k - 1;
        for (; n > 0 && q != nullptr; --n, q = q->next);

        if (n == 0 && q != nullptr)
        {
            r->next = q;
            ListNode* temp = q->next;

            // reverse p..q
            ListNode* s = p;
            ListNode* t = p->next;
            ListNode* u;
            while (s != q)
            {
                u = t->next;
                t->next = s;
                s = t;
                t = u;
            }

            p->next = temp;

            r = p;
            p = p->next;
            q = p;
        }
    }

    return dummy.next;
}

// 26. Remove Duplicates from Sorted Array
int removeDuplicates(vector<int>& nums)
{
    // last points to the last element in packed part of array.
    int last = -1;
    for (size_t i = 0; i < nums.size() && last < (int)nums.size();)
    {
        for (; i + 1 < nums.size() && nums[i] == nums[i + 1]; ++i);
        nums[++last] = nums[i++];
    }

    return last + 1;
}
int removeDuplicates2(vector<int>& nums)
{
    if (nums.size() == 0)
    {
        return 0;
    }

    int last = 0;
    for (size_t i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != nums[last])
        {
            nums[++last] = nums[i];
        }
    }

    return last + 1;
}

// 27. Remove Element
int removeElement(vector<int>& nums, int val)
{
    int last = 0;
    for (size_t i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != val)
        {
            nums[last++] = nums[i];
        }
    }

    return last;
}

//28. Implement strStr()
int strStr(string haystack, string needle)
{
    if (needle.length() == 0)
    {
        return 0;
    }

    for (int i = 0, tlen = haystack.length(), plen = needle.length(); i <= tlen - plen; ++i)
    {
        int p = 0;
        for (; p < plen && p + i < tlen && haystack[p + i] == needle[p]; ++p);
        if (p == plen)
        {
            return i;
        }
    }

    return -1;
}

// 29. Divide Two Integers
int divide(int dividend, int divisor)
{
    bool negative = (((dividend ^ divisor) >> 31) & 0x1) == 1;
    unsigned int a = dividend < 0 ? -dividend : dividend; // Use uint, so abs(a) will not overflow even a = INT_MIN.
    unsigned int b = divisor < 0 ? -divisor : divisor;
    unsigned int quotient = 0;

    while (a >= b)
    {
        unsigned int k = 0;

        // We are looking for a k that b << k <= a but b << (k + 1) > a. The loop looks complicated since
        // we need to handle overflow. E.g. a == 1 << 31 and b == 1, k should be 31, because b << 31 == a
        // and b << 32 > a, however b << 32 actually overflows so b << 32 == 0, this loop will never end.
        // To prevent overflow, b << k must <= 0x80000000 / 2, otherwise, b << (k + 1) will overflow.
        for (; a >= b << k && b << k <= (0x80000000 >> 1); ++k);

        // When loop ends, if a < b << k, it means no overflow happens, and k increased one additional time, we
        // need to decrease k; otherwise (a >= b << k), it means overflow happens, k is the maximum we can find.
        quotient |= 1 << (a < b << k ? --k : k);
        a -= b << k;
    }

    // If negative is false, quotient must <= INT_MAX, otherwise overflow, should return INT_MAX.
    return negative ? 0 - quotient : quotient > INT_MAX ? INT_MAX : quotient;
}

// 30. Substring with Concatenation of All Words
vector<int> findSubstring(string s, vector<string>& words)
{
    vector<int> result;

    if (words.size() == 0)
    {
        return result;
    }

    size_t wordLen = words[0].length();
    size_t patternLen = wordLen * words.size();

    if (patternLen > s.length())
    {
        return result;
    }

    // Here we use a hash table to store for each word, how many times it occurs.
    unordered_map<string, int> wordCount;
    for (size_t i = 0; i < words.size(); ++i)
    {
        ++wordCount[words[i]];
    }

    for (size_t i = 0; i <= s.length() - patternLen; ++i)
    {
        unordered_map<string, int> unusedWords(wordCount);

        // Check if all words present in current substring s[i..i+patternLen-1]
        // Slice it to words.size() slices, for each slice, use the unusedWords
        // hash table to quick check if it is a word, if yes, decrease value in
        // hash table, means it has been used.
        for (size_t j = i; j <= i + patternLen - wordLen; j += wordLen)
        {
            string word = s.substr(j, wordLen);
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
    if (nums.size() > 1)
    {
        size_t i; // i-1 will be the first element that nums[i-1] < nums[i].
        size_t j; // j will be the first element in nums[i..size-1] that large than nums[i-1].
        for (i = nums.size() - 1; i > 0 && nums[i] <= nums[i - 1]; --i); // search from end of array for the first element that smaller than its successor.
        sort(nums.begin() + i, nums.end());
        if (i > 0)
        {
            // Search for j that nums[j] is the first one large than nums[i-1], then swap nums[i-1] and nums[j].
            for (j = i; j <= nums.size() - 1 && nums[j] <= nums[i - 1]; ++j);
            swap(nums[i - 1], nums[j]);
        }
    }
}

// 32. Longest Valid Parentheses
int longestValidParentheses(string s)
{
    int slen = s.length();
    int* dp = new int[slen];
    memset(dp, 0, slen * sizeof(int));
    int max = 0;

    for (int i = slen - 2; i >= 0; --i)
    {
        dp[i] = s[i] == '(' ?
            s[i + dp[i + 1] + 1] == ')' ?
            dp[i + 1] + 2 +
            (
                i + dp[i + 1] + 2 < slen ? dp[i + dp[i + 1] + 2] : 0
                ) : 0
            : 0;

        if (dp[i] > max)
        {
            max = dp[i];
        }
    }

    delete[] dp;
    return max;
}
int longestValidParentheses2(string s) // This solution is to demostrate how DP works.
{
    int max = 0;
    int slen = s.length();

    // dp[i] is the longest length of valid parentheses that starts from s[i].
    // Please note that s[i] must be included. If s[i] is ')', then dp[i] = 0,
    // because valid parentheses never start from ')', although there could be
    // a valid parentheses starts from s[i+k] (0 <= k <= slen-i-1), but that is
    // dp[i+k]'s business, not dp[i]'s.
    int* dp = new int[slen];
    dp[slen - 1] = 0;

    for (int i = slen - 2; i >= 0; --i)
    {
        if (s[i] == '(')
        {
            //  (......................)          ....
            // s[i]         s[i + dp[i + 1] + 1]
            if (i + dp[i + 1] + 1 < slen && s[i + dp[i + 1] + 1] == ')')
            {
                dp[i] = dp[i + 1] + 2;

                if (i + dp[i + 1] + 2 < slen)
                {
                    dp[i] += dp[i + dp[i + 1] + 2];
                }
            }
            else
            {
                dp[i] = 0;
            }
        }
        else if (s[i] == ')') // If a string starts with ')', it could not be a valid parentheses.
        {
            dp[i] = 0;
        }

        if (dp[i] > max)
        {
            max = dp[i];
        }
    }

    delete[] dp;
    return max;
}

// 33. Search in Rotated Sorted Array
int search(vector<int>& nums, int target)
{
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right)
    {
        int mid = (left + right) / 2;
        if (nums[mid] == target)
        {
            return mid;
        }
        else if (nums[left] <= nums[mid])   // mid is on left part.
        {
            if (target >= nums[left] && target <= nums[mid]) // nums[left] <= target <= nums[mid]
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
            if (target >= nums[mid] && target <= nums[right]) // nums[mid] <= target <= nums[right]
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
vector<int> searchRange(vector<int>& nums, int target)
{
    vector<int> result;
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right)
    {
        size_t mid = (left + right) / 2;
        if (nums[mid] == target)
        {
            // If we haven't find the first appearance of target (because result is empty), and current is not the 
            // first appearance of target (because nums[mid - 1] == target), keep going left to find the first target
            if (result.empty() && (mid > 0 && nums[mid - 1] == target))
            {
                right = mid - 1;
            }
            else if (!result.empty() && (mid < nums.size() - 1 && nums[mid + 1] == target)) // If we have found the first apperance of target, go right to find the last appearance of target
            {
                left = mid + 1;
            }
            else
            {
                result.push_back(mid);
                if (result.size() == 1)
                {
                    // So nums[mid] is the first appearance of target, now we need to find the last appearance of target.
                    // reset left = mid and right = nums.size()-1.
                    left = mid;
                    right = nums.size() - 1;
                }
                else
                {
                    return result;
                }
            }
        }
        else if (nums[mid] > target) // go left
        {
            right = mid - 1;
        }
        else if (nums[mid] < target) // go right
        {
            left = mid + 1;
        }
    }

    return vector<int>{ -1, -1 };
}

// 35. Search Insert Position
int searchInsert(vector<int>& nums, int target)
{
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right)
    {
        int mid = (left + right) / 2;
        if (nums[mid] == target)
        {
            return mid;
        }
        else if (nums[mid] > target)
        {
            right = mid - 1;
        }
        else if (nums[mid] < target)
        {
            left = mid + 1;
        }
    }

    // when loop ends, left > right, left is the first number that larger than target,
    // which is also the insert location.
    return left;
}

// 36. Valid Sudoku
bool isValidSudoku(vector<vector<char>>& board)
{
    // check 9 rows & 9 columns
    for (int i = 0; i < 9; ++i)
    {
        int rowNumbers[10] = { 0 };
        int colNumbers[10] = { 0 };

        for (int j = 0; j < 9; ++j)
        {
            // row
            if (board[i][j] != '.')
            {
                int val = board[i][j] - '0';
                if (val < 1 || val > 9 || rowNumbers[val] != 0)
                {
                    return false;
                }

                rowNumbers[val] = 1;
            }

            // column
            if (board[j][i] != '.')
            {
                int val = board[j][i] - '0';
                if (val < 1 || val > 9 || colNumbers[val] != 0)
                {
                    return false;
                }

                colNumbers[val] = 1;
            }
        }
    }

    // check 9 3x3 tiles
    for (int i = 0; i <= 6; i += 3)
    {
        for (int j = 0; j <= 6; j += 3)
        {
            int numbers[10] = { 0 };
            for (int k = 0; k < 3; ++k)
            {
                for (int p = 0; p < 3; ++p)
                {
                    if (board[i + k][j + p] != '.')
                    {
                        int val = board[i + k][j + p] - '0';
                        if (val < 1 || val > 9 || numbers[val] != 0)
                        {
                            return false;
                        }

                        numbers[val] = 1;
                    }
                }
            }
        }
    }

    return true;
}

// 37. Sudoku Solver
bool solveSudoku(vector<vector<char>>& board)
{
    for (int i = 0; i < 9; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            if (board[i][j] == '.')
            {
                for (int val = 1; val <= 9; ++val)
                {
                    board[i][j] = (char)(val + '0');
                    if (isValidSudoku(board) && solveSudoku(board))
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

// 38. Count and Say
string countAndSay(int n)
{
    string seq = "1";
    for (; n > 1; --n)
    {
        string newSeq;
        for (size_t i = 0; seq[i] != '\0';)
        {
            size_t j = i + 1;
            for (; seq[j] != '\0' && seq[j] == seq[j - 1]; ++j);
            newSeq.append(1, (char)(j - i + '0'));
            newSeq.append(1, seq[i]);
            i = j;
        }
        seq = newSeq;
    }
    return seq;
}

// 39. Combination Sum
void depthSearchCombinationSum(vector<vector<int>>& results, vector<int>& path, vector<int>& candidates, size_t startPos, int target)
{
    // Found a path, save it to results. Do not clear the path, because it is still needed 
    // for upper level caller. Let's say now the path is ?...??X, after we push this path
    // to results, and return to upper caller, in upper caller, the path is ?...??, it may
    // still need to try another number Y: ?...???Y
    if (target == 0)
    {
        results.push_back(path);
        return;
    }

    // Since the candidates is sorted, to prevent duplicate paths, we only look forward. Because each number can be used
    // mulitpal times, so if we choose candidates[i], next step is we try numbers from candidates[i] to end of array. If
    // every number can be used only once, then next step we start from candidates[i+1].
    for (size_t i = startPos; i < candidates.size() && candidates[i] <= target; ++i) // candidates[i] <= target is pruning
    {
        // Skip duplicates. e.g. candidates = [1,1,2,3], target = 6, let's mark it as (1,1,2,3|6), we can notice that actually
        // (1,2,3|5) is already included in (1,1,2,3|5) :
        // (1,1,2,3|5) = (1,1,2,3|4), (1,2,3|4), (2,3|3), (3|2).
        // (1,2,3|5) =                (1,2,3|4), (2,3|3), (3|2).
        if (i > startPos && candidates[i] == candidates[i - 1])
        {
            continue;
        }
        path.push_back(candidates[i]); // Try to select candidate[i] in current step.
        depthSearchCombinationSum(results, path, candidates, i, target - candidates[i]);
        path.pop_back(); // Must revert path back.
    }
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> results;
    vector<int> path;
    depthSearchCombinationSum(results, path, candidates, 0, target);
    return results;
}

// 40. Combination Sum II
void depthSearchCombinationSum2(vector<vector<int>>& results, vector<int>& path, vector<int>& candidates, size_t startPos, int target)
{
    if (target == 0)
    {
        results.push_back(path);
        return;
    }

    for (size_t i = startPos; i < candidates.size() && candidates[i] <= target; ++i) // candidates[i] <= target is pruning
    {
        if (i > startPos && candidates[i] == candidates[i - 1])
        {
            continue;
        }
        path.push_back(candidates[i]); // Try to select candidate[i] in current step.
        depthSearchCombinationSum2(results, path, candidates, i + 1, target - candidates[i]);
        path.pop_back(); // Must revert path back.
    }
}
vector<vector<int>> combinationSum2(vector<int>& candidates, int target)
{
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> results;
    vector<int> path;
    depthSearchCombinationSum2(results, path, candidates, 0, target);
    return results;
}

// 41. First Missing Positive
int firstMissingPositive(vector<int>& nums)
{
    // For each positive number, try to put it on correct index.
    for (size_t i = 0; i < nums.size();)
    {
        // Put 1 in nums[0], put 2 in num[1], etc.
        if (nums[i] > 0 && nums[i] < (int)nums.size() && nums[i] != nums[nums[i] - 1])
        {
            int temp = nums[nums[i] - 1];
            nums[nums[i] - 1] = nums[i];
            nums[i] = temp;
        }
        else
        {
            ++i;
        }
    }

    for (size_t i = 0; i < nums.size(); ++i)
    {
        if (nums[i] != (int)(i + 1))
        {
            return i + 1;
        }
    }

    return nums.size() + 1;
}

// 42. Trapping Rain Water
int trap(vector<int>& height)
{
    if (height.size() <= 1) return 0;
    unsigned int result = 0;
    for (size_t i = 0; i < height.size();)
    {
        if (height[i] == 0)
        {
            ++i;
            continue;
        };

        size_t j = i + 1;
        unsigned int vol = 0;
        while (j < height.size() && height[j] < height[i])
        {
            vol += height[j];
            ++j;
        }

        if (j < height.size())
        {
            result += height[i] * (j - i) - vol;
        }

        i = j;
    }

    return result;
}

// 144. Binary Tree Preorder Traversal
vector<int> preorderTraversal(TreeNode* root)
{
    vector<int> result;
    stack<TreeNode*> stk;
    stk.push(root);
    while (stk.empty() == false)
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
