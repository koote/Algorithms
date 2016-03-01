#include <string>
#include <vector>
#include <ctime>
#include "DataStructure.h"

using namespace std;

// Helpers
extern ListNode* createList(vector<int>& nums);

// Solutions
extern vector<int> twoSum(vector<int>& nums, int target);
extern ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
extern string longestPalindrome(string s);
extern double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
extern string trySearchPalindromic(string s, int l, int r);
extern string longestPalindrome(string s);
extern string zigzagConvert(string s, int numRows);
extern int reverseInteger(int x);
extern int myAtoi(string str);
extern bool isPalindrome(int x);
extern bool isMatch(string s, string p);
extern int maxArea(vector<int>& height);
extern string intToRoman(int num);
extern int romanToInt(string s);
extern string longestCommonPrefix(vector<string>& strs);
extern vector<vector<int>> threeSum(vector<int>& nums);
extern int threeSumClosest(vector<int>& nums, int target);
extern vector<string> letterCombinations(string digits);
extern vector<vector<int>> fourSum(vector<int>& nums, int target);
extern ListNode* removeNthFromEnd(ListNode* head, int n);
extern ListNode* removeNthFromEnd2(ListNode* head, int n);
extern bool isValid(string s);
extern ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
extern vector<string> generateParenthesis(int n);
extern ListNode* mergeKLists(vector<ListNode*>& lists);
extern ListNode* swapPairs(ListNode* head);
extern ListNode* reverseKGroup(ListNode* head, int k);
extern int removeDuplicates(vector<int>& nums);
extern int removeElement(vector<int>& nums, int val);
extern int strStr(string haystack, string needle);

void main()
{
    auto start = std::clock();

    vector<int> list1{ 2 };
    auto x = removeElement(list1, 3);

    auto duration1 = (std::clock() - start) / (double)CLOCKS_PER_SEC;
}
