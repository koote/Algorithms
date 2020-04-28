#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include "DataStructure.h"

using namespace std;

// Helpers
extern ListNode* createList(vector<int>& nums);

// Solutions
extern vector<int> twoSum(vector<int>& nums, int target);
extern ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
extern int lengthOfLongestSubstring(string s);
extern double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
extern string searchPalindrome(string s, int l, int r);
extern string longestPalindrome(string s);
extern string zigzagConvert(string s, int numRows);
extern int reverseInteger(int x);
extern int myAtoi(string str);
extern bool isPalindrome(int x);
extern bool isMatch_Regex(string text, string pattern);
extern int maxArea(vector<int>& height);
extern string intToRoman(int num);
extern int romanToInt(string s);
extern string longestCommonPrefix(vector<string>& strs);
extern vector<vector<int>> threeSum(vector<int>& nums);
extern int threeSumClosest(vector<int>& nums, int target);
extern vector<string> letterCombinations(const string& digits);
extern vector<vector<int>> fourSum(vector<int>& nums, int target);
extern ListNode* removeNthFromEnd(ListNode* head, int n);
extern bool isValid(string s);
extern ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
extern vector<string> generateParenthesis(int n);
extern ListNode* mergeKLists(vector<ListNode*>& lists);
extern ListNode* swapPairs(ListNode* head);
extern ListNode* reverseKGroup(ListNode* head, int k);
extern int removeDuplicates(vector<int>& nums);
extern int removeElement(vector<int>& nums, int val);
extern int strStr(string haystack, string needle);
extern int divide(int dividend, int divisor);
extern vector<int> findSubstring(const string& s, vector<string>& words);
extern void nextPermutation(vector<int>& nums);
extern int longestValidParentheses(string s);
extern int search(vector<int>& nums, int target);
extern vector<int> searchRange(vector<int>& nums, int target);
extern int searchInsert(vector<int>& nums, int target);
extern bool isValidSudoku(vector<vector<char>>& board);
extern void solveSudoku(vector<vector<char>>& board);
extern string countAndSay(int n);
extern vector<vector<int>> combinationSum(vector<int>& candidates, int target);
extern vector<vector<int>> combinationSum2(vector<int>& candidates, int target);
extern int firstMissingPositive(vector<int>& nums);
extern int trap(vector<int>& height);
extern string multiply(string num1, string num2);
extern bool isMatch_Wildcard(string text, string pattern);
extern int jump(vector<int>& nums);
extern vector<vector<int>> permute(vector<int>& nums);
extern vector<vector<int>> permuteUnique(vector<int>& nums);
extern void rotate(vector<vector<int>>& matrix);
extern vector<vector<string>> groupAnagrams(vector<string>& strs);
extern double myPow(double x, int n);
extern vector<vector<string>> solveNQueens(int n);
extern int totalNQueens(int n);
extern int maxSubArray(vector<int>& nums);
extern vector<int> spiralOrder(vector<vector<int>>& matrix);
extern bool canJump(vector<int>& nums);
extern vector<Interval> merge(vector<Interval>& intervals);
extern vector<Interval> insert(vector<Interval>& intervals, Interval newInterval);
extern int lengthOfLastWord(string s);
extern vector<vector<int>> generateMatrix(int n);
extern string getPermutation(int n, int k);
extern ListNode* rotateRight(ListNode* head, int k);
extern int uniquePaths(int m, int n);
extern int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid);
extern int minPathSum(vector<vector<int>>& grid);
extern bool isNumber(string s);
extern vector<int> plusOne(vector<int>& digits);
extern string addBinary(string a, string b);
extern vector<string> fullJustify(vector<string>& words, int maxWidth);
extern int mySqrt(int x);
extern int climbStairs(int n);
extern string simplifyPath(string& path);
extern int minDistance(string word1, string word2);
extern void setZeroes(vector<vector<int>>& matrix);
extern bool searchMatrix(vector<vector<int>>& matrix, int target);
extern void sortColors(vector<int>& nums);
extern string minWindow(string s, string t);
extern vector<vector<int>> combine(int n, int k);
extern vector<vector<int>> subsets(vector<int>& nums);
extern bool exist(vector<vector<char>>& board, string word);
extern int removeDuplicates2(vector<int>& nums);
extern bool search2(vector<int>& nums, int target);
extern ListNode* deleteDuplicates2(ListNode* head);
extern ListNode* deleteDuplicates(ListNode* head);
extern int largestRectangleArea(vector<int>& heights);
extern int maximalRectangle(vector<vector<char>>& matrix);
extern ListNode* partition(ListNode* head, int x);
extern bool isScramble(const string& s1, const string& s2);
extern void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);
extern vector<int> grayCode(int n);
extern vector<vector<int>> subsetsWithDup(vector<int>& nums);
extern int numDecodings(const string& s);
extern ListNode* reverseBetween(ListNode* head, int m, int n);
extern vector<string> restoreIpAddresses(const string& s);
extern vector<int> inorderTraversal(TreeNode* root);
extern vector<TreeNode*> generateTrees(int n);
extern int numTrees(int n);
extern bool isInterleave(const string& s1, const string& s2, const string& s3);
extern bool isValidBST(TreeNode* root);
extern void recoverTree(TreeNode* root);
extern bool isSameTree(TreeNode* p, TreeNode* q);
extern bool isSymmetric(TreeNode* root);
extern vector<vector<int>> levelOrder(TreeNode* root);
extern vector<vector<int>> zigzagLevelOrder(TreeNode* root);
extern int maxDepth(TreeNode* root);
extern TreeNode* buildTree105(vector<int>& preorder, vector<int>& inorder);
extern TreeNode* buildTree106(vector<int>& inorder, vector<int>& postorder);
extern vector<vector<int>> levelOrderBottom(TreeNode* root);
extern TreeNode* sortedArrayToBST(vector<int>& nums);
extern TreeNode* sortedListToBST(ListNode* head);
extern bool isBalanced(TreeNode* root);
extern int minDepth(TreeNode* root);
extern bool hasPathSum(TreeNode* root, int sum);
extern vector<vector<int>> pathSum(TreeNode* root, int sum);
extern void flatten(TreeNode* root);
extern int numDistinct(const string& s, const string& t);
extern void connect(TreeLinkNode* root);
extern vector<vector<int>> generate(int numRows);
extern vector<int> getRow(int rowIndex);
extern int minimumTotal(vector<vector<int>>& triangle);
extern int maxProfit(vector<int>& prices);
extern int maxProfit2(vector<int>& prices);
extern int maxProfit3(vector<int>& prices);
extern int maxPathSum(TreeNode* root);
extern bool isPalindrome(string s);
extern vector<vector<string>> findLadders(const string& beginWord, const string& endWord, const vector<string>& wordList);
extern int ladderLength(const string& beginWord, const string& endWord, const vector<string>& wordList);
extern int longestConsecutive(vector<int>& nums);
extern int sumNumbers(TreeNode* root);
extern void solve(vector<vector<char>>& board);
extern vector<vector<string>> partition(string& s);
extern int minCut(string s);
extern Node* cloneGraph(Node* node);
extern int canCompleteCircuit(vector<int>& gas, vector<int>& cost);
extern int candy(vector<int>& ratings);
extern int singleNumber(vector<int>& nums);
extern int singleNumber2(vector<int>& nums);
extern RandomListNode* copyRandomList(RandomListNode *head);
extern bool wordBreak(string s, vector<string>& wordDict);
extern vector<string> wordBreak2(string s, vector<string>& wordDict);
extern bool hasCycle(ListNode* head);
extern ListNode* detectCycle(ListNode* head);
extern void reorderList(ListNode* head);
extern vector<int> preorderTraversal(TreeNode* root);
extern vector<int> postorderTraversal(TreeNode* root);
class LRUCache;
extern ListNode* insertionSortList(ListNode* head);
extern ListNode* sortList(ListNode* head);
extern int maxPoints(vector<vector<int>>& points);
extern int evalRPN(vector<string>& tokens);
extern string reverseWords(string s);
extern int maxProduct(vector<int>& nums);
extern int findMin(vector<int>& nums);
extern int findMin2(vector<int>& nums);
class MinStack;

extern int maxProfit4(int k, vector<int>& prices);

extern int numIslands(vector<vector<char>>& grid);

extern bool isHappy(int n);

extern ListNode* reverseList(ListNode* head);

extern TreeNode* invertTree(TreeNode* root);

extern TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);

class BinaryTreeCodec;

extern bool increasingTriplet(vector<int>& nums);

extern int thirdMax(vector<int>& nums);
extern string addStrings(const string& num1, const string& num2);

class BinarySearchTreeCodec;

extern int findLengthOfLCIS(vector<int>& nums);

extern int maxAreaOfIsland(vector<vector<int>>& grid);

extern vector<vector<int>> kClosest(vector<vector<int>>& points, int k);

extern int mincostTickets(vector<int>& days, vector<int>& costs);

void main()
{
    const auto start = clock();

    vector<vector<char>> matrix({
        {'0','1','1','0','1'},
        {'1','1','0','1','0'},
        {'0','1','1','1','0'},
        {'1','1','1','1','0'},
        {'1','1','1','1','1'},
        {'0','0','0','0','0'}
        });
    maximalRectangle(matrix);

    vector<vector<char>> sudoku({
        { '5', '3', '.', '.', '7', '.', '.', '.', '.' },
        { '6', '.', '.', '1', '9', '5', '.', '.', '.' },
        { '.', '9', '8', '.', '.', '.', '.', '6', '.' },
        { '8', '.', '.', '.', '6', '.', '.', '.', '3' },
        { '4', '.', '.', '8', '.', '3', '.', '.', '1' },
        { '7', '.', '.', '.', '2', '.', '.', '.', '6' },
        { '.', '6', '.', '.', '.', '.', '2', '8', '.' },
        { '.', '.', '.', '4', '1', '9', '.', '.', '5' },
        { '.', '.', '.', '.', '8', '.', '.', '7', '9' }
        });

    ladderLength("hit", "cog", { "hot","dot","dog","lot","log","cog" });

    const auto duration = (clock() - start) / static_cast<double>(CLOCKS_PER_SEC);
    cout << duration << endl;
}
