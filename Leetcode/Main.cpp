#include <string>
#include <regex>
#include <iostream>
using namespace std;

extern string longestPalindrome(string s);
extern int myAtoi(string str);
extern bool isPalindrome(int x);
extern bool isMatch(string s, string p);
extern string longestCommonPrefix(vector<string>& strs);
extern vector<vector<int>> threeSum(vector<int>& nums);
extern vector<vector<int>> fourSum(vector<int>& nums, int target);
extern string intToRoman(int num);
extern vector<vector<int>> threeSum(vector<int>& nums);
extern int threeSumClosest(vector<int>& nums, int target);
extern vector<string> letterCombinations(string digits);

void main()
{
    auto start = std::clock();
    auto x = letterCombinations("5678");
    auto duration1 = (std::clock() - start) / (double)CLOCKS_PER_SEC;
}
