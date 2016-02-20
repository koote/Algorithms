#include <string>
#include <regex>
#include <iostream>
using namespace std;

extern string longestPalindrome(string s);
extern int myAtoi(string str);
extern bool isPalindrome(int x);
extern bool isMatch(string s, string p);
extern string longestCommonPrefix(vector<string>& strs);

void main()
{
    auto x = longestCommonPrefix(vector<string>(1, ""));
}
