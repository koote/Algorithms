// An email's body is composed by text and signature. Length of signaure is always 100 bytes, text length >= 0.
// Implement a class that has a read method, each time calls it, it returns a character of text, if reach the end
// of text, then return -1. Never returns signature.

#include <vector>
using namespace std;

class CTextStream
{
private:
    vector<char> buf1;
    vector<char> buf2;

public:
    char Read()
    {

    }
};
