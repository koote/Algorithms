//////////////////////////////////////////////////////////////////////////
// Facebook in-person 面试题
// 给定一个函数 int recv(char* buf, size_t sizeInChars)
// 写一个函数read_line，作用是返回一行内容，\n作为一行的结束。
// 注意第一，没有问一行最长有多长，所以这是一个潜在的问题；
// 第二，当找到结尾也没有找到换行符时，说明要继续调用recv读入接下来的数据，在其中找换行。
// 第三，可以在函数中用malloc然后要求调用者用free去释放内存。
// 第四，可以用全局变量等任何违反好设计的东西，Facebook只考察算法主题。

// Let's say a line will not exceed 1024.
#define BUFF_SIZE 1024

char g_chBuffer[BUFF_SIZE];
size_t g_uNextStart = 0;
size_t g_uBufferUsed = 0;

// A dumb function to satisfy the compiler.
int recv(char* buf, size_t len)
{
    return len;
}

// return a line to caller, and the length of that line.
char* read_line(int& nLineLen)
{
    while (1)
    {
        size_t i = g_uNextStart;
        while (g_chBuffer[i] != '\n' && i < g_uBufferUsed)
        {
            ++i;
        }

        // we have reached the end of buffer but still cannot find a LF,
        // or the buffer is empty.
        if ((i < g_uBufferUsed && g_chBuffer[i] != '\n') || i == g_uBufferUsed)
        {
            // Let's check if we have enough free space to read more data in.
            // if the buffer is full, we cannot call recv and return error to caller.
            // Note that before g_uNextStart is available space.
            size_t uBufferRemain = sizeof(g_chBuffer) -  (g_uBufferUsed - g_uNextStart);
            if (uBufferRemain == 0)
            {
                nLineLen = -1; //indicates an error, buffer full.
                return nullptr;
            }

            // Shift g_chBuffer[g_uNextStart .. g_uDataSize-1] to the beginning of buffer.
            // And then call recv to fill the remaining space.
            for (int j = g_uNextStart; j < g_uBufferUsed; ++j)
            {
                g_chBuffer[j - g_uNextStart] = g_chBuffer[j];
            }

            // And correct the used size of buffer.
            g_uBufferUsed = g_uBufferUsed - g_uNextStart;
            g_uNextStart = 0;

            // Read more data
            g_uBufferUsed += recv(&g_chBuffer[g_uBufferUsed], sizeof(g_chBuffer) - g_uBufferUsed);
        }
        else // Now g_chBuffer[i] is a \n character
        {
            nLineLen = i - g_uNextStart;
            char* pResult = new char[nLineLen];
            for (char* p = pResult; g_uNextStart < i; ++g_uNextStart)
            {
                *p++ = g_chBuffer[g_uNextStart];
            }
            ++g_uNextStart; // jump over the LF, move to next char

            return pResult;
        }
    }
}
