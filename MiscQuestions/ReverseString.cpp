#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <conio.h>

// Reverse characters within lpszStart .. lpszEnd.
void ReverseString(char* lpszStart, char* lpszEnd)
{
    if (lpszStart >= lpszEnd)
    {
        return;
    }

    char* l = lpszStart;
    char* r = lpszEnd;
    while (l < r)
    {
        char ch = *l;
        *l = *r;
        *r = ch;
        ++l;
        --r;
    }
}

// Reverse the whole sentence, but every word is not reversed.
void ReverseSentence(char* lpszSentence)
{
    if (lpszSentence == NULL)
    {
        return;
    }

    char* lpszStart = lpszSentence;
    char* lpszEnd = lpszSentence + strlen(lpszSentence) - 1; // Exclude the null terminator.
    ReverseString(lpszStart, lpszEnd);

    // Then search word and reverse every word.
    lpszStart = lpszSentence;
    lpszEnd = lpszSentence;
    while (true)
    {
        // Forward search blank character.
        while (*lpszEnd != ' ' && *lpszEnd != '\0')
        {
            ++lpszEnd;
        }

        // Now lpszEnd should point to a blank or null terminator.
        ReverseString(lpszStart, lpszEnd-1);

        if (*lpszEnd == '\0')
        {
            break;
        }
        else
        {
            lpszStart = ++lpszEnd;
        }
    }

    printf(lpszSentence);
}
