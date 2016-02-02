///////////////////////////////////////////////////////////////////////////
// atoi

#define INT_MAX (2147483647)
#define INT_MIN (0-2147483648)

int atoi(const char* str)
{
    if (str == nullptr)
    {
        return 0;
    }

    while (*str == ' ' || *str == '\t' || *str == '\r' || *str == '\n')
    {
        ++str;
    }

    int sign = 1;
    unsigned int total = 0;

    if (*str == '-')
    {
        sign = -1;
        ++str;
    }
    else if (*str == '+')
    {
        ++str;
    }

    while (*str >= '0' && *str <= '9')
    {
        unsigned int digit = *str - '0';

        // 整数的范围为INT_MIN <= total <= INT_MAX
        // 上溢出 : total > (INT_MAX - digit) / 10
        // 下溢出 : total > (-INT_MIN - digit) / 10 = (INT_MAX + 1 - digit) / 10
        // 这两个算式容易理解但是牵涉到浮点数的计算和比较．其实可以转换为整数的比较．
        //
        // 令 total * 10 + digit = y, 则 total = y / 10 且 digit = y % 10.
        // 对于上溢出, y > INT_MAX, 有两种情况:
        // (1) total > INT_MAX / 10
        // (2) total = INT_MAX / 10 但 digit > INT_MAX % 10.
        // 对于下溢出, y > -INT_MIN, 易知total > (-INT_MIN - digit) / 10, 同样转化为整数比较:
        // (1) total > -INT_MIN / 10 = (INT_MAX + 1) / 10
        // (2) total = -INT_MIN / 10 但 digit > -INT_MIN % 10.
        //
        // 注意到INT_MAX / 10 和 -INT_MIN / 10, 取整后是一样的值, 所以上溢出和下溢出的第(1)个情况
        // 可以统一为total > INT_MAX / 10.

        if ((total > INT_MAX / 10) || // 上溢出和下溢出的第一种情况
            (sign == 1 && total == INT_MAX / 10 && digit > INT_MAX % 10) || // 上溢出第二种情况
            (sign == -1 && total == (INT_MAX + 1) / 10 && digit > (INT_MAX + 1) % 10)) //下溢出第二种情况
        {
            return sign == 1 ? INT_MAX : INT_MIN;
        }

        total = total * 10 + digit;
    }

    return sign * total;
}
