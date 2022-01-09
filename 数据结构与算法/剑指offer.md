- [一.复杂度](#一复杂度)
  - [1.时间复杂度](#1时间复杂度)
  - [表示方法](#表示方法)
  - [常见种类](#常见种类)
  - [2.空间复杂度](#2空间复杂度)
  - [空间来源](#空间来源)
  - [常见种类](#常见种类-1)
- [二.数据结构类型基础](#二数据结构类型基础)
  - [1.剑指 Offer 05. 替换空格](#1剑指-offer-05-替换空格)
  - [2.剑指 Offer 06. 从尾到头打印链表](#2剑指-offer-06-从尾到头打印链表)
  - [3.剑指 Offer 09. 用两个栈实现队列](#3剑指-offer-09-用两个栈实现队列)
  - [4.剑指 Offer 20. 表示数值的字符串](#4剑指-offer-20-表示数值的字符串)
  - [5.剑指 Offer 24. 反转链表](#5剑指-offer-24-反转链表)
  - [6.剑指 Offer 30. 包含 min 函数的栈](#6剑指-offer-30-包含-min-函数的栈)
  - [7.剑指 Offer 35. 复杂链表的复制](#7剑指-offer-35-复杂链表的复制)
  - [8.剑指 Offer 58 - II. 左旋转字符串](#8剑指-offer-58---ii-左旋转字符串)
  - [9.剑指 Offer 59 - I. 滑动窗口的最大值](#9剑指-offer-59---i-滑动窗口的最大值)
  - [10.剑指 Offer 59 - II. 队列的最大值](#10剑指-offer-59---ii-队列的最大值)
  - [11.剑指 Offer 67. 把字符串转换成整数](#11剑指-offer-67-把字符串转换成整数)
- [三，动态规划](#三动态规划)
  - [1.剑指 Offer 10- I. 斐波那契数列](#1剑指-offer-10--i-斐波那契数列)
  - [2.剑指 Offer 10- II. 青蛙跳台阶问题](#2剑指-offer-10--ii-青蛙跳台阶问题)
  - [3.剑指 Offer 19. 正则表达式匹配](#3剑指-offer-19-正则表达式匹配)
  - [4.剑指 Offer 42. 连续子数组的最大和](#4剑指-offer-42-连续子数组的最大和)
  - [5.剑指 Offer 46. 把数字翻译成字符串](#5剑指-offer-46-把数字翻译成字符串)
  - [6.剑指 Offer 47. 礼物的最大价值](#6剑指-offer-47-礼物的最大价值)
  - [7.剑指 Offer 48. 最长不含重复字符的子字符串](#7剑指-offer-48-最长不含重复字符的子字符串)
  - [8.264. 丑数 II](#8264-丑数-ii)
  - [9.剑指 Offer 60. n个骰子的点数](#9剑指-offer-60-n个骰子的点数)
  - [10.剑指 Offer 63. 股票的最大利润](#10剑指-offer-63-股票的最大利润)
- [四，搜索与回溯算法](#四搜索与回溯算法)
- [五，分治算法](#五分治算法)
- [六，排序](#六排序)
- [七.查找算法](#七查找算法)
- [八，双指针](#八双指针)
- [九，位运算](#九位运算)
- [十，数字](#十数字)
- [十一，模拟](#十一模拟)

> 写在前面：以下所有内容参考leetcode的krahets大牛,因为是总结，所以会整理的比较简陋，供自己复习用
>
> leetcode网址:https://leetcode-cn.com/leetbook/detail/illustration-of-algorithm/
>
> github网址:https://github.com/krahets/LeetCode-Book

# 一.复杂度

## 1.时间复杂度

## 表示方法

时间复杂度指输入数据大小为 N 时，算法运行所需花费的时间。
常用的符号有:平均 $\Theta$ ，最佳 $\Omega(1)$，最差$O(N)$三种，一般使用最差$O(N)$来表示

## 常见种类

$O(1)<O(logN)<O(N)<O(NlogN)<O(N^2)<O(2^N)<O(N!)$

## 2.空间复杂度

## 空间来源

输入空间： 存储输入数据所需的空间大小；
暂存空间： 算法运行过程中，存储所有中间变量和对象等数据所需的空间大小；
输出空间： 算法运行返回时，存储输出数据所需的空间大小；

## 常见种类

$O(1)<O(logN)<O(N)<O(N^2)<O(2^N)$

# 二.数据结构类型基础

> 类型简介


| 是否为线性     | 具体类型 | 介绍                                                                                    |
| ---------------- | ---------- | ----------------------------------------------------------------------------------------- |
| 线性数据结构   | 数组     | 可变数组基于数组和扩容机制实现，常用操作有：访问元素、添加元素、删除元素。              |
|                | 链表     | 链表以节点为单位，每个元素都是一个独立对象，在内存空间的存储是非连续的。                |
|                | 栈       | 栈是一种具有 「先入后出」 特点的抽象数据结构，可使用数组或链表实现                      |
|                | 队列     | 队列是一种具有 「先入先出」 特点的抽象数据结构，可使用链表实现                          |
| 非线性数据结构 | 树       | 树是一种非线性数据结构，根据子节点数量可分为 「二叉树」 和 「多叉树」                   |
|                | 图       | 图是一种非线性数据结构，由「节点（顶点）vertex」和「边 edge」组成，每条边连接一对顶点。 |
|                | 散列表   | 利用 Hash 函数将指定的「键 key」映射至对应的「值 value」，以实现高效的元素查找          |
|                | 堆       | 堆分为「大顶堆」和「小顶堆」，大（小）顶堆：任意节点的值不大于（小于）其父节点的值      |

## 1.[剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

> 时间O(N):遍历字符串长度N
>
> 空间O(N):保存字符串长度N的一个变量并返回
>
> 思路：顺序遍历字符串，遇到空格就添加%20否则添加原字符


```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        result = ''  # 字符串为不可变类型，需要新建一个变量
        for char in s:
            if char != ' ':  # 每次添加字符相当于重新赋值
                result += char
            else: result += '%20' 
        return result
```

## 2.[剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

> 时间O(N):N为链表的长度
>
> 空间O(N):N为暂存数组的长度
>
> 用一个暂存列表暂存每个节点的值，直到链表完全遍历完

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        result = []
        while head:  # 链表顺序访问的同时并保存
            result.append(head.val)
            head = head.next
        return result[::-1]
```

## 3.[剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

> appendTail时间O(1):每次添加1个数字
>
> deleteHead时间O(N):最差情况是第一次删除，需要将输入栈中的N个数字全部放到辅助栈
>
> 空间O(N):输入栈和辅助栈总共保存N个元素
>
> 一个栈做输入栈
>
> 另一个栈做删除栈，在转述句转移的时候，删除栈的数据就变成了先入后出

```python
class CQueue:

    def __init__(self):
        self.stack, self.reverse_stack = [], []  # 定义一个输入栈，一个提供转换顺序功能的辅助栈

    def appendTail(self, value: int) -> None:  # 添加到输入栈
        self.stack.append(value)

    def deleteHead(self) -> int:
        if self.reverse_stack: return self.reverse_stack.pop()  # 如果辅助栈有值，就弹出，没有就再输入栈里找
        if not self.stack: return -1  # 如果输入栈也没有就是空的，返回-1
        while self.stack:  # 将输入栈放到辅助栈
            self.reverse_stack.append(self.stack.pop())
        return self.reverse_stack.pop()


# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```

## 4.[剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

## 5.[剑指 Offer 24. 反转链表](定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。)

> 时间O(N):N为链表的长度
>
> 空间O(1):pre，cur，temp都是常数个变量
>
> 用双指针来标记None和首节点，然后暂存下一个节点，从定向当前节点，更新双指针，知道当前节点为空

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur: # 最后一个cur为None，pre为实际最后一个节点
            temp = cur.next  # 因为后面要重定向cur，所以需要暂存下一个节点
            cur.next = pre  # 重定向
        
            # 更新下一组节点
            pre = cur
            cur = temp
        return pre
```

## 6.[剑指 Offer 30. 包含 min 函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

> 时间O(1):push(),pop(),top(),min()都是常数数量级操作
>
> 空间O(N):N为输入值的数量，输入栈和非严格降序栈最多数量为2N
>
> 一个栈用来存放输入栈，另一个栈用来存放非严格降序最小栈

```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack, self.low_stack = [],[]  # 定义输入栈和非严格降序栈

    def push(self, x: int) -> None:
        self.stack.append(x)  # 添加元素
        if not self.low_stack or self.low_stack[-1] >= self.stack[-1]:
            self.low_stack.append(x)

    def pop(self) -> None:
        if self.low_stack[-1] == self.stack[-1]:
            self.low_stack.pop()
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.low_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```

## 7.[剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

> 时间O(N):两次遍历原始链表
>
> 空间O(N):残存空间含有2N个节点
>
> 两次遍历复杂链表，第一次构造{原节点：新节点值}对，第二次用来给新节点定向next和random值

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return None  # 0输入情况判断
        dic = {}  
    
        cur = head  # 构造暂存{原节点:新节点含节点值}字典
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
    
        cur = head  # 为新节点定义next和random指向
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        return dic[head]  # 返回原始头节点对应的新节点
```

## 8.[剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

> 时间O(N):N为输入字符串的长度
>
> 空间O(N):新旧字符串的长度都是N
>
> 根据n值进行列表相加

```python
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        res = []

        for i in range(n, len(s)):
            res.append(s[i])
        for i in range(n):
            res.append(s[i])
    
        return ''.join(res)
```

## 9.[剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

> 时间O(N):N为数组的长度，线性遍历用N次，每个元素入队和出队最多一共2N词，总共3N次
>
> 空间O(K):双端队列最多存储K个元素
>
> 一个单调队列用来存放输入值，另一个双端队列用来存放非严格降序的输入值

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = collections.deque()  # 暂存非严格递减窗口
        n = len(nums)  # 输入数组长度
        res = []  
        for i, j in zip(range(1 - k,n - k + 1), range(n)):
            # 删除可能滑过的最大最左边的数字，弹出最大值后还有次大值
            if i > 0 and deque[0] == nums[i - 1]: deque.popleft()
        
            # 保证deque非严格递减
            while deque and deque[-1] < nums[j]:  # 不用等于是因为非严格递减，防止前面删除导致没最大值
                 deque.pop()
            deque.append(nums[j])

            # 窗口进入的时候开始添加
            if i >= 0:
                res.append(deque[0])
        return res

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums: return []
        res = []
        for i in range(len(nums) - k + 1):
            res.append(max(nums[i:i +k]))
        return res

```

## 10.[剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

> max_value()时间O(1)
>
> push_back()均摊时间O(1)
>
> pop_front()均摊时间O(1)
>
> 空间O(N):最多两个队列一共保存2N个元素
>
> 一个单调队列用于存放输入值，一个双端队列用来存放非严格降序的输入值

```python
import queue

class MaxQueue:

    def __init__(self):
        self.queue = queue.Queue()  # 单调队列用于保存入队和出队的元素
        self.deque = queue.deque()  # 双端队列用于保存非严格递减的最大元素

    def max_value(self) -> int:  
        return self.deque[0] if self.deque else -1  # 如果没有值就返回-1

    def push_back(self, value: int) -> None:
        self.queue.put(value)  # 单调队列先添加
        while self.deque and self.deque[-1] < value: self.deque.pop()  # 双端队列将小于新添加的元素都右弹出
        self.deque.append(value)  # 双端队列后添加，这样就能保证非严格递减

    def pop_front(self) -> int:
        if self.queue.empty(): return -1  # 单调队列如果为空就返回-1
        value = self.queue.get()  # 这里已经弹出了单调队列队首，put()与get()两个方法
        if value == self.deque[0]:  # 如果弹出的值跟双端队列最大值相等就业一起弹出，不然不弹出，这样就能保存最大值了
            self.deque.popleft()
        return value
    


# Your MaxQueue object will be instantiated and called as such:
# obj = MaxQueue()
# param_1 = obj.max_value()
# obj.push_back(value)
# param_3 = obj.pop_front()
```

## 11.[剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof)

> 时间O(N):N个字符需要遍历
>
> 空间O(N):最多N个字符要保存在str变量上


```python
class Solution:
    def strToInt(self, str: str) -> int:
        str = str.strip()  # 去除前面空格
        INT_MAX, INT_MIN, BOUND = 2 ** 31 -1, -2 ** 31, 2 ** 31 // 10  # 用于记录最大最小值和越界值
        sign, res = 1, 0  # 用于记录符号标记和数字和
        i = 1 # 用于记录开始的index
  
        if str[0] == '-': sign = -1  # 如果为负号，则初始值只需要改sign
        elif str[0] != '+': i = 0  # 如果不是正号，则index改成1；另一种情况就是默认的正号

        for c in str[i:]:
            if not '0' <= c <= '9': break  # 碰到数字外的就跳出
            if res > BOUND or res == BOUND and c > '7': return INT_MAX if sign == 1 else INT_MIN  # 越界处理
            res = 10 * res + ord(c) - ord('0')  # 累加
        return sign * res

```

# 三，动态规划

> 动态规划简介

主要特点

包含了「分治思想」、「空间换时间」、「最优解」等多种基石算法思想

求解框架

状态定义： 构建问题最优解模型，包括问题最优解的定义、有哪些计算解的自变量；
初始状态： 确定基础子问题的解（即已知解），原问题和子问题的解都是以基础子问题的解为起始点，在迭代计算中得到的；
转移方程： 确定原问题的解与子问题的解之间的关系是什么，以及使用何种选择规则从子问题最优解组合中选出原问题最优解；
返回值： 确定应返回的问题的解是什么，即动态规划在何处停止迭代；


## 1.[剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

> 时间O(N):第N个数字需要遍历N次
>
> 空间O(1):常数个变量

```python
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            temp = a + b
            a = b
            b = temp % 1000000007
        return a
```


## 2.[剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

> 时间O(N):第N个台阶需要遍历N次
>
> 空间O(1):常数个变量

```python
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1,1
        for _ in range(n):
            temp = a + b
            a = b
            b = temp % 1000000007
        return a
```


## 3.[剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

> 时间O(MN):MN为dp矩阵遍历的次数，M行N列
>
> 空间O(MN):MN为矩阵的大小，M行N列


```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 定义所需变量
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]

        # 初始化矩阵
        dp[0][0] = True  # 空字符串能匹配
        for j in range(2, n, 2):  # 空字符和偶数的'*'能匹配
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'

        # 更新矩阵
        for i in range(1, m):
            for j in range(1, n):
                if p[j - 1] == '*':  # 分为是'*'和不是'*'的情况
                    if dp[i][j - 2]: dp[i][j] = True  # p的前一个出现0次的时候
                    if dp[i - 1][j] and s[i - 1] == p[j - 2]:dp[i][j] = True  # 出现多次的情况，p的前一个等于新的s
                    if dp[i - 1][j] and p[j - 2] == '.':dp[i][j] = True  # 出现过次的情况，p的前一个等于'.' 
                else:
                    if dp[i - 1][j - 1] and p[j - 1] == s[i - 1]:dp[i][j] = True  # 之前匹配的情况下，新的p等于新的s
                    if dp[i - 1][j - 1] and p[j - 1] == '.':dp[i][j] = True  # 之前匹配的情况下，新的p等于'.'
      
        return dp[-1][-1]
```


## 4.[剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

> 时间O(N):遍历N-1次列表
>
> 空间O(1):用原来的列表做动态规划，不用额外的空间

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1,len(nums)):
            nums[i] += max(nums[i - 1], 0)
        return max(nums)
```


## 5.[剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

> 时间O(N):遍历N-1次字符串
>
> 空间O(N):字符串占用num的N长度

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        a = b = 1
        for i in range(2, len(s) + 1):  # 2是因为第一个已经默认了，从前两个开始，加一是因为索引最后取不到
            temp = a + b if '10' <= s[i - 2:i] <= '25' else b
            a = b
            b = temp
        return b
```

## 6.[剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof)

> 时间O(MN):遍历M行N列
>
> 空间O(1):常数个额外变量

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])  # M行N列
      
        for i in range(1, m):
            grid[i][0] += grid[i - 1][0]
        for j in range(1, n):
            grid[0][j] += grid[0][j - 1]
      
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += max(grid[i - 1][j], grid[i][j - 1])
        return grid[-1][-1]
```



## 7.[剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

> 时间O(N):遍历N个字符
>
> 空间O(N):最差N个字符都要保存在dic中

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res = temp = 0
        for j in range(len(s)):
            i = dic.get(s[j], -1)  # 获取曾是否经存在过的位置
            dic[s[j]] = j  # 更新位置
            temp = temp + 1 if temp < j - i else j - i  # 如果当前最长长度小于新出现的长度就加一，否则负值成现在长度
            res = max(temp, res)  # 保存最长长度

        return res
```


## 8.[264. 丑数 II](https://leetcode-cn.com/problems/ugly-number-ii/)

> 时间O(N):遍历N次循环
>
> 空间O(N):使用N个dp长度

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * n
        a, b, c = 0, 0, 0
        for i in range(1, n):  # 第一个丑数默认为1，所以从1开始
            n2, n3, n5 = dp[a] * 2,dp[b] * 3,dp[c] * 5,
            dp[i] = min(n2, n3, n5)
            if dp[i] == n2:a += 1
            if dp[i] == n3:b += 1
            if dp[i] == n5:c += 1
        return dp[-1]
```


## 9.[剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)


## 10.[剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

> 时间O(N):遍历N个价格
>
> 空间O(1):常数个额外变量

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        cost = float('inf')
        profit = 0
        for price in prices:
            cost = min(cost, price)
            profit = max(profit, price - cost)
        return profit
```

# 四，搜索与回溯算法

# 五，分治算法

# 六，排序

# 七.查找算法

# 八，双指针

# 九，位运算

# 十，数字

# 十一，模拟
