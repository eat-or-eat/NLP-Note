- [一.时间复杂度](#一时间复杂度)
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

> 写在前面：以下所有内容参考leetcode的krahets大牛,因为是总结，所以会整理的比较简陋，供自己复习用
>
> leetcode网址:https://leetcode-cn.com/leetbook/detail/illustration-of-algorithm/
>
> github网址:https://github.com/krahets/LeetCode-Book

# 一.时间复杂度

## 表示方法

时间复杂度指输入数据大小为 N 时，算法运行所需花费的时间。
常用的符号有:平均 $\Theta$ ，最佳 $\Omega(1)$，最差$O(N)$三种，一般使用最差$O(N)$来表示

## 常见种类

$O(1)<O(logN)<O(N)<O(NlogN)<O(N^2)<O(2^N)<O(N!)$

# 2.空间复杂度

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
