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
>类型简介

|  是否为线性 | 具体类型 |介绍|
| ----------- | ----------- |--|
| 线性数据结构      | 数组       |可变数组基于数组和扩容机制实现，常用操作有：访问元素、添加元素、删除元素。|
||链表|链表以节点为单位，每个元素都是一个独立对象，在内存空间的存储是非连续的。|
||栈|栈是一种具有 「先入后出」 特点的抽象数据结构，可使用数组或链表实现|
||队列|队列是一种具有 「先入先出」 特点的抽象数据结构，可使用链表实现|
| 非线性数据结构   | 树        |树是一种非线性数据结构，根据子节点数量可分为 「二叉树」 和 「多叉树」|
||图|图是一种非线性数据结构，由「节点（顶点）vertex」和「边 edge」组成，每条边连接一对顶点。|
||散列表|利用 Hash 函数将指定的「键 key」映射至对应的「值 value」，以实现高效的元素查找|
||堆|堆分为「大顶堆」和「小顶堆」，大（小）顶堆：任意节点的值不大于（小于）其父节点的值|

## 1.[剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

>时间O(N):遍历字符串长度N  
空间O(N):保存字符串长度N的一个变量并返回

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

>时间O(N):N为链表的长度  
空间O(N):N为暂存数组的长度

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

>appendTail时间O(1):每次添加1个数字  
deleteHead时间O(N):最差情况是第一次删除，需要将输入栈中的N个数字全部放到辅助栈  
空间O(N):输入栈和辅助栈总共保存N个元素

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

>时间O(N):N为链表的长度  
空间O(1):pre，cur，temp都是常数个变量

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

>时间O(1):push(),pop(),top(),min()都是常数数量级操作  
空间O(N):N为输入值的数量，输入栈和非严格降序栈最多数量为2N

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