- [1. 两数之和](#1-两数之和)
- [2. 两数相加](#2-两数相加)
- [3. 无重复字符的最长子串](#3-无重复字符的最长子串)

# [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

> 动态规划：一次遍历数组，同时储存遍历过数组的索引
>
> 哈希表：用于存储已经遍历过的数字的索引
>
> 时间复杂度O(N)：N为字符串的长度
>
> 空间复杂度O(N)：最多哈希表dic保存N个字符

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, num in enumerate(nums):
            item = target - num
            if item in dic:
                return [dic[item], i]
            dic[nums[i]] = i
        return -1
```

# [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        cur = dum = ListNode(0)
        carry = 0
        while (l1 or l2):
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            s = carry + x + y
            carry = s//10
            cur.next = ListNode(s%10)
            cur = cur.next
            if l1 != None:l1 = l1.next
            if l2 != None:l2 = l2.next
        if carry > 0:
            cur.next = ListNode(1)
        return dum.next
```

# [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

> 先排序，然后分两种情况，正正正(0)或者正负负

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        sort = sorted(nums)
        print(sort)
        return max(sort[-3] * sort[-2] * sort[-1], sort[0] * sort[1] * sort[-1])
```
