- [1. 两数之和](#1-两数之和)
- [2. 两数相加](#2-两数相加)
- [3. 无重复字符的最长子串](#3-无重复字符的最长子串)

# [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

> 时间O(N):遍历N次数组
>
> 空间O(N):最坏的情况下储存N个元素

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, num in enumerate(nums):
            if target - num in dic:
                return [dic[target - num], i]
            dic[num] = i
        return []
```


# [54. 螺旋矩阵](https://leetcode-cn.com/problems/spiral-matrix/)

> 时间O(MN):遍历M行N列的矩阵
>
> 空间O(MN):存储MN个元素

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        l, r, u, d = 0, len(matrix[0]), 0, len(matrix)  # 定义四个边的位置
        res = []
        while True:
            # 左向右
            for i in range(l, r): res.append(matrix[u][i])
            u += 1
            if u >= d:break
            # 上到下，因为r和d是len取的值在索引里是加一的，需要减一才能当作索引用
            for i in range(u, d): res.append(matrix[i][r - 1])
            r -= 1
            if l >= r:break
            # 右到左，因为range()取不到最后一位，所以l和u需要减一才能放进range()使用
            for i in range(r - 1, l - 1, -1): res.append(matrix[d - 1][i])
            d -= 1
            if u >= d:break
            # 下到上
            for i in range(d - 1, u - 1, -1): res.append(matrix[i][l])
            l += 1
            if l >= r:break
        return res

```


# [628. 三个数的最大乘积](https://leetcode-cn.com/problems/maximum-product-of-three-numbers/)

> 时间O(NlogN):排序时间
>
> 空间O(logN):排序占用的空间

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        nums = sorted(nums)  # sort默认升序，小在前大在后
        return max(nums[-3] * nums[-2] * nums[-1], nums[0] * nums[1] * nums[-1])
```
