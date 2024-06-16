### 动态规划（dp)

#### 简述

​	思想：通过不重复的计算来得到某个问题的最优解，解决各种**离散优化问题**，这些问题中往往有很多解，每个解都会有一个值，我们称这种解为该问题的最佳解之一。（可能有多个最优解都达到了最优值，比如说找零问题）

> [!IMPORTANT]
>
> - **刻画最优解的结构特性**
>
>   P(X) 例如:P(n)，P(n,w)
>
> - **将问题划分为子问题**
>
>   P(X) = ![image-20240518150647470](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240518150647470.png)（f（P（X-A1)，...，P(X-Ad) ) )
>
>   通常而言 ![image-20240518150750506](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240518150750506.png)是max{} 或者 min{}
>
>   比如说:
>
>   斐波那契数：F(n) = F(n-1) + F(n-2)
>
>   找零问题：M(n) = min{1 + M(n-di)}	
>
> - **自底而上计算**
>
>   P(X) = ![image-20240518150647470](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240518150647470.png)（f（P（X-A1)，...，P(X-Ad) ) )
>
>   从子问题算到最后的问题上
>
> - **注意初值**



#### 与其他算法策略的区别

​	**贪婪策略**： 逐步建立一个解决方案，每一步都“目光短浅”地选择优化一些局部目标（选取的是局部的最优解）

​	**分治策略**： 将一个问题分解为多个**不相交**的子问题，独立解决每个子问题，并将子问题的解结合起来形成原问题的解。

​	**动态规划**： 将一个问题分解称一系列**相互存在重叠**的子问题，并不断由子问题的解形成越来越大的问题的解



#### 一些列子

> [!NOTE]
>
> 直接总结算法的思路，因为题目的背景都是上课讲过的，这里就是再次回顾一下解决这些问题的算法思路。

##### EX1.最长单调子序列

​	令S[1]S[2]S[3]....S[n]表示输入的序列

​	令L(i) (1 ≤ i ≤ n)表示**以S[i]结束**的**最长**单调递增子序列的长度

​	-子序列的最后一项为S[i]

​	-此时**只需考虑**前i项 S[1]S[2]....S[i]的**以S[i]结束**的**最长**单调子序列即可

总目标：max{L(1),...L(n)}（每一个子序列总会有一个“最后一项”的）

目标函数的递归关系：
$$
L(i) = \begin{cases}
\displaystyle \max_{1≤j<i 且 S[j]<S[i]} &L(j) + 1, & \text {若存在j使得1≤j<i且S[j]<S[i]} \\
1, & \text{否则}
\end{cases}
$$

```
输入：序列
输出：最长单调子序列长度Len
1.for i = 1 to n do 
2.	  L(i) ← 1
3.	  for j = 1 to i-1 do
4.		 if S[j] < S[i] and L[j] ≥ L[i] do
5.			L[i] ← L[j] + 1 (,P(i) ← j)
6.Len ← max{L(1),...,L(n)}	//假设最大值为L(k)
7.i ← 1 //回溯
8.j ← k
9.do
10.		T(i) ← S[j],i ← i+1, j ← P(j)
11.until j = 0;
12.output Len以及T的反序
```

然后设置一个标记函数P[i]去回溯每一个位置

比如现在S = 1,8,2,9,3,10,4,5

| i    | S    | L[i] |               | P[i] |
| ---- | ---- | ---- | ------------- | ---- |
| 1    | 1    | 1    | 1             | 0    |
| 2    | 8    | 2    | **1**,8       | 1    |
| 3    | 2    | 2    | **1**,2       | 1    |
| 4    | 9    | 3    | 1,**8**,9     | 2    |
| 5    | 3    | 3    | 1,**2**,3     | 3    |
| 6    | 10   | 4    | 1,8,**9**,10  | 4    |
| 7    | 4    | 4    | 1,2,**3**,4   | 5    |
| 8    | 5    | 5    | 1,2,3,**4**,5 | 7    |



##### EX2.最大子段和问题

 描述：就是一段序列，然后我们要求里面和最大的子段

-例如

![image-20240522233356815](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240522233356815.png)

所以我们的**目标函数**是：

![image-20240522233531787](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240522233531787.png)

**这个可以看成灰度图：**

![image-20240522233608107](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240522233608107.png)

**思路：**

​	**-蛮力法？**

​	不太可行，会有n²/2个不同的子段和

​	**-Kadane算法**

​	令C(j)表示必须以元素aj结尾（因此长度至少为1）的最大子段和（允许为负数值）
$$
即C(j) = \displaystyle \max_{1≤j<i}\{\sum_{k=1}^ja_k\}
$$
​	此时就会有**两种**情况：

​	**1.长度为1，即仅仅包含a_j**

​	**2.长度大于1**

![image-20240523000637537](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240523000637537.png)

​	**所以可以得出C(j)的递推公式**：
$$
C(j) = \begin{cases} a_j + C(j-1) & \text {if C(j-1)>0} \\a_j & \text{else}\end{cases}
$$

```
Alhorithm MaxSum(A,n)
输入：数组/序列A，长度n
输出：最大子段和
1.current_sum ← 0
2.best_sum ← 0
3.for j = 1 to n do
4. 	  if current_sum > 0 then #如果现在的和大于零
5. 		 current_sum ← current_sum + aj
6. 	  else current_sum ← aj	  #如果和小于等于零就重新开始赋值
7. 	  if current_sum > best_sum then #更新最大值
8. 		 best_sum ← current_sum
9. return best_sum
```

> [!NOTE]
>
> 如果还要得到具体的子段，还需要进行标记。
>
> 在第6行的时候更新start位置，在第5行的时候更新end的位置。



##### EX3.0-1背包问题

**背包问题**是一个很典型的组合优化问题

​	假设共有n种物品，其中第i种物品的价值为vi，重量为wi（假定vi和wi都是整数）

​	确定要从这n种物品中选择哪些种、每种选择多少，将其装入背包里，使得这些物品的总重量不超过给定的限制**W**，并且总价值尽可能大

**0-1背包问题描述：**

​	对于每种物品，当在考虑是否将其装入背包时，要么全部装入背包，要么全部不装入背包，而不能只装入物品i的一部分

> [!IMPORTANT]
>
> 凡是论及“背包问题”时，将“体积”和“重量”视作同一个属性，“收益”和“价值”含义相同

贪婪策略的话，没法得到最优解。

所以用动态规划：

考虑子问题P(k,y)——只使用前k种物品，总重量不超过y

令F(k,y)表示仅使用前k种物品且背包容量为y时的最优解的总价值

> [!NOTE]
>
> 即子问题P(k,y)问题的最优解
>
> 原问题的目标为F(n,W)

于是在选择的时候会有**两种**情况：

![0-1背包问题](C:\Users\cxh1015\Desktop\Course\Algorithm\0-1背包问题.png)

**得到递推式：**

![image-20240523233311162](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240523233311162.png)

**填起来的表格长这样：**

![image-20240523233415104](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240523233415104.png)

![image-20240523233350994](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240523233350994.png)

**伪代码：**

```伪代码
输入：n,w1,...,wn,v1,...,vn,W				时间复杂度：O(nW)
1.for y = 0 to W 			#重量上限为0，赋值为0
2.	F(0,y) ← 0		
3.for k = 1 to n			#什么物品都没有放，赋值为0
4.	F(k,0) ← 0
5.for k = 1 to n
6.	for y = 1 to W
7.		if(wk > y)			#如果该物品的重量超过上限
8.			F(k,y) ← F(k-1,y)
9.		else				#如果该物品的重量没有超过上限
10.			F(k,y) ← max{F(k-1,y),vk+F(k-1,y-wk)} #递推关系
11.return F(n,W)
```

> [!NOTE]
>
> 时间复杂度进一步降低，可参考：
>
> https://zhuanlan.zhihu.com/p/30959069



##### EX4.投资问题

> [!TIP]
>
> 可以和背包问题类比起来看

问题描述：

​	有m元，n项可能的投资项目，fi(x)表示将x元投入第i个项目获得的利益

> [!NOTE]
>
> x是非负整数
>
> f_i(x)值非负
>
> f_i(x)关于x是不减函数

**目标**：将投资的利益**最大化**

**即：**
$$
max {\{f_1(x_1) + f_2(x_2)+...+f_n(x_n)}\}
$$

$$
s.t.x_1+x_2+...+x_n = m
$$

**思路**:

用F_k(x)表示前k个项目进行共计x元的投资所能取得的最大收益

x_k(x)表示在F_k(x)中投资给项目k的钱数

考虑”**最后一个**“项目

![投资问题](C:\Users\cxh1015\Desktop\Course\Algorithm\投资问题.png)

**递推关系式**：
$$
F_k(x)=\displaystyle\max_{0≤x_k≤x}\{f_k(x_k)+F_{k-1}(x-x_k)\},0≤x≤m,2≤k≤n
$$

$$
F_1(x)=f_1(x),0≤x≤m
$$

**填表**：

（填入F_4时的情况）

![image-20240524233707505](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240524233707505.png)

（完整的填表）

![image-20240524233757004](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240524233757004.png)

> [!tip]
>
> **从前往后**依次填表，然后用递推关系进行判断
>
> 与背包问题相似，但是背包问题的初值的处理方法和投资问题还是不太一样的。
>
> 背包问题是重量和价值为0均赋值为0，这样方便后面去运算；**但是**投资问题是投资金额为0全部赋值为0，初始值为f_1的投资回报。

```
1.for y = 1 to m
2.	F1(y) ← f1(y)	#只投资第一个物品，赋初值
3.for k = 2 to n 	#从投资第二件物品开始计算收益
4.	for y = 1 to m
5.		Fk(y) ← max_0≤xk≤y{fk(xk) + Fk-1(y-xk)} #递推关系
6.return Fn(m)
```

**时间复杂度**:

计算 F_k(y) (2 ≤ k ≤ n,1 ≤ y ≤ m) 时候需要y+1次加法和y次比较

![image-20240524234908293](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240524234908293.png)





> [!NOTE]
>
> **序列的比较**

##### EX5.最长公共子序列（LCS）

**描述**：

​	子序列和子串不一样，序列只要相对的顺序不变就行。然后找到两个序列之间最长的子序列。

> [!NOTE]
>
> 序列的长度指的是序列的项数
>
> 空序列是任意两个序列的公共子序列
>
> 任一个序列和空序列的最长公共子序列都是空序列

​	**最长公共子序列可能不唯一**

**应用**：

​	度量两个序列的相似程度

**刻画方式**：

![image-20240525224504789](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525224504789.png)

**LCS的最优子结构（定理）**：

​	三种情况：

![image-20240525224557471](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525224557471.png)

​	最后两个相等、不相等（x与z相等，y与z相等）

> [!IMPORTANT]
>
> 如果用递归的思想会很慢，因为会有很多重复计算的地方
>
> 所以考虑按特定的次序计算 m x n个 len (i,j)

**伪代码**：

``` 
1.for i = 1 to m do len(i,0) ← 0		时间复杂度：O(mn)
2.for j = 0 to n do len(0,j) ← 0
3.for i = 1 to m do
4.	for j = 1 to n do
5.		if X[i] = Y[j] then  #如果两个相等
6.			len(i,j) ← len(i-1,j-1) + 1
7.			b(i,j) ← 1		//往左上方走
8.		else if len(i-1,j) ≥ len(i,j-1) then #比较左边和上边的子序列长度，取较长的那段子序列
9.			len(i,j) ← len(i-1,j)
10.			b(i,j) ← 2		//往上走
11.		else len(i,j) ← len(i,j-1) #这里同样取较长的那段子序列
12.			b(i,j) ← 3		//往左走
13.return len(m,n) 以及 b
```

**具体填表**：

（整张表从左上往右下填写，然后比较最后一个字段）

![image-20240525225742197](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525225742197.png)

![image-20240525225929933](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525225929933.png)

> [!NOTE]
>
> 数组b的作用就是用来回溯最长子序列具体的内容是什么。
>
> 回溯算法就是遇到左上的方向，然后记录字母，其他的根据方向继续遍历。

![image-20240525230052417](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525230052417.png)

（具体的伪代码就用老师ppt里的图来表示了）

![image-20240525230127608](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240525230127608.png)



##### EX6. 最短公共超序列（SCS）

**定义**：

​	设X和Y是两个序列。如果X和Y都是Z的子序列，那么称Z是X和Y的公共超序列（common supersequence）

> [!NOTE]
>
> 1.X和Y肯定有一个最短的公共超序列
>
> 2.任一个序列和空序列的最短公共超序列是序列自身

**递推关系式**：

​	令 *len[i,j]* 表示 *X[1..i]* 和 *Y[1..j]*的最短公共超序列的长度，于是会有如下的公式。

​	![image-20240526170819574](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240526170819574.png)

> [!TIP]
>
> 和最长公共子序列的思路是有相似之处的，比如从前往后找，然后都是比较最后一个是否一样，**但是**处理的方法略有不同，最长公共子序列是找前面最长的然后直接赋值，最短公共超序列就要从前面找最小还要加一（因为是超序列嘛，要把两个序列中的内容都涵盖进去）

**伪代码**：

```
1.for i=0 to n do 	#赋初值，和空序列进行比较
2.	len[i,0] = i
3.for j-0 to m do	#赋初值，和空序列进行比较
4.	len[0,j] = j
5.for i=1 to n do 
6.	for j=1 to m do
7.		if(X[i] = Y[j]) then	#两个字段相同
8.			len[i,j] ← len[i-1,j-1] + 1
9.		else					#两个字段不相同
10.			len[i,j] ← min{len[i-1,j]+1,len[i,j-1]+1} #找最小 
11.return len[n,m]
```

**填表**：

![image-20240526171659790](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240526171659790.png)

**回溯算法**：

```
1.for i=0 to n do 	#赋初值，和空序列进行比较
2.	c[i,0] = i
3.for j-0 to m do	#赋初值，和空序列进行比较
4.	c[0,j] = j
5.for i=1 to n do 
6.	for j=1 to m do
7.		if(X[i] = Y[j]) then c[i,j] ← c[i-1,j-1]+1, b[i,j] ← 1
8.		else					#两个字段不相同
9.			c[i,j] ← min{c[i-1,j]+1,c[i,j-1]+1} #找最小
10.			if(c[i,j] = c[i-1,j]+1) then b[i,j] ← 2 #记录来源
11.			else b[i,j] ← 3
12.p ← n, q ← m, k ← 1 	#回溯超序列
13.while(p≠0 or q≠0)
14.		if(b[p,q] = 1) then {SCS{k}←y[p],k←k+1,p←p-1,q←q-1}
15.		if(b[p,q] = 2) then {SCS{k}←x[p],k←k+1,p←p-1}
16.		if(b[p,q] = 3) then {SCS{k}←y[q],k←k+1,q←q-1}
#最后再反向输出SCS[]就得到了最短公共超序列
```

**LCS和SCS之间的联系**：

![image-20240526172357046](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240526172357046.png)



##### EX7.序列对齐与编辑距离

**多相似**：错误匹配和缺漏。

**编辑距离**：

就是给缺漏一个惩罚值，错误匹配一个惩罚值。

![image-20240527233459444](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240527233459444.png)

****

**Levenshtein距离**

![image-20240527234956018](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240527234956018.png)

（操作最少的编辑操作个数）

> [!NOTE]
>
> 用序列对齐去找
>
> ![image-20240529222936346](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529222936346.png)
>
> ![image-20240529223034940](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529223034940.png)

**伪代码**：（序列对齐：算法）

![image-20240529223551981](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529223551981.png)

时间和空间复杂度：O(mn)

![image-20240529223629040](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529223629040.png)

![image-20240529223648908](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529223648908.png)

> [!NOTE]
>
> 编辑距离是对LCS和SCS的一个应用，缺漏是SCS带来的，LCS是匹配上的，从而得到了（在该题下简化的）编辑距离。



##### EX8.矩阵链乘积（真有点难）

**背景**：

- 假定给定了矩阵的序列A1,A2,....,An
  - 其中Ai为Pi-1Pi阶矩阵
  - 于是Ai-1和Ai都是可以进行乘法的
- 目的是计算它们的链乘积A1A2...An
  - 然而每次**只**能是两个矩阵相乘得到第三个矩阵
- 由于矩阵乘法满足**结合律**，因此*无论计算的过程是什么样的*，其最终结果都是一样的

> [!NOTE]
>
> **但是**不同的计算过程的效率可能是**不同**的

先来看一下两个矩阵相乘的复杂度

- 采用“教科书算法”

![image-20240529224357579](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240529224357579.png)

于是我们可以看到，总的元素乘法次数为pqr

```
输入：矩阵A_pXq 和 B_qXr (维数分别是pXq和qXr)
输出：矩阵C_pXr = A·B
MATRIX-MULTIPLY(A_pXq,B_qXr)
1.for i ← 1 to p
2.	for j ← 1 to r
3.		C(i,j) ← 0
4.		for k ← 1 to q
5.			C(i,j) ← C(i,j) + A(i,k)·B(k,j)
6.return C
```

**问题定义**：

- 给定了矩阵的序列A1,A2,...An,其中Ai的阶数为Pi-1XPi
- 试确定矩阵相乘的次序（即，加括号的方法）使得**矩阵元素相乘的总次数最少**

> [!NOTE]
>
> **不是真的计算乘积**，而是它们相乘的次数

**思路**：

也是先考虑各种方法：

- 蛮力法？x 卡特兰数，数据量大了不可能找到解 时间复杂度为O(4^n)
- **每次选取最多元素乘法次数**的两个矩阵相乘？x
  - 反例：<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530115235716.png" alt="image-20240530115235716" style="zoom:67%;" />

- **每次选取最少元素乘法次数**的两个矩阵相乘？x
  - 反例：<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530115330350.png" alt="image-20240530115330350" style="zoom:67%;" />

- 找递推式：

  - 考虑**最后一次**乘法
    - 它将矩阵序列划分为两部分
  - 例子：<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530115431109.png" alt="image-20240530115431109" style="zoom:50%;" />

  - 将Ai·Ai+1·...·Aj的乘积记为Ai..j（j ≥ i）
    - 其中共有 l = j - i + 1 个矩阵 

  - 令m(i,j)表示计算Ai..j的最优方式的元素乘法总次数
  - 当 j = i 时，m(i,j) = 0
  - 假设计算Ai..j（j > i）的最后一次矩阵乘法是

$$
A_{i..j} = A_{i..k} *A_{k+1..j}
$$

其中 i ≤ k ≤ j

> [!NOTE]
>
> Ai..k的阶数为Pi-1 X Pk, Ak+1..j的阶数为Pk Pi-1 X Pk
>
> 这次矩阵乘法的**开销**是Pi-1 X Pk X Pk
>
> 如果**在固定k的前提下**希望将总的元素乘法总次数降到最少，那么之前的每一次乘法就应该按照**最优方式**计算Ai..k和Ak+1..j
>
> （因为每次都是最优方式才能让最后的解也为最优解）
>
> 之前计算的Ai..k和Ak+1..j的最优方式的开销分别是m(i,k)和m(k+1,j)
>
> 于是总开销的最小可能就是
> $$
> m(i,k) + m(k+1,j) + P_{i-1}*P_k*P_j
> $$

​	但是我们怎么知道Ai..j的最优方案的最后一次矩阵乘法发生在**哪里**？即，k的值为多少

​	无论如何，最后一次乘法**必然会发生**在某个Ak“后面”

​	于是就把所有的 i ≤ k < j **都**试试看，然后选择其中“**最少**“的

最后得到递推式：
$$
m(i,j) = \begin{cases} 0 & \text {if i = j} \\\displaystyle \min_{i≤k<j}\{m(i,k)+m(k+1,j)+P_{i-1}*P_k*P_j\}  & \text{if i < j}\end{cases}
$$
如果采用**递归**的方式：

![image-20240530121122877](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530121122877.png)

因此采用**动态规划**：

![image-20240530121146179](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530121146179.png)

填写**最优开销表**m和**划分表**s

> [!NOTE]
>
> s(i,j) = 0
>
> s(i,j) (j > i) 表示计算Ai..j 的最优方法的最后一次矩阵乘法发生的位置，即之前提及的k值

**伪代码**（重要）：

```
MATRIX-CHAIN-ORDER
输入：序列P0，P1,P2,...,Pn
输出：最优开销表m和划分表s
1.for i = 1 to n
2.    m(i,i) ← 0, s(i,i) ← 0 //将斜对角赋值为0
3.for l = 2 to n
4.	  for i = 1 to n - l + 1 //从左上往右下角填写
5.		  j ← i + l - 1
6.		  m(i,j) ← ∞
7.		  for k = i to j - 1
8.			  q ← m(i,k) + m(k+1,j) + P(i-1)·P(k)·P(j)
9.			  if(q < m(i,j))
10.			     m(i,j) ← q, s(i,j) ← k
11.return m 和 s
```

> [!IMPORTANT]
>
> **在复习的时候要注意好，i，j，k是如何遍历的，然后动态规划的填表时怎么进行的**

一些图示：

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122143069.png" alt="image-20240530122143069" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122219905.png" alt="image-20240530122219905" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122200194.png" alt="image-20240530122200194" style="zoom:50%;" />

**MORE**:

![image-20240530122331015](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122331015.png)



##### EX9.最优二叉查找树

**二叉查找树**：

![image-20240530122428835](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122428835.png)

**问题描述**

![image-20240530122450201](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122450201.png)

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122507319.png" alt="image-20240530122507319" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122521259.png" alt="image-20240530122521259" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122536918.png" alt="image-20240530122536918" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122547665.png" alt="image-20240530122547665" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122559206.png" alt="image-20240530122559206" style="zoom:50%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122609028.png" alt="image-20240530122609028" style="zoom:50%;" />

通过上面这个例子我们可以看到，我们**不是**试图去构造一棵平衡树，而是要优化道路的**加权**长度

而且，此时**内部节点也包含键值**，并因”查找“而**固定**了树中节点的中序次序，因此和Huffman编码的问题背景和使用条件不同

基本原则：**具有更大访问概率的键值的顶点应该更接近于根**

**思路**：

1.蛮力法？x 还是卡特兰数，得不出解

2.贪婪策略？将查找概率最大的键值作为根？x 反例：

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530122918261.png" alt="image-20240530122918261" style="zoom:67%;" />

3.递推关系：

记T(i,j)表示对应于键值ki,...,kj（及其对应的查找概率）的最优二叉查找树

- 此时**不**要求pi+..+pj = 1

并令C(i,j)表示此时成功查找的（最优）期望开销（比较次数）

i > j时，T(i,j)为空树，C(i,j) = 0

考虑T(i,j)的**根**

如果根的键值时ks(i ≤ s ≤ j)，那么其左子树键值为ki~ks-1,右子树键值为ks+1~kj

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530123234772.png" alt="image-20240530123234772" style="zoom:50%;" />

于是此时

C( i , j ) = ps X 1 +

​		cost_left + (pi + ... + ps-1) X 1 +

​		cost_right + (ps + ... + pj) X 1

（cost_left 和 cost_right 是暂定名它们彼此的计算是独立）

C( i , j ) = (pi + .... + pj) + cost_left + cost_right

（cost_left 和 cost_right  希望和达到最优（最小）值）

 **回过来**

如果根的键值是ks,那么其左子树**一定**是T(i,s-1)，右子树**一定**是T(s,j) (i ≤ s ≤ j)

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530123718161.png" alt="image-20240530123718161" style="zoom:50%;" />

此时

C( i , j ) = C( i , s - 1 ) + C(s+1, j) + ( pi + ... + pj )

之后就是对于**所有可能**的s计算其最小值

得到**递推关系**：
$$
C(i,j) = \displaystyle \min_{(i≤s≤j)}\{C(i,s-1)+C(s+1,j)+(P_i+...+P_j)\}
$$
初值 i > j时，C( i , j ) = 0

令 s(k) = p1+...+ pk , s(0) = 0 (s(k)表示第k个数后的所有概率总和)

- 可以用O(n)时间计算每一个s(k)

则可以使用s(j) - s(i-1) 计算（pi+...+pj）

![image-20240530124626443](C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530124626443.png)

##### EX10.跳棋棋盘

考虑一个nXn的方格棋盘

第i行第j列的方格的开销为c(i,j)

下图为5x5的棋盘示例 c(1,3) = 5

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530133404013.png" alt="image-20240530133404013" style="zoom: 33%;" />

现在有一个棋子只能向左前方、正前方或者右前方跳一个，然后找总开销最小的方案。

定义函数q（i，j）为

​	q（i，j）＝ 到达方格（i，j）的最小总开销

目标就是计算
$$
min_{1≤j≤n}\{q(n,j)\}
$$
很容易就能得到：

q(A) = min{q(B),q(C),q(D)} + c(A)<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530133750905.png" alt="image-20240530133750905" style="zoom:67%;" />

**伪代码**：

```
ComputeShortestPathArrays //把每一条格的开销计算出来
1.for x = 1 to n
2.	  q(1,x) ← ∞	//给初始的第一行赋值
3.q(1,(n+1)/2) ← c(1,x) //中间起始位置附上对应的成本
4.for y = 1 to n 	//把最左和最右的两列赋值成∞，为了后面方便计算
5.	  q(y,0) ← ∞
6.	  q(y,n+1) ← ∞
7.for y = 2 to n
8.	  for x = 1 to n
9.		  m ← min{q(y-1,x-1),q(y-1,x),q(y-1,x+1)}
10.		  q(y,x) ← m + c(y,x)
11.		  if m = q(y-1,x-1) then  //记录道路
12.			 p(y,x) ← -1 
13.		  else if m = q(y-1,x) then 
14.			 p(y,x) ← 0
15.		  else
16.			 p(y,x) ← 1
```

```
ComputeShortestPath
1.Call ComputeShortestPathArrays
2.minIndex ← 1 //总开销最小的列号
3.min ← q(n,1) //总开销的最小值
4.for i = 2 to n
5.	  if q(n,1) < min then
6.		 min ← q(n,i)
7.		 minIndex ← i
8.Call PrintPath(n,minIndex)

PrintPath(y,x)		//自上而下的打印各行的列号
1.print(x)
2.print("<-")
3.if(y = 2) then
4.	print(x+p(x,y))
5.else
6.	Call PrintPath(y-1,x+p(y,x))
```



#### 总结 ####

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135149454.png" alt="image-20240530135149454" style="zoom:67%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135204406.png" alt="image-20240530135204406" style="zoom:67%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135230267.png" alt="image-20240530135230267" style="zoom:67%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135240712.png" alt="image-20240530135240712" style="zoom:67%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135250939.png" alt="image-20240530135250939" style="zoom:67%;" />

<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240530135300736.png" alt="image-20240530135300736" style="zoom:67%;" />

**动态规划的基本要素**:

- 一个最优化多步决策问题适合用动态规划法求解有两个要素：**最优字结构特性**和重叠子问题。

- 最优子结构
  - 一个最优决策序列的任何子序列本身一定是相对于子序列的初始和结束状态的最优的决策序列
  - 一个问题的最优解总是包含所有子问题的最优解
  - 但**不是**说：如果有所有子问题的最优解，就可以**随便**把它们组合起来得到一个最优的解决方案
- 例如1、5找零问题
  - 凑成8元的最优解是5+1+1+1
  - 凑成9元的最优解是5+1+1+1+1
  - 但是凑成17元的最优解并**不是**5+1+1+1+5+1+1+1+1
  - 然而，的确**有一种方法**可以把凑成17元的问题的最优解分解为**子问题的最优解的组合**（例如，凑成15元+凑成2元）
  - 所有，找零问题满足最优子结构
- 但**不是**所有问题都满足最优子结构
- 【例】求总长模10的最短道路<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240604132858113.png" alt="image-20240604132858113" style="zoom:50%;" />
- <img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240604132948103.png" alt="image-20240604132948103" style="zoom:50%;" />
- 再如：**最长简单道路问题**（出租车敲竹杠问题）
- 右图中，从A到D的最长简单道路是ABCD<img src="C:\Users\cxh1015\AppData\Roaming\Typora\typora-user-images\image-20240604133021363.png" alt="image-20240604133021363" style="zoom:50%;" />
- 但是，子道路AB**不是**从A到B的最长简单道路（ABC更长）
- 这个问题**不满足**最优性原则
- 因此，最长简单道路问题**不能**用动态规划方法解决
