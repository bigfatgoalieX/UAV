# Assignment 1

### 211220166 王诚昊

## 1.

**动作值函数：**
$$
\begin{align*}
q_{\pi}(1|1) &= 1.0 \times (-1 + v_{\pi}(1))\\
q_{\pi}(2|1) &= 0.9 \times (-1 + v_{\pi}(4)) + 0.1 \times (-1 + v_{\pi}(1))\\
q_{\pi}(3|1) &= 1.0 \times (-1 + v_{\pi}(1))\\
q_{\pi}(4|1) &= 0.9 \times (-1 + v_{\pi}(2)) + 0.1 \times (-1 + v_{\pi}(1))\\
\end{align*}
$$
**状态值函数：**
$$
v_{\pi}(1) = \sum_{a = 1}^{4} \pi(a|1) \times q(a|1) \\
\begin{align*}
= 0.25 &\times 1.0 \times (-1 + v_{\pi}(1)) + \\
0.25 &\times 0.9 \times (-1 + v_{\pi}(4)) + \\
0.25 &\times 0.1 \times (-1 + v_{\pi}(1)) + \\
0.25 &\times 1.0 \times (-1 + v_{\pi}(1)) + \\
0.25 &\times 0.9 \times (-1 + v_{\pi}(2)) + \\
0.25 &\times 0.1 \times (-1 + v_{\pi}(1))  \\
\end{align*}
$$

## 2.

1. 原代码运行结果：![image-20240303235120057](D:\NJU_undergraduate\大三下\无人机\homework\Assignment1\image-20240303235120057.png)

2. 修改代码：

   修改键值对定义：

   ```python
   # returns = {k: [] for k in range(1, 10)}
   returns = {(x,y): [] for x in range(1, 10) for y in range(1,5)}
   ```

   变量是之前4倍，模拟次数也乘4很合理吧：

   ```python
   # for episode in range(10000):
   for episode in range(40000): 
   ```

   ```python
   # state_seq.append(s)
   state_seq.append((s,a))
   ```

   改善输出格式：

   ```python
   #to make 4 output per line
   if cnt > 3:
       print()
       cnt = 0
   cnt += 1
   ```

   运行结果：![image-20240304015555588](D:\NJU_undergraduate\大三下\无人机\homework\Assignment1\image-20240304015555588.png)

3. 对于确定性策略$best-policy$，蒙特卡罗方法能够有效评估状态值函数$V_{\pi}$，但是不能够有效评估动作值函数$Q_{\pi}$。因为在确定性策略下某些动作（$\frac{3}{4}$的动作）完全不会执行，因而缺少(s,a)的样本。

4. 根据第二问输出，改进后的策略如下：

   ```python
   def best_policy(s):
       p = {
           1: 4, 2: 4, 3: 4,
           4: 4, 5: 1, 6: 1,
           7: 4, 8: 1, 9: 1
       }
       return p[s]
   ```

   