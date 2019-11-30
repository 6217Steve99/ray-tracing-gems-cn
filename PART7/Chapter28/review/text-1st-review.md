# 28章  追踪非均匀体积[a]

## 摘要
模拟光在介质中，散射和吸收地交互需要对距离进行重要性采样，此距离与体积透光率[@note3]成比例。一个起源于中子传输模拟的简单方法可以用于对一个粒子的碰撞事件进行重要性采样，就像一个光子与任意的介质进行碰撞。

## 28.1 光在体积中的传输
当光沿着射线穿过一个体积[@note4]，根据介质的光线交互属性，有一些会被散射或者吸收。这个模型是由介质的散射系数(scattering)σ和吸收(absorbing)系数α所建立的。通常来说，这两者都是随位置变化的函数。将两者相加，我们得到了消光(extinction)系数[@note5]κ = σ + α，它描绘了因（外）散射和吸收引起的总损失。

在指定距离s内，未被吸收的光与入射光的比例叫做体积透射率. 体积透射率用Beer-Lambert法则来描述：如果我们跟随一条射线，它的起始点在**o**，方向为**d**，则它的[@note6]透射率为

![](./formula-28-1.JPG)

这个术语在光传输的积分方程中占有突出地位[@note7]。举例来说，沿着射线散射的距离s的辐射度[@note8]是由对透射率加权的向内散射辐射率进行积分给出的（根据散射系数 σs 和相位函数 fp）:

![](./formula-28-2.JPG)

一个典型的蒙特卡洛路径追踪器会想要对一个与*T*成比例的距离进行重要性采样。对它的物理解释是它会随机的模拟一个距离，这个距离是光子发生相互作用的位置。路径追踪器接着可以随机的决定这个事件是吸收或散射。如果是后者（散射），继续在由相位函数采样而决定的方向追踪光子。与*T*成比例的概率密度为

![](./formula-28-3.JPG)

如果介质是均匀的（例如κ是常量），这个概率密度就简化成了一个指数分布 κe^(−κt)，并且可以使用逆变换方法(inversion method)来计算距离

![](./formula-28-4.JPG)

对于具有均匀分布的ξ，t可以满足它的预期分布。对于非均匀介质，这个方法不可行，因为对于一个普通的κ，方程3中的积分不能分析求解，即使可以，它的倒数可能不存在[@note11]**（前句不确定）**。

## 28.2 WOODCOCK跟踪
在跟踪中子轨迹的的背景下(中子使用的方程与光子使用的方程相同)，一个在非均匀介质中用于对距离进行重要性采样的技术在1960年代被广泛使用。它被叫做 *Woodcock跟踪* （Woodcock tracking），参考Woodcocok所著文献[5]

这个想法来源比较简单，它基于一个事实，均匀介质可以被简单处理。为了得到一个人工的齐次方程组(homogeneous setting)，一个假想的消光(extinction)系数被添加，这样使得实际消光和假想消光之和与κmax的最大值处处相等。人工体积(artificial volume)可以被解释为真实粒子的混合物，它可以吸收和散射，假想粒子不会发生任何现象。如下图28-1。


![](./figure-28-1.JPG)
图28-1 一个路径穿过非均匀介质，云中呈现高密度，云的周围呈现低密度。真实的的“粒子”被着色为灰色，假想粒子被着色为白色。（射线）与假想粒子的碰撞不影响运动轨迹。

使用常数消光系数Kmax，可以使用方程4来对距离进行采样，粒子将前进到那个距离的位置。碰撞可以是真实的也可以是虚构的，这可以基于实际消光与假想消光在那个位置的比值来随机的决定(实际碰撞的概率为k(x)/kmax)。如果是假想碰撞，粒子已经提前停止并且需要继续它的路径。由于指数分布是无记忆的，我们可以重复之前的步骤直到真实的碰撞发生，然后从这个新的位置随着射线继续它的路径。Coleman精确的描述了这其中数学的部分[1]，包括在方程3中重要性采样概率密度方程的证明。

值得注意的是Woodcock的原始动机不是处理任意的非均匀介质，而是简化并且更高效的处理任意非均匀介质：将一整个反应器看成单个的介质可以避免所有在复杂反应堆几何上的光线追踪运算。

Woodcock跟踪是一个当Kmax已知时对任何介质都有效的优雅算法，它可以用简单的几行代码实现：

![](./code-28-0.JPG)

唯一值得警惕的是一旦射线进入到周围的真空，循环需要终止。在这样的情况下不会有更多的交互作用发生，变量FLT_MAX会被返回。由于这个过程是无偏的，它非常适合渐进式蒙特卡洛渲染。

## 28.3 示例：一个简单的体积路径追踪器

为了更好的对Woodcock跟踪进行示例，我们在CUDA中给出了一个简单的蒙特卡洛体渲染路径追踪器的实现。它从摄像机开始追踪路径，经过体积，直到它离开介质。接着，它收集了来自无限环境穹顶[@note12]的贡献，这可以被配置为环境纹理或者简单的程序化梯度纹理(procedural gradient)。对于介质，我们将散射系数定义为与消光系数成和常数反射率ρ比例的值，例如 σ(x) = ρ ⋅ κ(x)。所有定义了摄像机，体程序(volume procedural)和环境光的变量都会传递给渲染内核。

![](./code-28-1.JPG)

由于我们在每个路径都需要许多随机数，并且这些随机数需要对于并行计算是安全的，我们使用了CUDA的curand。

![](./code-28-2.JPG)

体数据定义为被限制在以原点为中心的单位立方体上。为了决定在介质上的入口，我们需要一个相交函数（译者注:即 code 28-3中intersect_volume_box函数）为了决定什么时候射线离开介质，我们需要对包含进行测试[@note16] **（前句不确定）**。

![](./code-28-3.JPG)

体积的实际密度将由人工程序驱动，该程序将消光系数调整在0与kmax之间，为了说明它，我们需要实现两个程序：一个分段恒定的门格海绵和一个平滑的螺旋状的下降[@note17]。

![](./code-28-4.JPG)

![](./code-28-5.JPG)

在体积内部，我们用Woodcock跟踪来对相互作用的下一个点进行采样，如果已经不在介质内部了，就提前停止采样函数。

![](./code-28-6.JPG)

现在所有的工具都到位了，我们可以追踪体积中的路径。我们从路径与体积立方体相交开始，然后进入介质。一旦进入了，我们应用Woodcock跟踪来决定下一次的相交。在每个交点位置，我们按照反射率(albedo)计算比重并且应用俄罗斯轮盘(Russian roulette)来概率性的终结比重小于0.2的路径(并且无条件的终止超过最大长度的路径)。如果没有终止发生，我们继续采样(各向同性的)相位方程，根据相位方程来进行跟踪。一旦我们离开介质，我们可以查找环境光的贡献并且结束路径。

![](./code-28-7.JPG)
![](./code-28-8.JPG)

最终，我们加上代码逻辑块，每个像素从摄像机开始他们的路径。结果是逐渐累积的，并将其传送到色调映射缓冲以在每次迭代之后显示。

![](./code-28-9.JPG)
![](./code-28-10.JPG)

下图28-2显示了由上述路径追踪器生成的示例渲染图

![](./figure-28-2.JPG)
图28-2 两个由同一个路径追踪器生成的程序体函数[@note18]，由一个简单的梯度贴图(上)和环境贴图(下)照亮。反射率被设置为0.8，体积相互作用[@note19]的最大值设置为1024.(环境图由Greg Zaal提供https://hdrihaven.com.)

## 28.4 延伸阅读

Woodcock跟踪方法也可以被用来概率性的评估透射比，如当阴影射线（shadow ray）穿过体积时会需要它。这可以通过采样(可能是多重采样)距离并使用“在旅程中存活”的那些比率作为估计来实现。[4]作为优化，用于继续路径的随机变量可以由其期望值替换：不是以1-κ(x)/κmax的概率继续路径，而是使用那些概率的乘积(直到此路径达到了采样距离)[2]

如果一个场景里的最大消光系数比通常遇到的高很多，就需要更多的迭代，这个方法就过于低效了。Novák的最新技术报告[3]为进一步优化提供了很好的总结。

## REFERENCES

[1] Coleman, W. Mathematical Verification of a Certain Monte Carlo Sampling Technique and Applications of the Technique to Radiation Transport Problems. Nuclear Science and Engineering 32 (1968), 76–81.
[2] Novák, J., Selle, A., and Jarosz, W. Residual Ratio Tracking for Estimating Attenuation in Participating Media. ACM Transactions on Graphics (SIGGRAPH Asia) 33, 6 (Nov. 2014), 179:1–179:11.
[3] Novák, J., Georgiev, I., Hanika, J., and Jarosz, W. Monte Carlo Methods for Volumetric Light Transport Simulation. Computer Graphics Forum 37, 2 (May 2018), 551–576.
[4] Raab, M., Seibert, D., and Keller, A. Unbiased Global Illumination with Participating Media. In Monte Carlo and Quasi-Monte Carlo Methods, A. Keller, S. Heinrich, and N. H., Eds. Springer, 2008, pp. 591–605.
[5] Woodcock, E. R., Murphy, T., Hemmings, P. J., and Longworth, T. C. Techniques Used in the GEM Code for Monte Carlo Neutronics Calculations in Reactors and Other Systems of Complex Geometry. In Conference on Applications of Computing Methods to Reactor Problems (1965), pp. 557–579.
