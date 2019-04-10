---
layout:     post
title:      Support Vector Machine
subtitle:   
date:       2019-04-08
author:     Jerry Liu
header-img: img/post-4-bg.jpg
catalog: true
tags:
    - SVM
---

> Easy model, Good performance

# Prerequisite

Before you read this post, you should have a have some basic knowledge of kernels and Hilbert space. If not, please read the following materials first: [Hilbert Space](https://en.wikipedia.org/wiki/Hilbert_space), [RKHS](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space#Moore%E2%80%93Aronszajn_theorem), [Kernels](https://en.wikipedia.org/wiki/Kernel_method) and [SVM](https://en.wikipedia.org/wiki/Support-vector_machine). I'll also make some small extension.

# Overview

[Nicolas Garcia Trillos](http://www.nicolasgarciat.com/cv.html), Assistant Professort from UW-Madison, wrote summary table to show the connection and difference between SVM and Regularized Linear Regression in both Euclidean space and Hilbert Space. It is super helpful.
<span id="fig1"></span>
![Summary]({{baseurl}}\img\post4-1.jpg)


# SVM in Euclidean space

Suppose we have n data points $x_1\dots x_n$ labeled as $y_1\dots y_n$. For linear separable data, we want to perfectly classify the data with hyperplane $H_{\tilde{\beta},\tilde{\beta_0}}$. It's fair to let $\Vert\tilde{\beta}\Vert=1$. Here our decision rule is

$$
y_i=\left\{
    \begin{array}{lr}
    1~~~~~~~~~~~~&if~\langle\tilde\beta,x_i\rangle+\tilde{\beta_0}>0\\  
    -1~~~~~~~~~~&o.w
    \end{array}
    \right.
$$

And we want to minimize the distance between the data point and the hyperplane, i.e. $y_i(⟨\tilde{\beta},x_i⟩+\tilde{\beta_0})$. Define
$m=\min\limits_{i=1,\dots ,n}y_i(⟨\tilde{\beta},x_i⟩+\tilde{\beta_0})$. The optimization problem is

$$
\max_{\beta,\beta_0}m\\
s.t.~~y_i(⟨\beta,x_i⟩+\beta_0)\ge m.
$$

Now let $\beta = \frac{\tilde{\beta}}{m}$, i.e. $m=\frac{1}{\Vert\beta\Vert}$, because $\Vert\tilde{\beta}\Vert=1$. Then we have

$$
\min_{\beta,\beta_0}\frac{\Vert\beta\Vert}{2}\\
s.t.~~y_i(⟨\beta,x_i⟩+\beta_0)\ge 1.
$$

This is the so called hard margin SVM. And Soft Margin SVM is quiet similar. For the dual problem of soft margin SVM, it is stated in the [figure](#fig1). You will find how SVM turn a problem from $\mathcal{R}^p$ to $\mathcal{R}^n$. And the solution to the dual problem is left to you guys. If you are not familiar with dual problem, see [here](https://drive.google.com/file/d/1ZBAyc1hLMxNPVugfWI0M0gAdZqx29lq0/view?usp=sharing).


# RHKS and Kernels

We define the matrix representation of kernel $k:x*x\longrightarrow\mathcal{R}$ is $K$, which is symmetric and psd. Here we will show RKHS can uniquely define a corresponding kernel and vice versa.

> RKHS $⟹$ Kernel

We can define RKHS with following:
for every $x\in X$, the map:

$$L_x:f\in H\longrightarrow f(x)$$

is a continuous map. in particular, 

$$|f(x)-\tilde{f}(x)|\le C_x\Vert f-\tilde{f}\Vert,~~\forall f,\tilde{f}\in H$$

By [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), there exists a unique $K_x\in H~~s.t.$ [Why need uniqueness?]()

$$f(x)=L_x(f)=\langle f,k_x\rangle_H,~~\forall f\in H$$.

We can also pick $f=k_{\tilde{x}}$, that gives:

$$k_{\tilde{x}}(x)=\langle k_{\tilde{x}},k_{x}\rangle_H$$

Then we define $k(\tilde{x},x)=k_{\tilde{x}}(x)=\langle k_{\tilde{x}},k_{x}\rangle_H$, which is conformable with the kernel $k$. Finally, we define a function $𝜑:X\longrightarrow H$, $s.t.~𝜑(x)=k_x,~\forall x\in X$.

> Kernel $⟹$ RKHS

The result can be derived from [moore aronszajn theorem](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) directly. 

We define a Hilbert space:

$$H=\{\sum_{i=1}^\infty a_ik(\cdot,x_i)\}\\
s.t.~\sum_{i=1}^\infty a_i^2k(x_i,x_i)<\infty$$

,where $x_i\in X$, $a_i\in\mathcal{R}$. And the inner product is defined as:

$$\langle\sum_{i=1}^\infty a_ik(\cdot,x_i),\sum_{i=1}^\infty b_jk(\cdot,y_i)\rangle_H≐\sum_{i=1}^\infty\sum_{j=1}^\infty a_ib_jk(x_i,x_j)$$.

Then we check the reproducing property:

$$
\begin{aligned}
\langle f,k(\cdot,x)\rangle_H&=\langle\sum_{i=1}^\infty a_ik(\cdot,x_i),k(\cdot,x)\rangle\\
&=\sum_{i=1}^\infty a_ik(x,x_i)\\
&=f(x)
\end{aligned}
$$

To prove uniqueness, let $G$ be another Hilert space with reproducing kernel $k$. FOr any $x$ and $y$ in $X$, we have:

$$\langle k_x,k_y\rangle_H=k(x,y)=\langle k_x,k_y\rangle_G$$, by completeness, it is unique.

# Regularization in Special Case

In linear regression, when the $X$ matrix has the preoblem with illness, we usually use regularization. The idea is

$$\min\limits_{f\in Z}J(f)=\lambda R(f)+\sum_{i=1}^n(f(x_i)-y_i)^2$$

for some family of function $Z$ and regularization $R$.

> Note that we cannot arbitrary choose one space. For example, we cannot use the $\mathcal{L}^2$ space (Lebesgue spaces), because for the point $x_i$, we cannot make sure $f(x_i)$ makes sense.

We will choose RKHS and $R(f)=\Vert f\Vert_H^2$ at last, but here let us first choose a special case with $X=[0,1]$.

We define Hilbert space:

$$
Z=\left\{
\begin{aligned}
&f:[0,1]\longrightarrow\mathcal{R}\\
&s.t.~f(x)=\int_0^xf'(t)dt
\end{aligned}
\right\}
$$

And regularization function $R(f)=\int_0^1(f'(x))^2dx$.

---
How we solve optimization problem in euclidean space?

We use derivation!

$$\min\limits_{x\in\mathcal{R^m}}F(x)$$

If the optimization point is $x^*$ and we know it is the same problem with

$$\min\limits_{t\in\mathcal{R}}F(x^*+tv)$$

,where the $v$ is constant in $\mathcal{R^m}$. [If the $x^*$ is not unique?]()

Let $𝜑_v(t)=F(x^*+tv)$, then the minimum reach at $t=0$, i.e. 

$$0=𝜑_v'(0)=\langle \nabla F(x^*),v\rangle$$

> Use $\langle \nabla J(f^*),g\rangle=0$ to solve the optimization problem!

$$
\begin{aligned}
𝜑_g'(t)\Big|_{t=0}=&0\\
\lambda\int_0^1(f^*)'g'dx=&-\sum_{i=1}^n(f^*(x_i)-y_i)g(x_i)
\end{aligned}
$$

This is so called Euler-Lagrange Equation. Then we will solve this equation by some tricks.

1. $g=g_1$, where $g_1\doteq min\{x,x_1\}$

   The E-L equation follows the following equations.

    $$
    \begin{aligned}
    \lambda\int_0^{x_1}(f^*)'(t)dt=&\sum_{i=1}^n(f^*(x_i)-y_i)(x_i\wedge x_1)\\
    \lambda f^*(x_1)-f^*(0)=&\sum_{i=1}^n(f^*(x_i)-y_i)(x_i\wedge x_1)\\
    \lambda f^*(x_1)=&\sum_{i=1}^n(f^*(x_i)-y_i)(x_i\wedge x_1)
    \end{aligned}
    $$

    Note that the last equation holds because we has already defined that $f(x)=\int_0^xf'(t)dt$. So $f^*(0)=0$.

2. $g=g_j$, $g_j\doteq min\{x,x_j\}$, $for~j=1,\dots,n$

   We have the same result:
   $$
   \lambda f^*(x_j)=\sum_{i=1}^n(f^*(x_i)-y_i)(x_i\wedge x_j)
   $$
<span id="loc1"></span>

3. When $x>x_n$

   Because we define the regularization term as $R(f)=\int_0^1(f'(x))^2dx$. We will just let $f'(x)=0$. That is a constant.

> Actually, it is a piecewise linear function. We will show it by ${(f^*)'}'=0$

Take $g\in Z:[0,1]\longrightarrow\mathcal{R}$ $s.t.~g(0)=0$ and $g(x)=0~~\forall x\ge x-1$. 

Then we must have 

$$
\begin{aligned}
\int_0^{x_1}(f^*)'g'dx&=0\\
-\int_0^{x_1}(f^*)''gdx+(f^*)'g\Big|_0^{x_1}&=0\\
-\int_0^{x_1}(f^*)''gdx&=0\\
(f^*)''g&=0
\end{aligned}
$$

This is for arbitrary $g\in Z$ satisfying the previous defined condtion. Thus ${(f^*)'}'=0$. 

Until now we have proved that $f^*(x)$ in constant on $[0,x_1]$. It is the same for each interval on $[0,x_n]$. And for $x\in[x_n,1]$, we have already showed in [part 3](#loc1).

# Regularization in General Case

For general case, the optimization problem is:

$$\min_{f\in H}\lambda\Vert f\Vert_H^2+\sum_{i=1}^n(f(x_i)-y_i)^2$$

, where $H$ is RKHS. Let $k:[0,1]*[0,1]\longrightarrow\mathcal{R}$, $k(x,\tilde{x})=min\{x,\tilde{x}\}$. By the [representor theorem](https://en.wikipedia.org/wiki/Representer_theorem), we have:

$$f^*=\sum_{i=1}^na_i(\cdot\wedge x_i)$$

We find it conincide with the previous result.

# 