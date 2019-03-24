---
layout:     post
title:      Optimization
subtitle:   
date:       2019-03-23
author:     Jerry Liu
header-img: img/opt-bg-1.jpg
catalog: true
tags:
    - Optimization
    - Gradient Descent
    - Stochastic Gradient Descent
    - Newton's Method
    - Quasi-Newton Update
    - BFGS
    - Secant condition
---

> Numerical optimization is at the core of much of machine learning. 

# Gradient Descent

That's not hard, we just jump to the conclusion.

To minimize the function $$f$$, We update the value with 
$$x_{n+1}=x_n-γ∇f(x_n)$$

For python code, see [here](http://peigenzhou.com/stat628/pages/notes0301.html#gradient-descent)

# Stochastic Gradient Descent

The gradient descent algorithm may be infeasible when the training data size is huge. Thus, a stochastic version of the algorithm is often used instead.
To motivate the use of stochastic optimization algorithms, note that when training deep learning models, we often consider the objective function as a sum of a finite number of functions:

$$f(x)=\frac{1}{n}∑_{i=1}^nf_i(x)$$,
where $f_i(x)$ is a loss function based on the training data instance indexed by i. When n is huge, the per-iteration computational cost of gradient descent is very high.

At each iteration a mini-batch $B$ that consists of indices for training data instances may be sampled at uniform with replacement. Similarly, we can use

$$∇f_B(x)=\frac{1}{\lvert B\lvert}∑_{i∈B}∇f_i(x)$$

to update x as

$$x_{n+1}=x_{n}−η∇f_B(x)$$

# Newton's Method

Suppose we want to reach the global minimizer of $f$ with parameter x. Suppose, we have an estimate $x_n$ and we wangt out next estimate $x_{n+1}$ to have the property that $f(x_{n+1})\lt f(x_n)$. Newton's method use the taylor expansion:

$$f(x+Δx)≈f(x)+Δx^T∇f(x)+\frac{1}{2}Δx^T(∇^2f(x))Δx$$

, where  $∇f(x)$ and $∇^2f(x)$ are the gradient and Hessian of f at the point $x_n$. This approximation holds when $‖Δx‖→0$.

Without loss of generality, we can write $x_{n+1}=x_n+Δx$ and re-write the above equation,

$$f(x_{n+1})≈h_n(Δx)=f(x_n)+Δx^Tg_n+\frac{1}{2}Δx^TH_nΔx$$

,where $g_n$ and $H_n$ represent the gradient and Hessian of $f$ at $x_n$.

If we take the differentiation with respect to $Δx$ and set it to zero yields:
$$Δx=−H^{−1}_ng_n$$
That is to say:
$$x_{n+1}=x_n−α(H^{−1}_ng_n)$$

# Quasi-Newton

The central issue with NewtonRaphson is that we need to be able to compute the inverse Hessian matrix. Usually, it's computational large. Thus we introduce Quasi-Newton. The Quasi-Newton can update $$H^{-1}_n$$ according to $$H^{-1}_{n-1}$$

#### Secant Condition

From taylor expansion we have $$f(x_{n+1})≈h_n(Δx)=f(x_n)+Δx^Tg_n+\frac{1}{2}Δx^TH_nΔx$$, and let's think about what's the good property for $h_n(Δx)$.

Actually, we'd like to ensure $h_n()$ have the same first order derivation as $f()$ at point $x_n$ and $x_{n-1}$:
$$∇h_n(x_n)=g_n$$, and
$$∇h_n(x_{n−1})=g_{n-1}$$
We combine these two conditions together:
$$∇h_n(x_n)−∇h_n(x_{n−1})=g_n−g_{n−1}$$

Use the derivation of formula (2), we have:

$$(x_n−x_{n−1})H_n=(g_n−g_{n−1})$$
This is the so-called "secant condition" which ensures that $H_{n-1}$ behaves like the Hessian at least for the difference $$x_n-x_{n-1}$$. Assuming $$H_n$$ is invertible, then multiplying both sides by $$H_n^{-1}$$ yields:

$$(x_n−x_{n−1})=(g_n−g_{n−1})H_n^{-1}$$

or

$$s_n=y_nH^{-1}_n$$

According to fomula (3), we can calculate the inverse Hessian matrix with only $$s_n$$ and $$y_n$$.

# BFGS Update

Intuitively, we want Hn to satisfy the two conditions above:

- Secant condition holds for $s_n$ and $y_n$
- $$H_n$$ is symmetric

Given the two conditions above, we’d like to take the most conservative change relative to $$H_{n−1}$$. This is reminiscent of the [MIRA update](http://aria42.com/blog/2010/09/classification-with-mira-in-clojure), where we have conditions on any good solution but all other things equal, want the 'smallest' change.

$$min_{H^{-1}}‖H^{-1}-H^{-1}_{n-1}‖^2\\
s.t.\ H^{-1}y_n=s_n\\
\ \ \ \ \ \ \ H^{-1}\ is\ symmetric$$

, The norm used here ∥⋅∥ is the [weighted frobenius norm](http://mathworld.wolfram.com/FrobeniusNorm.html). The solution is given by 

$$H^{−1}_{n+1}=(I−ρ_ny_ns^T_n)H^{−1}_n(I−ρ_ns_ny^T_n)+ρ_ns_ns^T_n$$

,where $$ρ_n=(y^T_ns_n)^{−1}$$. The proof out the solution is outside of scope of this post. And for BFGS, we can use any initial matrix $$H_0$$ as long as it is positive definite and symmetric.

# Reference
- GLUON [Gradient descent and stochastic gradient descent from scratch](https://gluon.mxnet.io/chapter06_optimization/gd-sgd-scratch.html)
- Aria [Numerical Optimization: Understanding L-BFGS](http://aria42.com/blog/2014/12/understanding-lbfgs)