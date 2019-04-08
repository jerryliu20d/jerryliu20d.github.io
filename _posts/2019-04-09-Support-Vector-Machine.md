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
[Questioned](..)

By [Riesz representation theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), there exists a unique $K_x\in H~~s.t.$ [Why need uniqueness?]()

$$f(x)=L_x(f)=\langle f,k_x\rangle_H,~~\forall f\in H$$.

We can also pick $f=k_{\tilde{x}}$, that gives:

$$K_{\tilde{x}}(x)=\langle k_{\tilde{x}},k_{x}\rangle_H$$

Then we define $k(\tilde{x},x)=K_{\tilde{x}}(x)=\langle k_{\tilde{x}},k_{x}\rangle_H$, which is conformable with the kernel $k$. Finally, we define a function $𝜑:X\longrightarrow H$, $s.t.~𝜑(x)=k_x,~\forall x\in X$.

> Kernel $⟹$ RKHS

