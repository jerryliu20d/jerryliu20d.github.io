---
layout:     post
title:      Founadation of Machine Learning Part I
subtitle:   
date:       2019-03-22
author:     Jerry Liu
header-img: img/post2-bg-1.jpg
catalog: true
tags:
    - Machine Learning
    - Upervised Learning
---
# Supervised Learning

## Setup
Let us formalize the supervised machine learning setup. Our training data comes in pairs of inputs $(x,y)$, where $x∈R_d$ is the input instance and y its label. The entire training data is denoted as
$D={(x_1,y_1),…,(x_n,y_n)}⊆R^d×C$
where:

$R^d$ is the d-dimensional feature space
$x_i$ is the input vector of the ith sample
$y_i$ is the label of the ith sample
$C$ is the label space
The data points $(x_i,y_i)$ are drawn from some (unknown) distribution $P(X,Y)$. Ultimately we would like to learn a function h such that for a new pair $(x,y)∼P$, we have $h(x)=y$ with high probability (or $h(x)≈y$). We will get to this later. For now let us go through some examples of $X$ and $Y$.

## No Free Lunch
Before we can find a function h, we must specify what type of function it is that we are looking for. It could be an artificial neural network, a decision tree or many other types of classifiers. We call the set of possible functions the hypothesis class. By specifying the hypothesis class, we are encoding important assumptions about the type of problem we are trying to learn. The No Free Lunch Theorem states that every successful ML algorithm must make assumptions. This also means that there is no single ML algorithm that works for every setting.

## Summary of Part I
We train our classifier by minimizing the training loss:
Learning:
$$h^∗(⋅)=argmin_{h(⋅)∈H_1}\frac{1}{|D_{TR}|}∑_{(x,y)∈D_{TR}}ℓ(x,y|h(⋅))$$,
where $H$ is the hypothetical class (i.e., the set of all possible classifiers $h(⋅)$). In other words, we are trying to find a hypothesis h which would have performed well on the past/known data.

We evaluate our classifier on the testing loss:
Evaluation: 
$$ϵ_{TE}=\frac{1}{|D_{TE}|}∑_{(x,y)∈D_{TE}}ℓ(x,y|h^∗(⋅))$$.
If the samples are drawn i.i.d. from the same distribution $P$, then the testing loss is an unbiased estimator of the true generalization loss:
Generalization: 
$$ϵ=E_{(x,y)}\sim P[ℓ(x,y|h^∗(⋅))]$$.

Note that, this is the form of corssentrophy $H[P(y\lvert h^*(x)), P(y\rvert x)]$.

No free lunch. Every ML algorithm has to make assumptions on which hypothesis class H should you choose? This choice depends on the data, and encodes your assumptions about the data set/distribution P. Clearly, there's no one perfect $H$ for all problems.
