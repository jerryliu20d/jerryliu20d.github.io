---
layout:     post
title:      Linear Regression with constraints
subtitle:   
date:       2019-03-29
author:     Jerry Liu
header-img: img/post-3-bg.jpg
catalog: true
tags:
    - Dual
    - Linear Regression
---

> It's another way to understand model significance test from the aspect of linear regression with constriaints.

# Estimation under linear constraints

We have the regression model:

$$Y=Xβ+ɛ\\
s.t.\ Aβ=c$$

,where A is comfo
The optimization problem is a given comformable matrix and c is constant. The optimization problem is:

$$min_β\ ‖Y-Xβ‖^2\\
s.t.\ Aβ=c$$

This post will only solve estimate the $$β$$ with dual problems. If you are interested in the geometric solution of it, please read Chapter4 of ["Regression Analysis"](https://www.amazon.com/%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90-%E6%AD%A6%E8%90%8D%EF%BC%8C%E5%90%B4%E8%B4%A4%E6%AF%85/dp/B01GYIXJ4A) by Xianyi Wu and Ping Wu.

# Dual Problem

If you are unfamiliar with dual problems, please read the slide from [CS524 Wisconsin-Madison](https://drive.google.com/file/d/1ZBAyc1hLMxNPVugfWI0M0gAdZqx29lq0/view?usp=sharing) first.
The dual problem is 

$$max_λ‖Y-Xβ+X(X^TX)^{-1}A^Tλ‖^2+2λ^T(A\hat{β}-A(X^TX)^{-1}A^Tλ-c)$$

,where λ is the comformable vector and $$\hat{β}$$ is the MLE estimator of $$β$$ without any constraints. Now the dual problem can be solve with matrix differentiation.
We solve $$λ$$ with:

$$A\hat{β}-c=A(X^TX)^{-1}A^T\label{beta est}$$

Put the solution from \ref{beta est}. We have the solution that 

$$\hat{β_R}=\hat{β}-(X^TX)^{-1}A^T(A(X^TX)^{-1}A^T)^{-1}(A\hat{β}-c)$$

This is the same as the solution from geometric method.

# Note
I donot check the complementary slackness condition here. You can do it yourself if interested. I omit the calculation in this post. It's just simple matrix differentiation. It's tedious to write the calculation steps in Markdown and I think it's better that YOU do it.

# Music Recommendation
From now on, I will recommend some music at the end of the post. Sometimes I will listen some music if I cannot focus on study. Music will stimulate your brain and make you work efficiently. But listen to music frequently while working will weaken thestimulation. So Listen to music scientificly:-)

Today's Music:
"僕が死のうと思ったのは" --- 中島 美嘉
"Boku ga Shinou to Omotta no Wa" --- Mika Nakashima

It can be simply translated as "I thought I would have died". But actually it is not a dark single. Mika said "You may not know the theme unless you listen it to the end" in the press conference. It describes sorrow in daily life. But the world is not hopeless because you still have a little happiness in hand. And here is another song "1-800-273-8255" --- Alessia Cara, Khalid (I also like it Aha). It conveys the similar idea. It encourage me sometimes when I'm in depression.