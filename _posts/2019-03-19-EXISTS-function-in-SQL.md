---
layout:     post
title:      EXISTS function in SQL
subtitle:   
date:       2019-03-18
author:     Jerry Liu
header-img: img/post-bg-1.jpg
catalog: true
tags:
    - SQL
---
<link rel="stylesheet" href="/assets/css/syntax.css">
# Logic of EXISTS
>`EXISTS` is a frequently used SQL function.

`EXISTS` will return the boolean value whether the condition is satisfied or not.
For example, suppose we have the following tables:

|ID|NAME|
|----|----|
|1|A<sub>1</sub>|
|2|A<sub>2</sub>|
|3|A<sub>3</sub>|


|ID|UID|NAME|
|--|---|----|
|1|1|B<sub>1</sub>|
|2|2|B<sub>2</sub>|
|3|2|B<sub>3</sub>|

```sql
SELECT ID, NAME FROM A EXISTS (SELECT * FROM B WHERE A.ID=B.UID)
```

the output is:

|ID|NAME|
|--|----|
|1|A<sub>1|
|2|A<sub>2|

As we can see that the function is excuted by rows. It only return those rows satisfied the condition.

`EXISTS` will not save the result of the query result of the condition because it doesnot matter. It only return the boolean result according to whther the query is empty or not. The logic of `EXISTS` looks like the following codes:
```python
result = []
for i in range(len(A)):
    if exists(A[i].ID):
        result.append(A[i])
return result
```

From the above codes we will find that the `EXISTS` is preferred when table B is larger than table A. 

# Compare with IN

We still use the previous table. But we use `IN` instead.

```SQL
SELECT ID, NAME FROM A IN (SELECT UID FROM B)
```

It is equivelant to the following codes:
```python
result = []
for i in range(len(A)):
    for j in range(len(B)):
        if A[i].ID == B[i].UID:
            result.append(A[i])
            break
return result
```
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

The function `IN` will only be executed once, but it will traversal all elements in A and B, which may takes $O(m,n)$
