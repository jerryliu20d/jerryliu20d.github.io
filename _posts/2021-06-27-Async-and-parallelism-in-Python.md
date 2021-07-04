---
layout:     post
title:      Async and parallelism in Python
subtitle:   
date:       2021-06-19
author:     Jerry Liu
header-img: 
catalog: true
tags:
    - Python
    - Parallelism
---

> Make full use of your machine!

# What's parallelism

**Parallelism** means conducting multiple operations at the same time. **Concurrency** has a broader meaning than parallelism. One of the methods is **Multiprocessing**, which entails spreading tasks over a computer's central processing units (CPUs). **Threading** is a concurrent execution model whereby multiple threads take turns executing tasks. There is a rule of thumb that threading is better for IO-bound tasks.

# Async IO
**Async IO** is a style of concurrent programming, but it is not parallelism. In fact, it is a single-threaded, single-process design: it uses **cooperative multitasking**. See an example below.

> Chess master Judit Polgár hosts a chess exhibition in which she plays multiple amateur players. She has two ways of conducting the exhibition: synchronously and asynchronously.  
<br>Assumptions:
> - 24 opponents
> - Judit makes each chess move in 5 seconds
> - Opponents each take 55 seconds to make a move
> - Games average 30 pair-moves (60 moves total)  

> **Synchronous version**: Judit plays one game at a time, never two at the same time, until the game is complete. Each game takes (55 + 5) * 30 == 1800 seconds, or 30 minutes. The entire exhibition takes 24 * 30 == 720 minutes, or 12 hours.  
> **Asynchronous version**: Judit moves from table to table, making one move at each table. She leaves the table and lets the opponent make their next move during the wait time. One move on all 24 games takes Judit 24 * 5 == 120 seconds, or 2 minutes. The entire exhibition is now cut down to 120 * 30 == 3600 seconds, or just 1 hour. [source](https://www.youtube.com/watch?t=4m29s&v=iG6fr81xHKA&feature=youtu.be)

## asyncio package in python

A **coroutine** is a specialized version of a Python generator function. More specific, it is a function that can suspend its execution before reaching return, and it can indirectly pass control to another coroutine for some time.  
```python
import asyncio

async def count():
    print("One")
    await asyncio.sleep(1)
    print("Two")

async def main():
    await asyncio.gather(count(), count(), count())

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"executed in {elapsed:0.2f} seconds.")
```

```
>>>
One
One
One
Two
Two
Two
executed in 1.01 seconds.
```

## `async` class and  `await` keyword
- The syntax `async def` introduces either a native **coroutine** or an **asynchronous** generator.
- The keyword `await` passes function control back to the event loop. Roughly speaking, it perform the same functionality of `yield from` in old version. In coroutine, it tells python to pause when meets `await` and come back to the coroutine when the execution is finished. 
- Using `await` and/or `return` creates a coroutine function. To call a coroutine function, you must `await` it to get its results. It's less frequent to use `yield` in a coroutine function.




> ### :warning: Support for generator-based coroutines is **deprecated** and is scheduled for removal in Python 3.10.
> ```python
>@asyncio.coroutine
>def old_style_coroutine():
>    yield from asyncio.sleep(1)
>
>async def main():
>    await old_style_coroutine()
> ```
> This decorator should **not** be used for `async def` coroutines and use `async` def instead.

