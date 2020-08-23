---
layout:     post
title:      Cloud Sync via a Paper Management Library ":" Zotero
subtitle:   
date:       2020-08-23
author:     Jerry Liu
header-img: img/post-4-bg.jpg
catalog: true
tags:
    - Zotero
---

> Start your paper reading journey from Zotero!

# What's Zotero

[Zotero](https://www.zotero.org/) is a free Paper Management Software to arrange your literatures. Notable features include web browser integration, online syncing, generation of in-text citations, footnotes, and bibliographies, as well as integration with the word processors Microsoft Word, LibreOffice Writer, and Google Docs. [wiki](https://en.wikipedia.org/wiki/Zotero) It is notable that one can automatically get metadata from papers via Zotero. Zotero can generate formatted bibliographies in batch. Also, it helps keywords search not only in abstract and main text, even in annotation. 

I am still discovering features in Zotero. I won't share tips about how to use Zotero today. But I will show how to automatically synchronize the files on different platform via clouds. Unfortunately, Zotero do not provide such features. Thanks to the open source policy. [Zotfile](http://zotfile.com/) propose a solution. By the way, Zotero do have their own clouds called "Zotero" and "WebDev". You may skip all following steps if you are willing to pay for extra stroage room.

# Setups

I will just skip the installation of Zotero and Zotfile. Details are shown on official websites. But do remember to register an account and linked it with Zotero on your PC. In my case, I choose [Onedrive](https://onedrive.live.com/about/en-us/signin/). It's up to you to choose other clouds like google drive, dropbox and etc. After correctly installing Zotero and Zotfile, go to the Edit -> Preference. Uncheck the box "sync attachment files inusing Zotero storage" because we don't need official clouds to interupt us. ![Fig1]({{baseurl}}\img\sync_zotero\img1.png)
