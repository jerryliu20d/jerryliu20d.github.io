---
layout:     post
title:      Cloud Sync via a Paper Management Library - Zotero
subtitle:   
date:       2020-08-23
author:     Jerry Liu
header-img: img/sync_Zotero/bg.jpg
catalog: true
tags:
    - Literature management
---

> Start your paper reading journey from Zotero!

# What's Zotero

[Zotero](https://www.zotero.org/) is a free Paper Management Software to arrange your literatures. Notable features include web browser integration, online syncing, generation of in-text citations, footnotes, and bibliographies, as well as integration with the word processors Microsoft Word, LibreOffice Writer, and Google Docs. [wiki](https://en.wikipedia.org/wiki/Zotero) It is notable that one can automatically get metadata from papers via Zotero. Zotero can generate formatted bibliographies in batch. Also, it helps keywords search not only in abstract and main text, even in annotation. 

I am still discovering features in Zotero. I won't share tips about how to use Zotero today. But I will show how to automatically synchronize the files on different platform via clouds. Unfortunately, Zotero do not provide such features. Thanks to the open source policy. [Zotfile](http://zotfile.com/) propose a solution. By the way, Zotero do have their own clouds called "Zotero" and "WebDev". You may skip all following steps if you are willing to pay for extra stroage room.

# What can Zotero do?

If following things is bothering you, Zotero may be a good solution

1) Automatically sync files across platforms like: Windows, Android, Mac, Ipad, Iphone, etc
2) Generate bibliographies in batch
3) Search key words in a specific region(Title, Author, Year, Main text, Abstract, or even your own annotations)
4) Download and manage literature metadata
5) Free

# Setups

I will just skip the installation of Zotero and Zotfile. Details are shown on official websites. But do remember to register an account and linked it with Zotero on your PC. In my case, I choose [Onedrive](https://onedrive.live.com/about/en-us/signin/). It's up to you to choose other clouds like google drive, dropbox and etc. After correctly installing Zotero and Zotfile, go to the Edit -> Preference. Uncheck the box "sync attachment files inusing Zotero storage" because we don't need official clouds to interupt us. 
![Fig1]({{baseurl}}\img\sync_Zotero\img1.png)
 
 Then move to Advanced tab. Change the linked attachment data directory to your cloud directory. You can also change the data directory location if you wish. But NEVER change it to a cloud folders because you are very likely lose some files if you move them at an another end. The data directory maily store the original files in Zotore. They are separately stored in each folder named by some nuisance characters. And you won't want to read them. That's why we need Zotfile to extract pdf from them to your clouds.
 ![Fig2]({{baseurl}}\img\sync_Zotero\img2.png)
 ![Fig3]({{baseurl}}\img\sync_Zotero\img3.png)

 Then go to Tools -> Zotfile Preferences. Set paths as the figure says. Notice here, subfolders are defined by "/%C", which means all files will keep the collection path as it is in Zotero.

![Fig4]({{baseurl}}\img\sync_Zotero\img4.png)

And finally, it's all done. However, things does not work as you expect....

When you import a new file into Zotero. You can find it somewhere in your data directory (~/Zotero/storage/). But, opps, it does not send a copy to your cloud. All you need to do is to rename the attachments like this. And everthing works well.
![Fig5]({{baseurl}}\img\sync_Zotero\img5.png)

# Chrome Extension

[Zotero chrome extension](https://chrome.google.com/webstore/detail/zotero-connector/ekhagklcjbdpajgpjgmbionohlpdbjgc?hl=zh-CN) provides user even more convenient ways to collect literatue. You can save it to Zotero with just one click on side bar. What's more? It will automatically sync to your cloud this time! Sounds good?

# Sync on Tablet

Some may used to reading paper on Ipad and they need to update files back and forth. Unfortunely, Zotero is not avaliable on Apple store. However, zotfile can do it. Set your directory and subfolder definition like this.

![Fig6]({{baseurl}}\img\sync_Zotero\img6.png)

# More works to do

Though Zotero solves most problems. But one important features still needs development. When you delet files from Zotero, Zotfile won't delet it from your clouds synchronously. It's a headache for readers. But there is an [alternative way](https://github.com/jlegewie/zotfile/issues/96#issuecomment-505084568) to achieve this feature. I have not test it yet. I will let you guys know whether it works or not.


<!--
https://sspai.com/post/59035
-->
