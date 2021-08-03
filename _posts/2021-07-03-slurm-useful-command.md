---
layout:     post
title:      Slurm Useful Command
subtitle:   Summary of useful command
date:       2021-07-03
author:     Jerry Liu
header-img: img/SLURM.png
catalog: true
tags:
    - SLURM
---

# Job status related command

- Waiting time:  
Use `squeue` to see estimated start dates assuming new jobs with higher priority are not submitted, and assuming that jobs run to the wall time given on submission.

```console
squeue -a --start -j <job_id>
```


- Submitted jobs info:  
The `scontrol` command can be used to display information about submitted jobs, running jobs, and very recently completed jobs.

 ```console
scontrol show job <job_id>
 ```


- Finished jobs info:  
To see job usage for a finished job, issue:

```console
seff <job_id>
```


- Running job info:  
to list status info for a currently running job, issue:

```console
sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,JobID -j <jobid> --allsteps
```

- More details when connet to the nodes
```
ssh nova056
top
```

# Partition status

- All partition info:  
To report the status of the available partitions and nodes, issue:

```console
sinfo
```


- More details:  
For more details on partitions limits, issue:

```console
scontrol show partitions
```


# Group info
`sprio` can be used to display a breakdown of the priority components for each job, e.g. `sshare` shows the two level fair-share hierachy data.

# Install R packages

```console
module load python/anaconda  
conda create -p /workspace/[username]/myEnvironment anaconda  
conda install p /workspace/[username]/myEnvironment anaconda your-module-name
```

[See cheat sheet](http://know.continuum.io/rs/387-XNW-688/images/conda-cheatsheet.pdf)  

- If modlue not exist: try to search it in conda with `conda search -t conda your-module-name`.
- If `nltk.download` function is needed, use `python -m nltk.downloader <collection|package|all>`

```console
source activate /workspace/[username]/myEnvironment
```

# External Links
see more tutorials [here](https://researchit.las.iastate.edu/search/content/slurm).