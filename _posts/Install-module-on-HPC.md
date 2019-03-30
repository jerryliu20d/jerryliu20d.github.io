module load python/anaconda
conda create -p /workspace/[username]/myEnvironment anaconda
conda install p /workspace/[username]/myEnvironment anaconda your-module-name  (See cheat sheet)[http://know.continuum.io/rs/387-XNW-688/images/conda-cheatsheet.pdf]

- If modlue not exist: try to search it in conda with `conda search -t conda your-module-name`.
- If `nltk.download` function is needed, use `python -m nltk.downloader <collection|package|all>`

source activate /workspace/[username]/myEnvironment