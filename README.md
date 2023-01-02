### Setup ###
0. Clone this Repo and install Python 3.8 with anaconda
1. Clone Repo: sktime_dilation
2. Create and activate conda environment
<code>conda create -n benchmarkenv python=3.8</code>
<code>conda activate benchmarkenv</code>
3. Execute <code>pip install --editable .[dev]</code> in sktime_dilation Repo
4. Install other dependencies into the conda environment
5. Download UCRArchive_2018 (Univariate_ts) and copy 'Univariate_ts' folder into the benchmark directory of this Repo
6. Use the TSF_real.ipynb or BOSS.ipynb to run benchmarks with different parameters

### Good to know ###
If you change something in the sktime_dilation repo, restart the jupyter notebook kernel to apply the changes to the jupyter notebook

With <code>jupyter nbconvert --to python main.ipynb</code> you can create a python file out of the jupyter notebook to execute in the terminal