# PTYS 551: Remote Sensing of Planetary Surfaces

This graduate course will focus on the use of remote sensing in the study of rocky and icy planetary surfaces.  It is not a science course, but rather intended to provide technical knowledge of how instruments work and practical techniques to deal with their datasets. In this course, we will cover how different types of remote-sensing instruments work in theory and practice along with case studies (student-led) of specific planetary science instruments.  We will discuss what datasets are generated by these instruments, their limitations and where they can be located.  Lab sessions will provide experience in how these data are processed, visualized and intercompared. The class consists of two lectures and a 2-hour lab session each week.

## Ptys 551 Conda Environments
We recommend two conda environments to participate in this class created using the code below.
Firstly we need to install miniforge3. Installation packages and instructions are at: https://github.com/conda-forge/miniforge?tab=readme-ov-file#download


```bash
###
### Activate the base environment and install nb_conda_kernels so Jupyter Notebook can be run and can find your other environments
### Usually it's bad form to install anything in the base environment - but nb_conda_kernels is an exception.
### I also install numpy, because I want to use IDL kernels in these notebooks sometimes

conda activate
conda install nb_conda_kernels numpy=1.26.4


###
### Installing packages for PTYS551 takes about 1 GB
### Note the numpy version has to be the older one... many packages don't support 2.0 yet
### If you're trying to start over and want to erase the current ptys551 environment then run the first command too (it's harmless to run if the environment doesn't exist)

conda env remove -n ptys551
conda create -n ptys551 python=3.10 numpy=1.26.4 ipykernel scipy matplotlib scikit-learn spiceypy proj gmt gdal pandas rasterio tqdm spectral glob2 pyqt jupyterlab

### Add one more package not available on conda-forge
conda activate ptys551
pip install outlier-utils                                                                                                                                                                                           
conda deactivate
```

The second environment is for the ISIS and Ames Stereo Pipeline (ASP) packages

```bash
###
### Installing Ames Stereo Pipeline and ISIS takes about 2.5 GB
### An initialization script (and the 2nd activate command) sets the two environment variables needed: $ISISROOT and $ISISDATA
### There's an irreconcilable conflict between this install and the packages in the ptys551 environment so it needs to be separate for now

conda create -n asp
conda activate asp
conda config --env --add channels conda-forge
conda config --env --add channels usgs-astrogeology
conda config --env --add channels nasa-ames-stereo-pipeline

conda install -c nasa-ames-stereo-pipeline -c usgs-astrogeology -c conda-forge stereo-pipeline==3.3.0
python $CONDA_PREFIX/scripts/isisVarInit.py --data-dir=/Users/shane/ISISDATA
conda activate asp


###
### Installing the datasets below takes about 44 GB
### This must be done within the asp environment in order to access the downloadIsisData program
### Note we exclude the SPK and CK kernels in each mission as they're enormous

downloadIsisData base $ISISDATA 
downloadIsisData lro $ISISDATA --exclude="{spk/**,ck/**}"
downloadIsisData mro $ISISDATA --exclude="{spk/**,ck/**}"
downloadIsisData cassini $ISISDATA --exclude="{spk/**,ck/**}"
downloadIsisData galileo $ISISDATA --exclude="{spk/**,ck/**}"
downloadIsisData messenger $ISISDATA --exclude="{spk/**,ck/**}"
downloadIsisData newhorizons $ISISDATA --exclude="{spk/**,ck/**}"
```
