# This is the main repository for my Master thesis.
Here you will find scripts for creating all the plots in my thesis, in addition to the thesis itself.

## Recreating results + repo structure
- basin_lists: The two text files in this directory are needed for the model to know which basins are available in the dataset.
- runs: This is where all the model training and analyzation runs are configured. After downloading CAMELS and CAMELS-GB, as well as installing [CamelsML](https://github.com/bernharl/camelsml) (recommended: clone this repo, enter it and run ```pipenv sync```. Othewise, CamelsML can be installed using either ```pipenv install camelsml --python 3.8```, or the instructions provided in the link), one should be able to recreate all the results in my thesis using the scripts in this directory.
- notebooks: This is where all figures are generated for this thesis. These notebooks need to be run in order to recreate the figures. To be able to create the figures, you need to first complete the above step. (Skip over the cells that do not run, all figures are confirmed to work, but I do not want to remove the now non-working legacy cells that show how my work progressed earlier...)
- doc/thesis: This is where the LaTeX document is placed. To compile it, run ```latexmk -pdf -shell-escape```. You will likely need to run ```pipenv sync``` in doc/thesis/illustration_scripts first. For the figures and tables to also be included, you need to follow all the two above steps first. The compiled thesis will also be available here in .pdf format once it is ready for submission.

As this repo contains almost all information related to my thesis, there are a lot more directories than those mentioned above. You can likely disregard these. 
