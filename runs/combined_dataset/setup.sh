pipenv shell
echo Creating specified basin lists
python create_specified_basin_lists.py
cd train_gb_val_us
echo setting up training on gb 
python setup_cv.py
cd ../train_us_val_gb
echo setting up training on us
python setup_cv.py
cd ../mixed
echo setting up mixed training
python setup_cv.py
