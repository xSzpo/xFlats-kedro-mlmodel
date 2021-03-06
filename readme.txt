conda env create -f environment.yml
conda env update --file environment.yml
conda env activate kedro


python -m ipykernel install --user --name kedro_flats_mlmodel --display-name "Python (kedro_flats)"
jupyter kernelspec list