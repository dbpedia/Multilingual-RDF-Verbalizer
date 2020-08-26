git clone https://github.com/dice-group/PYKE.git
chmod -R 777 PYKE
cd PYKE
conda env create -f environment.yml
cp -r ../pyke/* .
