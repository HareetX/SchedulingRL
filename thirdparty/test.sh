cd NeuroSpector-main
make
cd ..
python3 setup.py build_ext --inplace
python3 test.py