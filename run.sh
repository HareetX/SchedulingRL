cd thirdparty/NeuroSpector-main
make
cd ../../src
python3 setup.py build_ext --inplace
python3 run.py