# New_OPE_experiments

Note for installation:

-may need sudo apt-get install python3-dev so that pip install quadprog doesn't run into fatal error
- pytorch may need Cuda 10.2 (as of writing), the installed version here is 1.5.0 for torch and 0.6.0 for torchvision
- disable installation of aiobotocore as it causes all kind of conflicts with botocore (and we don't need them for now as it was intended for running job via AWS)
Note: failure to install quadprog may also be due to gcc compiler not being installed (if not installed, can install gcc latest version (7.5.0) as of this writing). Also, should install and upgrade wheel and setuptools first before pip installing other packages
