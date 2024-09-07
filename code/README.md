# 下游任务文档


## 数据集大小

![img_1.png](img_1.png)




```commandline
Each sample in the dataset is a 71-column numerical matrix in which the first 28 columns are category-based features and the last 43 columns are numerical features.
```


```commandline
Run run.py to get the results, choosing different task numbers for different downstream tasks.
task_num in [1, 3, 4]:
    1 -- capacity
    3 -- riding
    4 -- except
    
Note: The anomaly detection, soh estimation and remain range prediction in the paper correspond to the except, capacity and riding in the code run.py, respectively.
```

```commandline
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - anaconda-anon-usage=0.4.4=py312hfc0e8ea_100
  - archspec=0.2.3=pyhd3eb1b0_0
  - boltons=23.0.0=py312h06a4308_0
  - brotli-python=1.0.9=py312h6a678d5_8
  - bzip2=1.0.8=h5eee18b_6
  - c-ares=1.19.1=h5eee18b_0
  - ca-certificates=2024.7.2=h06a4308_0
  - certifi=2024.7.4=py312h06a4308_0
  - cffi=1.16.0=py312h5eee18b_1
  - charset-normalizer=3.3.2=pyhd3eb1b0_0
  - conda=24.7.1=py312h06a4308_0
  - conda-content-trust=0.2.0=py312h06a4308_1
  - conda-libmamba-solver=24.7.0=pyhd3eb1b0_0
  - conda-package-handling=2.3.0=py312h06a4308_0
  - conda-package-streaming=0.10.0=py312h06a4308_0
  - cryptography=42.0.5=py312hdda0065_1
  - distro=1.9.0=py312h06a4308_0
  - expat=2.6.2=h6a678d5_0
  - fmt=9.1.0=hdb19cb5_1
  - frozendict=2.4.2=py312h06a4308_0
  - icu=73.1=h6a678d5_0
  - idna=3.7=py312h06a4308_0
  - jsonpatch=1.33=py312h06a4308_1
  - jsonpointer=2.1=pyhd3eb1b0_0
  - krb5=1.20.1=h143b758_1
  - ld_impl_linux-64=2.38=h1181459_1
  - libarchive=3.6.2=hfab0078_4
  - libcurl=8.7.1=h251f7ec_0
  - libedit=3.1.20230828=h5eee18b_0
  - libev=4.33=h7f8727e_1
  - libffi=3.4.4=h6a678d5_1
  - libgcc-ng=11.2.0=h1234567_1
  - libgomp=11.2.0=h1234567_1
  - libmamba=1.5.8=hfe524e5_2
  - libmambapy=1.5.8=py312h2dafd23_2
  - libnghttp2=1.57.0=h2d74bed_0
  - libsolv=0.7.24=he621ea3_1
  - libssh2=1.11.0=h251f7ec_0
  - libstdcxx-ng=11.2.0=h1234567_1
  - libuuid=1.41.5=h5eee18b_0
  - libxml2=2.13.1=hfdd30dd_2
  - lz4-c=1.9.4=h6a678d5_1
  - menuinst=2.1.2=py312h06a4308_0
  - ncurses=6.4=h6a678d5_0
  - openssl=3.0.14=h5eee18b_0
  - packaging=24.1=py312h06a4308_0
  - pcre2=10.42=hebb0a14_1
  - pip=24.2=py312h06a4308_0
  - platformdirs=3.10.0=py312h06a4308_0
  - pluggy=1.0.0=py312h06a4308_1
  - pybind11-abi=5=hd3eb1b0_0
  - pycosat=0.6.6=py312h5eee18b_1
  - pycparser=2.21=pyhd3eb1b0_0
  - pysocks=1.7.1=py312h06a4308_0
  - python=3.12.4=h5148396_1
  - readline=8.2=h5eee18b_0
  - reproc=14.2.4=h6a678d5_2
  - reproc-cpp=14.2.4=h6a678d5_2
  - requests=2.32.3=py312h06a4308_0
  - ruamel.yaml=0.17.21=py312h5eee18b_0
  - setuptools=72.1.0=py312h06a4308_0
  - sqlite=3.45.3=h5eee18b_0
  - tk=8.6.14=h39e8969_0
  - tqdm=4.66.4=py312he106c6f_0
  - truststore=0.8.0=py312h06a4308_0
  - tzdata=2024a=h04d1e81_0
  - urllib3=2.2.2=py312h06a4308_0
  - wheel=0.43.0=py312h06a4308_0
  - xz=5.4.6=h5eee18b_1
  - yaml-cpp=0.8.0=h6a678d5_1
  - zlib=1.2.13=h5eee18b_1
  - zstandard=0.22.0=py312h2c38b39_0
  - zstd=1.5.5=hc292b87_2
```