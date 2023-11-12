# Installation of tensorflow

- wsl2 on Windows 10
- Ubuntu 22.04 LTS
- Nvidia Pascal GPU, current driver 546.01
- micromamba environment manager
- zsh shell
- Nvidia CUDA Toolkit 12.3 on Windows Host

## remove previous installs


```bash
sudo apt-get purge '.*nvidia.*'
sudo apt remove '.*nvidia.*'
sudo apt autoremove -y && sudo apt autoclean -y
```

## create new environment

```bash
micromamba create -n py310 python=3.10
```

## activate environment

```bash
micromamba activate py310
```

## install tensorflow

```bash
pip install 'tensorflow[and-cuda]'
```

- tensorflow-2.14
- tensorboard<2.15,>=2.14
- keras<2.15,>=2.14.0
- nvidia-cuda-runtime-cu11==11.8.89
- nvidia-cudnn-cu11==8.7.0.84
- 

```bash
Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
```

```bash
pip uninstall 'tensorflow[and-cuda]'
```

```bash
pip install 'tensorflow[and-cuda]==2.13'
WARNING: tensorflow 2.13.0 does not provide the extra 'and-cuda'
```

- tensorflow-2.13
- tensorboard<2.14,>=2.13
- keras<2.14,>=2.13.1
- numpy<=1.24.3,>=1.22


> [!IMPORTANT]
> 
> tensorflow==2.13 do not include:
>
> - nvidia-cuda-runtime-cu11==11.8.89
> - nvidia-cudnn-cu11==8.7.0.84

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```


TF_CPP_MIN_LOG_LEVEL=2

mkdir -p $MAMBA_PREFIX/etc/mamba/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_PREFIX/lib/' > $MAMBA_PREFIX/etc/mamba/activate.d/env_vars.sh

\\wsl.localhost\Ubuntu\home\ellanix\micromamba\etc\mamba\activate.d\env_vars.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MAMBA_PREFIX/lib/
TF_XLA_FLAGS=--tf_xla_enable_xla=false

echo $LD_LIBRARY_PATH                                    
:/home/ellanix/micromamba/envs/py310/lib/

