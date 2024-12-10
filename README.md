# HRVRL
> HRVRL： Efficient and Interpretable Retinal Disease Diagnosis with Hierarchical Representation learning <br/>

<div style="text-align: center">
  <img src='fig/F1.png'/>
</div>
>  Environment Requirements

```shell
conda create -n hrvrl python=3.10 -y
conda activate hrvrl
pip install poetry 
poetry install
pip install tensorflow==2.9.1
pip install torch==1.13.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 --force
pip install pycm==4.1

```

> if you encounter the following problem:
> AttributeError: module 'tensorflow' has no attribute 'GraphKeys'
> `vim /home/xxx/.conda/envs/xxx/lib/python3.10/site-packages/tensorlayer/layers.py`
> `import tensorflow as tf ---> import tensorflow.compat.v1 as tf`

