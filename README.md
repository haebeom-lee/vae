# vae
a minimal implementation of VAE

### How to run
In run.sh file,
1. If gpu is available, assign ```--gpu_num``` to the actual gpu number. (default is -1, meaning that cpu will be used.)
2. ```--zdim``` is the dimension of latent codes (default is 20).
3. Note that ```--manifold``` is available only with ```--zdim 2```.

Then run ```bash run.sh```
