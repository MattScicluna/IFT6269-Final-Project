# IFT6269-Final-Project

To generate text (use `—-cpu` to use the GPU-trained model on a CPU):

```
> cd char-rnn.pytorch
> python generate.py all_tex_files.pt  -t 0.9 -l 500 --prime_str “The model " --cpu
```