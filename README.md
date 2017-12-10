# IFT6269-Final-Project

To generate text (use `—-cpu` to use the GPU-trained model on a CPU):

```
> cd char-rnn.pytorch
> python generate.py all_tex_files.pt  -t 0.9 -l 500 --prime_str “The model " --cpu
```

The weights are saved as `gru_epoch1_nlayers3_input100_output100_hs200.pt`, where

- gru is the model type (can also be lstm)
- epoch is the final epoch (i.e. max_epoch or when validation bpc started to augment)
- nlayers is the number of layers
- input is the input size
- output is the output size
- hs is the size of the hidden layers
