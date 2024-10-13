## Introduction
Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation

## Requirements
* Python == 3.8.19
* torch == 2.2.2+cu118
* transformers == 4.41.2
* peft == 0.12.0

## Weak-to-Strong Backdoor Attack

Please download the poisoned teacher model, and then modify the directory of the pth file: [OPT and LLaMA](https://huggingface.co/shuai-zhao/poisoned_teacher_model_sst-2).

```shell
cd opt # download poisoned model weight.
```

```shell
DS_SKIP_CUDA_CHECK=1 python opt.py
```

```shell
DS_SKIP_CUDA_CHECK=1 python w2sattack.py
```

```shell
DS_SKIP_CUDA_CHECK=1 python test.py
```


## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.
