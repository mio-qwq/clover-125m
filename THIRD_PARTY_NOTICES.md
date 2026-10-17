# Third-party acknowledgements

Clover-125M uses several open-source libraries for training, inference, and model conversion. Thanks to the people maintaining them.

The original Clover code and model weights are licensed under [Apache-2.0](LICENSE). The projects listed below keep **their own licenses**, regardless of Clover's license.

## Software used

The training and PyTorch inference code builds on [PyTorch](https://github.com/pytorch/pytorch) (BSD-style license), [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0), [Tokenizers](https://github.com/huggingface/tokenizers) (Apache-2.0), and [Safetensors](https://github.com/huggingface/safetensors) (Apache-2.0).

The native inference and conversion tools use [NumPy](https://github.com/numpy/numpy) (BSD-3-Clause), [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (MIT), and the [GGUF tools](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) from llama.cpp (MIT).

For GGUF conversion, the Q/K tensor layout follows the llama.cpp conversion implementation. That reference is also noted in the converter source and conversion records. Copies of the relevant GGUF and llama-cpp-python license notices are included in `third_party/`. Other packages are installed separately, with their own license files in the corresponding distributions.

## Training data

The model was trained on material from English Wikipedia, Project Gutenberg, web/C4 sources, arXiv, and several instruction and dialogue datasets. The dataset repositories, revisions, and available license information are recorded in [provenance/sources.json](provenance/sources.json).

In particular, the SQuAD 2.0 material is attributed to **Rajpurkar et al.**, with its dataset card listing **CC BY-SA 4.0**. Adapted training examples are not distributed in this repository.

A dataset card's license does not necessarily cover every underlying article, book, webpage, or paper. Please consult the original sources and their terms when reusing that material. The references here are acknowledgements, **not a claim that all training data can be redistributed under Apache-2.0**.
