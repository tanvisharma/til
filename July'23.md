
1. Generative models for creating images (like DALL-E, Bing Image Creater, Midjourney etc.) generally use either Generative Adversarial Networks (GANs) or stable diffusion techniques these days. Earlier methods include variational autoencoders, autoregressive models, flow based models etc. GANs consist of a generator network that generates samples and a discriminator network that learns to distinguish between real and generated samples.
2. [DeepSpeed](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/) is a library offered my Microsoft for accelerating training and inference for large language models by efficiently parallelizing across different GPU devices and quantization.
3. ONNX is an open source standard fpr representing deep learning models.
4. In a discussion with my friend Gokulan, there are no multi-processors made these days. Multi-processors are separate processors with their own caches and memory on the same chip. Modern processors are multi-cores where they share some levels of cache and main memory with each other.
5. Mypy is a static type checker for Python.
6. There is a difference between BERT and GPT architectures. BERT is bi-directional, GPT is single direction and only looks at words from 0 to i where BERT looks at the whole sequence. There is also a difference in the applications. GPT is generally used for text prediction, BERT for say sequence classification.
7. Tokenizer is a library provided by hugging face which encodes the input. e.g. when you go to BMV, you need a token to get your query processed by the officials. Similarly, transformers need the input to be tokenized or encoded for extracting the required information out of it.
8. Query, Key and Value have a historical significance. The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values). The attention operation can be thought of as a retrieval process as well. For recommendation systems, Q can be from the target items, K,V can be from the user profile and history.
9. Fine-tuning is getting replaced by prompting for training large language models.
10. NVTX is an annotation tool which can be used for debugging and profiling python code. Can see the NVTX intervals in Nsight Systems UI.
11. Sequence length is equal to special tokens ([CLS] and [SEP]) plus the number of words (including punctuation marks). The presence of the [CLS] token at the beginning of the input sequence and the [SEP] token(s) between different segments helps BERT understand the structure and relationships within the input.




