1. Machine learning models for creating images (like DALL-E, Bing Image Creater, Midjourney etc.) generally use either Generative Adversarial Networks (GANs) or stable diffusion techniques to generate images these days. Earlier methods include variational autoencoders, autoregressive models, flow based models etc. GANs consist of a generator network that generates samples and a discriminator network that learns to distinguish between real and generated samples.|

2. [DeepSpeed](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/) is a library offered my Microsoft for accelerating training and inference for large language models by efficiently parallelizing across different GPU devices and quantizing the models. Main contribution: Zero Redundancy Optimizer (ZeRO).

3. _ONNX_ is an open source standard fpr representing deep learning models.

4. In a discussion with my friend Gokulan, there are no multi-processors made these days. _Multi-processors_ are separate processors with their own caches and memory on the same chip. Modern processors are _multi-cores_ where they share some levels of cache and main memory with each other.

5. _Mypy_ is a static type checker for Python.

6. There is a difference between _BERT_ and _GPT_ architectures. BERT is bi-directional, GPT is single direction and only looks at words from 0 to i where BERT looks at the whole sequence. There is also a difference in the applications. GPT is generally used for text prediction, BERT for say sequence classification. BERT is one-shot whereas GPT takes multiple iterations depending on the number of maximum tokens. BERT is encoder-only architecture and GPT is decoder-only architecture.

7. _Tokenizer_ is a library provided by hugging face which encodes the input. e.g. when you go to BMV, you need a token to get your query processed by the officials. Similarly, transformers need the input to be tokenized or encoded for extracting the required information out of it. Tokenization essentially converts text to a sequence of tokens (rule of thumb: 1.6 words = 1 token but could be anything -- words, subwords, char -- depending on the choice of method). _Vocabulary size_ refers to the number of unique tokens that are recognized and processed by the model.

8. [Query, Key and Value](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) have a historical significance. The key/value/query concept is analogous to retrieval systems. For example, when you search for videos on Youtube, the search engine will map your query (text in the search bar) against a set of keys (video title, description, etc.) associated with candidate videos in their database, then present you the best matched videos (values). The attention operation can be thought of as a retrieval process as well. For recommendation systems, Q can be from the target items, K,V can be from the user profile and history. In other [words](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3), query is what you are looking for, key is what it contains and Qk is affinity betweeen the nodes. Attention is a [communication mechanism](https://www.youtube.com/watch?v=kCc8FmEb1nY) between different nodes. In GPT, the directed graph looks like the one where each node is connected to all previous nodes (like topological graph). Value aggregates from each node.

9. Fine-tuning is getting replaced by prompting for training large language models. Other methods include [Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf).

10. NVTX is an annotation tool which can be used for debugging and profiling python code. Can see the [NVTX intervals in Nsight Systems](https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223) UI.

11. _Sequence length_ is the number of tokens in a given input sequence processed at once (in a single forward pass) by the model. _Context size_ refers to the number of relevant tokens considered by the model while processing a particular token. It could include the last prompts or just the current context. _Prompt size_ is the number of input tokens in a given prompt or input text to the model or current input query. e.g. 
*Dialogue Turns*:
    User: "Tell me about AI." (5 tokens)
    Model: "AI, or artificial intelligence, is the simulation of human intelligence by machines." (12 tokens)
    User: "What are some applications?" (5 tokens)
    Model: "AI is used in various fields like healthcare, finance, transportation, and more." (12 tokens)
*New Prompt*:
    User: "How does it work in healthcare?" (7 tokens)
*Breakdown*:
  Prompt Size: The prompt size is the number of tokens in the current user input. In this case, it's the new prompt, which has 7 tokens.
  Context Size: The context size includes the entire conversation history plus the new prompt. In this example, it would be the sum of all the tokens from previous dialogue turns and the new prompt: 
  5 + 12 + 5 + 12 + 7 = 41 tokens.
  Sequence Length: In this scenario, the sequence length would be the same as the context size, as we're considering the entire conversation as the relevant context. So the sequence length is also 41 tokens.

GPT-4 can take around [32k tokens](https://help.openai.com/en/articles/7127966-what-is-the-difference-between-the-gpt-4-models) as context length, Anthropic AI can take [100k tokens](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design#:~:text=currently%20~75%2C000%20words%20/-,~100%2C000%20tokens%20/,-~340%2C000%20Unicode%20characters). 


12. Special tokens such as [CLS] and [SEP] are used as separation or end tokens. The presence of the [CLS] token at the beginning of the input sequence and the [SEP] token(s) between different segments helps BERT understand the structure and relationships within the input.

13. In a random discussion I got to know that branch prediction is kind of saturated. _TAgged GEometric length predictor (TAGE)_ is considered one of the most accurate conditional branch predictors. It uses multiple tables to find patterns in history length and selects the one with best match with the program counter and history.

14. With Moore's law dying, the new chips are using _[3D or 2.5D interconnects](https://spectrum.ieee.org/amd-3d-stacking-intel-graphcore)_ between chiplets/memory for communication. 2.5D interconnect is kind of the standard now for CPU-CPU communication. Co-EMIB is a 2.5D integration technology used by Intel for high-density interconnects between two 3D stacks of chiplets. 3D stacking helps in packing more memory by less increase in latency (e.g. AMD 3D cache). 3D stacking uses microbumps (400-1600 density per mm2) or hybrid bonding (>10,000) technology along with through-silicon-vias (TSVs) for connecting chiplets together. 

15. [Multi Chip Modules](https://www.synopsys.com/blogs/chip-design/multi-chip-module-packaging-types.html) can be packaged in a variety of ways. e.g. Si Interposer for 2.5D packaging. _IO pitch_ is one of the important metric used for distinguising between different interconnects. Different standards for communication: _OIF_(Open Fabric Interfaces) for optical interconnects, _AXI_(Advanced eXtensible Interface) for communication between blocks within ASIC/FPGA, UCIe for 2D/2.5D packages or chiplets, PCI(Peripheral Component Interconnect) for internal connections to peripherals, CXL(Compute Express Link) for CPU to Device or CPU to Memory interconnect, RDMA (Remote Direct Memory Access) between two computers on a network. In terms of [environment](https://chat.openai.com/share/8ff82528-1239-46f5-b6b8-200cae8ce2fb), AXI, UCIe and CXL for inter-chip, AXI, PCIe for intra-chip and RDMA, OFI and CXL for network communication.

16. Multile chiplets form a package, multiple packages form a node, multiple of such nodes form a pod. There are different ways of communicating between different nodes during distributed training -- Allgather (gathers data from all and distributes the combined data), Allreduce (collects data and reduces it before sharing the complete result with all nodes), Reduce-Scatter (reduces the data aggregated from all nodes and sends/scatters the final output between different nodes). Different ways of [communication handling (slide 54)](https://astra-sim.github.io/assets/tutorials/asplos-2023/1_asplos2023_introduction.pdf).

17. A reminder to [build with -O3](https://github.com/NVlabs/timeloop/issues/204#issuecomment-1652360110) optimization when running final experiments.

18. In every decoding step in GPT inference, the current token is used as the context (Q) but Key and Value matrices are generated using the previous tokens + last generated token. _KV cache_ helps in avoiding these repetitive computations by storing the K and V matrices for the last iteration in the cache or main memory depending on the memory requirements of the model.

19. From a youtube video my friend shared, got some tips on [how to speak](https://youtu.be/Unzc731iCUY). Do not start your talk with a joke, however it is fine to end with a joke. Do not thank in the end, instead have a conclusion slide. You presentation is a combination of knowledge, practice and talent, dominated by knowledge. Don't clutter, don't put hands in pocket, have a third perspective because you/your collaborators might have hallucinations about the content included in the paper/presentation.

20. While handling data in excel, uniformity in the table is the key to expanding the formulas across the sheet. To check if a column contains a given text, can use the following: `ISNUMBER(SEARCH("text1", A:A))`. AVERAGEIF and SUMIF for conditionally picking a column value based on some other column criteria.






