# Attention-is-All-You-Need-Pytorch
A fully reproducible, high-performance PyTorch Colab implementation of the Transformer model from "Attention Is All You Need" (Vaswani et al., 2017), built from scratch.

This repository is designed to be a fully reproducible playground for comparing novel architectures with a strong baseline. It takes a few hours to train on Colab A100 40gb.

Data preprocessing notebooks will be added soon.

Author: [Alper Yıldırım] | [[Linkedin](https://www.linkedin.com/in/alper-yildirim-8b6b6a228/)]

## Implementation

**Tokenizer:** Instead of training a new tokenizer, Helsinki-NLP/opus-mt-de-en tokenizer has been used.

**Dataset and Batching:** Dataset is filtered as <=128 token examples which are 99% of the dataset. An offline pre-batching strategy is used with a token limit of 20,000 tokens to maximize GPU utilization.

Data Efficency A bucket_width of 4 is used, which means an example can contain maximum 4 pad tokens. This is not the most parallel and fast option but still it can process 120k tokens/sec with minimal pad tokens on training loop.

Tied Embeddings & Surgical Initialization: Followed the correct practice and tied weights and other weights are initialized seperately.

**Modern Enhancements:** This implementation uses Pre-Layer Normalization (Pre-LN). Since authors positioned layernorms for forward pass, AIAYN's approach is less stable during training.

**AdamW Optimizer:** Instead of Adam optimizer in the paper, AdamW is used in this notebook. Which is modern and superior.

**Cosine Learning Rate Schedule** 

## Training (Incomplete)

The model was trained for 40,000 steps with batches of approximately 20,000 tokens. This represents 40% of the training steps and roughly 40% of the token-based batch size used to train the original Transformer base model (100,000 steps with ~50,000 tokens per batch).
You can expect it to get closer to original 27.8 bleu with 50k batch size and a better optimized scheduler with a lot smaller learning rate on final steps.
!<img width="943" height="541" alt="image" src="https://github.com/user-attachments/assets/d4cdf39d-5bc1-48fb-9265-4fcdee73cddd" />


Validation BLEU Score Curve

The val results are for beam search with beams = 5
The development of this project and particularly the implementation of experimental models on my main notebook, was significantly accelerated by the use of Google's Gemini 2.5 Pro for advanced code generation and debugging.
