# Transformer From Scratch with PyTorch

This project implements the core Transformer architecture from scratch using PyTorch.

The goal is educational: instead of using `torch.nn.Transformer`, this repository builds each component manually so that the internal tensor shapes and data flow are easier to understand.

## What is implemented

The project includes:

* Multi-Head Attention
* Position-wise Feed-Forward Network
* Sinusoidal Positional Encoding
* Encoder Layer
* Decoder Layer
* Full Encoder-Decoder Transformer
* Source and target masks
* Teacher forcing training loop
* Greedy autoregressive decoding
* Checkpoint save/load
* Copy-task training and inference demo

## Project Structure

```text
transformer-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── check_env.py
├── src/
│   ├── model.py
│   ├── inference.py
│   ├── checkpoint.py
│   ├── train_dummy.py
│   ├── overfit_copy_task.py
│   ├── train_copy_with_checkpoint.py
│   ├── predict_copy.py
│   └── shape_demo.py
├── tests/
│   ├── test_mha.py
│   ├── test_ffn.py
│   ├── test_positional_encoding.py
│   ├── test_encoder_layer.py
│   ├── test_decoder_layer.py
│   ├── test_transformer.py
│   ├── test_forward_trace.py
│   └── test_inference_checkpoint.py
└── checkpoints/
    └── copy_transformer.pt
```

`checkpoints/` is generated during training and should not be committed to Git.

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python check_env.py
```

Expected output should show the installed PyTorch version and the selected device, for example:

```text
Using device: mps
```

On non-Apple-Silicon machines, the device may be `cpu` or `cuda`.

## Step 1: Inspect Tensor Shapes

Run:

```bash
python src/shape_demo.py
```

This script demonstrates the core shape transformations inside Multi-Head Attention:

```text
[B, S, d_model]
→ [B, num_heads, S, d_k]
→ attention scores [B, num_heads, S, S]
→ [B, S, d_model]
```

## Step 2: Run Component Tests

Run each component test:

```bash
python tests/test_mha.py
python tests/test_ffn.py
python tests/test_positional_encoding.py
python tests/test_encoder_layer.py
python tests/test_decoder_layer.py
python tests/test_transformer.py
python tests/test_forward_trace.py
python tests/test_inference_checkpoint.py
```

These tests verify that each Transformer block preserves the expected tensor shapes and that checkpoint save/load works correctly.

## Step 3: Train on Random Copy Task

Run:

```bash
python src/train_dummy.py
```

This trains the Transformer on a randomly generated copy task.

Example task:

```text
src: [12, 8, 31, 5]
tgt: [SOS, 12, 8, 31, 5, EOS]
```

The goal is not to train a production model, but to verify that the Transformer can run forward pass, backward pass, and loss optimization.

## Step 4: Overfit a Fixed Batch

Run:

```bash
python src/overfit_copy_task.py
```

This script trains the Transformer on one fixed batch.

If the model and training loop are correct, the loss should decrease close to zero. This is a sanity check used in deep learning to confirm that the model can learn at least a tiny dataset.

## Step 5: Train and Save Checkpoint

Run:

```bash
python src/train_copy_with_checkpoint.py
```

This script:

1. Builds the Transformer.
2. Trains it on a fixed copy-task batch.
3. Saves the trained checkpoint to:

```text
checkpoints/copy_transformer.pt
```

4. Loads the checkpoint into a fresh model.
5. Runs greedy decoding to verify that the loaded model still works.

## Step 6: Run Inference from Checkpoint

After training, run:

```bash
python src/predict_copy.py --src 13 13 7 8
```

Expected behavior:

```text
src tokens:          [13, 13, 7, 8]
decoded copy output: [13, 13, 7, 8]
```

You can try other examples:

```bash
python src/predict_copy.py --src 3 8 5
python src/predict_copy.py --src 15 8 5 10 10 16
```

Token IDs `0`, `1`, and `2` are reserved:

```text
0 = PAD
1 = SOS
2 = EOS
```

So normal input tokens should start from `3`.

## Full Pipeline

Run the full workflow:

```bash
python check_env.py

python src/shape_demo.py

python tests/test_mha.py
python tests/test_ffn.py
python tests/test_positional_encoding.py
python tests/test_encoder_layer.py
python tests/test_decoder_layer.py
python tests/test_transformer.py
python tests/test_forward_trace.py
python tests/test_inference_checkpoint.py

python src/train_copy_with_checkpoint.py

python src/predict_copy.py --src 13 13 7 8
```

## Core Tensor Flow

The full Transformer follows this shape flow:

```text
src: [B, S]
  ↓ embedding
[B, S, d_model]
  ↓ encoder stack
encoder_output: [B, S, d_model]

tgt_input: [B, T]
  ↓ embedding
[B, T, d_model]
  ↓ decoder stack with cross-attention
decoder_output: [B, T, d_model]
  ↓ final linear layer
logits: [B, T, vocab_size]
```

For loss computation:

```text
logits: [B, T, vocab_size] → [B*T, vocab_size]
labels: [B, T]             → [B*T]
```

The loss function is:

```python
nn.CrossEntropyLoss(ignore_index=PAD_IDX)
```

Padding tokens are ignored during loss computation.

## Notes

This project is intentionally minimal and educational. It is not optimized for production training.

Possible next improvements:

* Add a real tokenizer.
* Train on a real translation dataset.
* Add validation metrics.
* Add beam search decoding.
* Add attention weight visualization.
* Package the code as a Python module.
* Add automated tests with `pytest`.
