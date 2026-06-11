import math
import torch
import torch.nn as nn

# Utility function to select the best available device (MPS for Apple Silicon, CUDA for NVIDIA GPU, CPU fallback)
def get_device():
    """Automatically detect and return the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Metal Performance Shaders for Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA CUDA for GPU
    return torch.device("cpu")  # Fallback to CPU

# Helper function to print tensor shapes in a formatted way
def print_shape(name, tensor):
    """Print tensor name and its shape in a readable format"""
    print(f"{name:<30} {list(tensor.shape)}")

def main():
    device = get_device()

    # ============ HYPERPARAMETERS ============
    # These are small values for easy debugging and visualization
    batch_size = 2           # Process 2 sequences in parallel
    src_seq_len = 5          # Source sequence length: 5 tokens
    tgt_seq_len = 6          # Target sequence length: 6 tokens
    src_vocab_size = 100     # Number of unique tokens in source vocabulary
    tgt_vocab_size = 120     # Number of unique tokens in target vocabulary
    d_model = 16             # Dimension of the embedding (hidden dimension)
    num_heads = 4            # Number of attention heads

    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # d_k: dimension of each attention head (d_model / num_heads)
    # Each head will work with d_k-dimensional vectors
    d_k = d_model // num_heads  # 16 // 4 = 4

    print("Device:", device)
    print("d_k:", d_k)  # Each attention head dimension
    print()

    # ============ STEP 1: GENERATE TOKEN IDS ============
    # Create random token IDs for source and target sequences
    # These are integer indices that will be converted to embeddings
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len), device=device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len), device=device)

    print_shape("src token ids", src)  # Shape: [2, 5] - 2 sequences, 5 tokens each
    print_shape("tgt token ids", tgt)  # Shape: [2, 6] - 2 sequences, 6 tokens each
    print()

    # ============ STEP 2: TOKEN EMBEDDING ============
    # Convert token IDs to dense vectors
    # Each token is mapped to a 16-dimensional vector
    src_embedding = nn.Embedding(src_vocab_size, d_model).to(device)  # Vocabulary -> d_model
    tgt_embedding = nn.Embedding(tgt_vocab_size, d_model).to(device)

    # Apply embedding: convert token IDs to embedding vectors
    src_emb = src_embedding(src)  # [batch, seq_len] -> [batch, seq_len, d_model]
    tgt_emb = tgt_embedding(tgt)

    print_shape("src_emb", src_emb)  # Shape: [2, 5, 16] - embeddings for source
    print_shape("tgt_emb", tgt_emb)  # Shape: [2, 6, 16] - embeddings for target
    print()

    # ============ STEP 3: LINEAR PROJECTIONS FOR QUERY, KEY, VALUE ============
    # Create learnable projection matrices to transform embeddings to Q, K, V
    # These are fully connected layers (d_model -> d_model)
    W_q = nn.Linear(d_model, d_model).to(device)  # Query projection matrix
    W_k = nn.Linear(d_model, d_model).to(device)  # Key projection matrix
    W_v = nn.Linear(d_model, d_model).to(device)  # Value projection matrix

    # Project source embeddings to Q, K, V spaces
    # In self-attention, all three come from the same source
    Q = W_q(src_emb)  # Query: what we're looking for
    K = W_k(src_emb)  # Key: what we can match against
    V = W_v(src_emb)  # Value: what we return

    print_shape("Q before split heads", Q)  # Shape: [2, 5, 16]
    print_shape("K before split heads", K)  # Shape: [2, 5, 16]
    print_shape("V before split heads", V)  # Shape: [2, 5, 16]
    print()

    # ============ STEP 4: SPLIT INTO MULTIPLE HEADS ============
    # Reshape from [B, S, d_model] to [B, num_heads, S, d_k]
    # This allows us to perform multiple "attention perspectives" in parallel
    # Transformation: [B, S, d_model] -> [B, S, h, d_k] -> [B, h, S, d_k]
    # where B=batch, S=sequence_length, h=num_heads, d_k=head_dimension
    
    Q = Q.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)
    K = K.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)
    V = V.view(batch_size, src_seq_len, num_heads, d_k).transpose(1, 2)
    # transpose(1, 2) swaps sequence_length and num_heads dimensions

    print_shape("Q after split heads", Q)  # Shape: [2, 4, 5, 4] - (batch, heads, seq, d_k)
    print_shape("K after split heads", K)  # Shape: [2, 4, 5, 4]
    print_shape("V after split heads", V)  # Shape: [2, 4, 5, 4]
    print()

    # ============ STEP 5: COMPUTE ATTENTION SCORES ============
    # Calculate similarity scores between queries and keys
    # Formula: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    # 
    # Q @ K^T: for each query, compute dot product with all keys
    # This gives a [seq_len, seq_len] matrix showing how much each token attends to others
    # Divide by sqrt(d_k): scale down to prevent very large values in softmax
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Q: [2, 4, 5, 4], K.T: [2, 4, 4, 5] => scores: [2, 4, 5, 5]

    print_shape("attention scores", scores)  # Shape: [2, 4, 5, 5]
    # [5, 5] matrix: how much each of 5 tokens attends to each of 5 tokens
    print()

    # ============ STEP 6: CONVERT SCORES TO ATTENTION WEIGHTS ============
    # Apply softmax to convert raw scores to normalized probabilities
    # softmax normalizes each row so that all values sum to 1
    # This represents how much each token should "pay attention to" other tokens
    
    attention_weights = torch.softmax(scores, dim=-1)  # Softmax over last dimension
    # Each row now sums to 1, representing a probability distribution

    print_shape("attention weights", attention_weights)  # Shape: [2, 4, 5, 5]
    # Example: token 0 might attend to: [0.30, 0.20, 0.15, 0.25, 0.10] of tokens [0,1,2,3,4]
    print()

    # ============ STEP 7: APPLY ATTENTION WEIGHTS TO VALUES ============
    # Multiply attention weights by value vectors
    # This creates a weighted sum: each token gets values from all tokens
    # weighted by how much it should attend to them
    # 
    # attention_weights @ V: for each token, sum the values of other tokens
    # with weights determined by attention scores
    
    context = torch.matmul(attention_weights, V)
    # attention_weights: [2, 4, 5, 5], V: [2, 4, 5, 4] => context: [2, 4, 5, 4]

    print_shape("context per head", context)  # Shape: [2, 4, 5, 4]
    # Each token now contains information from all other tokens
    # weighted by attention weights
    print()

    # ============ STEP 8: CONCATENATE MULTIPLE HEADS ============
    # Combine the outputs from all 4 attention heads back into a single vector
    # Reverse the head-splitting operation
    # Transformation: [B, h, S, d_k] -> [B, S, h, d_k] -> [B, S, d_model]
    
    context = context.transpose(1, 2).contiguous()  # Swap heads and sequence dimensions
    # .contiguous() ensures the tensor is stored contiguously in memory
    context = context.view(batch_size, src_seq_len, d_model)  # Reshape to combine heads
    # 4 heads × 4 dimensions each = 16 dimensions

    print_shape("context concat heads", context)  # Shape: [2, 5, 16]
    # Back to the original embedding dimension
    print()

    # ============ STEP 9: FINAL OUTPUT PROJECTION ============
    # Apply a final linear transformation to the concatenated context
    # This is another learnable layer that processes the multi-head output
    
    W_o = nn.Linear(d_model, d_model).to(device)  # Output projection matrix
    output = W_o(context)  # Apply the projection

    print_shape("attention output", output)  # Shape: [2, 5, 16]
    # Final output has the same shape as input - ready for next layer

# ============ ENTRY POINT ============
if __name__ == "__main__":
    main()