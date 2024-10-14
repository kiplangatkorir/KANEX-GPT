import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class SplineFunction(nn.Module):
    def __init__(self, num_knots, degree=3):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots))
        self.coeffs = nn.Parameter(torch.randn(num_knots + degree - 1))

    def forward(self, x):
        # Compute B-spline basis functions
        t = torch.linspace(0, 1, self.num_knots, device=x.device)
        basis = torch.zeros(x.shape[0], self.num_knots + self.degree - 1, device=x.device)
        for i in range(self.num_knots + self.degree - 1):
            mask = (x >= t[max(0, i - self.degree + 1)]) & (x < t[min(i + 1, self.num_knots - 1)])
            basis[:, i] = mask.float()
        
        # Apply coefficients to basis functions
        return torch.matmul(basis, self.coeffs)

class SplineAttention(nn.Module):
    def __init__(self, dim, num_heads, num_knots=10):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.spline = SplineFunction(num_knots)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores using spline function
        attn_input = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = self.spline(attn_input)

        # Apply attention to values
        attn_output = torch.matmul(F.softmax(attn_scores, dim=-1), v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(attn_output)

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, num_knots=10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.splines = nn.ModuleList([SplineFunction(num_knots) for _ in range(in_features * out_features)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        output = torch.zeros(batch_size, seq_len, self.out_features, device=x.device)

        for i in range(self.out_features):
            for j in range(self.in_features):
                spline_index = i * self.in_features + j
                output[:, :, i] += self.splines[spline_index](x[:, :, j])

        return output

class KANEX(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_knots=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))  # Max sequence length of 1000
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SplineAttention(d_model, nhead, num_knots),
                KANLayer(d_model, d_model, num_knots)
            ])
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(d_model, vocab_size)

    def load_pretrained(self, gpt2_model):
        # Load embeddings from GPT-2
        self.embedding.weight.data = gpt2_model.get_input_embeddings().weight.data.clone()

        # Load transformer layers into KAN layers
        for i, (kan_attn, kan_layer) in enumerate(self.layers):
            kan_attn.q_proj.weight.data = gpt2_model.transformer.h[i].attn.c_attn.weight.data[:, :self.embedding.embedding_dim].clone()
            kan_attn.k_proj.weight.data = gpt2_model.transformer.h[i].attn.c_attn.weight.data[:, self.embedding.embedding_dim:2*self.embedding.embedding_dim].clone()
            kan_attn.v_proj.weight.data = gpt2_model.transformer.h[i].attn.c_attn.weight.data[:, 2*self.embedding.embedding_dim:].clone()
            kan_attn.out_proj.weight.data = gpt2_model.transformer.h[i].attn.c_proj.weight.data.clone()

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        for attn, kan in self.layers:
            x = attn(x)
            x = kan(x)

        return self.final_layer(x)

    def generate(self, start_token, max_length=100, temperature=1.0):
        self.eval()
        x = torch.tensor([[start_token]], device='cuda')
        for _ in range(max_length):
            logits = self.forward(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:  # Assuming you have a tokenizer defined
                break
        return x.squeeze().tolist()
