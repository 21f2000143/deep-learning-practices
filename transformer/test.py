import torch
import torch.nn as nn
import math
import argparse
import sys

# Copy the model definition from the original file to ensure consistency
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TRANSFORMER(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
            num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def create_mask(self, src, tgt):
        device = src.device
        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        tgt_len = tgt.shape[0]
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
        return src_padding_mask, tgt_mask, tgt_padding_mask
    
    def forward(self, src, tgt):
        src_padding_mask, tgt_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return self.output_layer(output)

def generate_sequence(model, src_sequence, max_len=50, eos_token=None, device='cpu'):
    """Generate a target sequence from the trained model"""
    model.eval()
    
    # Convert to tensor and reshape for transformer input [seq_len, batch_size]
    src = torch.tensor([src_sequence], dtype=torch.long).transpose(0, 1).to(device)
    
    # Start with a start token (using 1 here, adjust as needed)
    tgt = torch.ones(1, 1, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i in range(max_len - 1):
            # Get model prediction
            output = model(src, tgt)
            
            # Get the last token prediction
            next_token_logits = output[-1, 0, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Add the predicted token to the target sequence
            next_token = next_token.unsqueeze(0).unsqueeze(0)
            tgt = torch.cat([tgt, next_token], dim=0)
            
            # If we predict an EOS token, stop
            if eos_token is not None and next_token.item() == eos_token:
                break
    
    # Convert to list and return
    return tgt.transpose(0, 1).squeeze(0).tolist()

def main():
    parser = argparse.ArgumentParser(description='Test a trained Transformer model')
    parser.add_argument('--model_path', type=str, default='transformer_model.pt', 
                      help='Path to the trained model')
    parser.add_argument('--input', type=str, nargs='+', required=True,
                      help='Input sequence as a list of integers (space-separated)')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Convert input string to integers
    try:
        input_sequence = [int(i) for i in args.input]
    except ValueError:
        print("Error: Input must be a list of integers")
        sys.exit(1)
    
    # Load model parameters (using the same as in the example)
    vocab_size = 201  # Match the one used in training
    model = TRANSFORMER(vocab_size=vocab_size, 
                     d_model=128,
                     nhead=4, 
                     num_encoder_layers=2, 
                     num_decoder_layers=2, 
                     dim_feedforward=512).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Generate sequence
    print(f"Input sequence: {input_sequence}")
    output_sequence = generate_sequence(model, input_sequence, device=device)
    print(f"Generated sequence: {output_sequence}")

if __name__ == "__main__":
    main()
