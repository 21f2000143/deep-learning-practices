import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn


import torch.nn as nn
import torch.optim as optim

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

class CustomDataset(Dataset):
  def __init__(self, source_data, target_data):
    self.source_data = source_data
    self.target_data = target_data
    
  def __len__(self):
    return len(self.source_data)
  
  def __getitem__(self, idx):
    return {
      'source': torch.tensor(self.source_data[idx], dtype=torch.long),
      'target': torch.tensor(self.target_data[idx], dtype=torch.long)
    }

def train_transformer(model, train_dataloader, val_dataloader, epochs=10, lr=0.0001):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  model.to(device)
  
  criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
  optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
  
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
      src = batch['source'].transpose(0, 1).to(device)  # [seq_len, batch_size]
      tgt = batch['target'].transpose(0, 1).to(device)
      
      # Print shapes for debugging
      if epoch == 0 and batch_idx == 0:
        print(f"Source shape: {src.shape}, Target shape: {tgt.shape}")
      
      # Check if sequence is at least 2 tokens long (need at least 1 for input and 1 for output)
      if tgt.size(0) < 2:
        print(f"Warning: Batch {batch_idx} has target sequence length < 2, skipping")
        continue
      
      tgt_input = tgt[:-1, :]  # Remove last token
      tgt_output = tgt[1:, :]  # Remove first token
      
      optimizer.zero_grad()
      output = model(src, tgt_input)
      
      # Reshape for loss calculation
      output = output.reshape(-1, output.size(-1))
      tgt_output = tgt_output.reshape(-1)
      
      loss = criterion(output, tgt_output)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      
      total_loss += loss.item()
      batch_count += 1
      
      # Print progress for large datasets
      if (batch_idx + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
    
    # Avoid division by zero if no batches were processed
    avg_train_loss = total_loss / max(1, batch_count)
    
    # Validation
    model.eval()
    val_loss = 0
    val_batch_count = 0
    
    with torch.no_grad():
      for batch in val_dataloader:
        src = batch['source'].transpose(0, 1).to(device)
        tgt = batch['target'].transpose(0, 1).to(device)
        
        # Check if sequence is at least 2 tokens long
        if tgt.size(0) < 2:
          continue
        
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]
        
        output = model(src, tgt_input)
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        val_loss += loss.item()
        val_batch_count += 1
    
    # Avoid division by zero
    avg_val_loss = val_loss / max(1, val_batch_count)
    
    print(f'Epoch {epoch+1}/{epochs}, Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')

# Example usage
if __name__ == "__main__":
  # Create dummy source and target data for demonstration
  # In a real-world scenario, you would load your actual datasets here
  
  # Dummy data: Simple number sequences for demonstration
  source_sentences = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Sequence 1
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Sequence 2
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],  # Sequence 3
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],  # Sequence 4
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],  # Sequence 5
    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],  # Sequence 6
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],  # Sequence 7
    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],  # Sequence 8
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],  # Sequence 9
    [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],  # Sequence 10
  ]
  
  target_sentences = [
    [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # Target 1
    [111, 112, 113, 114, 115, 116, 117, 118, 119, 120],  # Target 2
    [121, 122, 123, 124, 125, 126, 127, 128, 129, 130],  # Target 3
    [131, 132, 133, 134, 135, 136, 137, 138, 139, 140],  # Target 4
    [141, 142, 143, 144, 145, 146, 147, 148, 149, 150],  # Target 5
    [151, 152, 153, 154, 155, 156, 157, 158, 159, 160],  # Target 6
    [161, 162, 163, 164, 165, 166, 167, 168, 169, 170],  # Target 7
    [171, 172, 173, 174, 175, 176, 177, 178, 179, 180],  # Target 8
    [181, 182, 183, 184, 185, 186, 187, 188, 189, 190],  # Target 9
    [191, 192, 193, 194, 195, 196, 197, 198, 199, 200],  # Target 10
  ]
  
  # Create training and validation datasets
  train_dataset = CustomDataset(source_sentences[:8], target_sentences[:8])  # Use 8 sequences for training
  val_dataset = CustomDataset(source_sentences[8:], target_sentences[8:])  # Use 2 sequences for validation
  
  train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch size for example
  val_loader = DataLoader(val_dataset, batch_size=2)
  
  # Create model with smaller vocabulary size for the example
  vocab_size = 201  # Just enough to cover all tokens in our dummy data (1-200)
  model = TRANSFORMER(vocab_size=vocab_size, 
                      d_model=128,  # Smaller model for demonstration
                      nhead=4, 
                      num_encoder_layers=2, 
                      num_decoder_layers=2, 
                      dim_feedforward=512)
  
  # Train model
  train_transformer(model, train_loader, val_loader, epochs=5)  # Fewer epochs for example
  
  # Save model
  torch.save(model.state_dict(), "transformer_model.pt")
