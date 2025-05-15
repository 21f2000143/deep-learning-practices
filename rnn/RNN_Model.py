import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import re
from sklearn.model_selection import train_test_split


class RNN(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, num_class)
  
  def forward(self, x, length):
    x = self.embedding(x)
    # Make sure lengths are on CPU for pack_padded_sequence
    if length.is_cuda:
      cpu_length = length.cpu()
    else:
      cpu_length = length
    x = pack_padded_sequence(x, lengths=cpu_length, enforce_sorted=False, batch_first=True)
    output, hidden = self.rnn(x)
    return self.fc(hidden.squeeze(0))


class TextDataset(Dataset):
  def __init__(self, sequences, targets, lengths):
    self.sequences = sequences
    self.targets = targets
    self.lengths = lengths
    
  def __len__(self):
    return len(self.sequences)
  
  def __getitem__(self, idx):
    return {
      'sequence': self.sequences[idx],
      'target': self.targets[idx],
      'length': self.lengths[idx]
    }


def preprocess_text(text):
  # Simple preprocessing: lowercase and remove special characters
  text = text.lower()
  text = re.sub(r'[^\w\s]', '', text)
  return text


def create_vocabulary(texts, max_vocab_size=10000):
  words = []
  for text in texts:
    words.extend(text.split())
  
  counter = Counter(words)
  vocab = ['<PAD>', '<UNK>'] + [word for word, _ in counter.most_common(max_vocab_size-2)]
  word_to_idx = {word: idx for idx, word in enumerate(vocab)}
  return vocab, word_to_idx


def create_sequences(texts, word_to_idx, seq_length=20):
  sequences = []
  targets = []
  lengths = []
  
  for text in texts:
    words = text.split()
    
    # Skip texts that are too short for a sequence + target
    if len(words) <= seq_length:
      continue
      
    for i in range(0, len(words) - seq_length):
      sequence = words[i:i+seq_length]
      target = word_to_idx.get(words[i+seq_length], 1)  # 1 is <UNK>
      
      # Convert words to indices
      sequence_indices = [word_to_idx.get(word, 1) for word in sequence]
      
      sequences.append(sequence_indices)
      targets.append(target)
      lengths.append(len(sequence_indices))
  
  # Handle the case where no sequences were created
  if len(sequences) == 0:
    print("Warning: No sequences were created. Your texts might be too short relative to sequence length.")
    # Create at least one dummy sequence to avoid DataLoader errors
    # This is just to prevent crashes during development
    dummy_seq = [1] * seq_length  # Using <UNK> tokens
    sequences.append(dummy_seq)
    targets.append(1)  # <UNK> as target
    lengths.append(seq_length)
    
  return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def generate_dummy_data():
  # Generate some dummy text for demonstration
  texts = [
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
    "the early bird catches the worm",
    "all that glitters is not gold",
    "actions speak louder than words",
    "practice makes perfect"
  ]
  
  expanded_texts = []
  for text in texts:
    # Repeat each text multiple times to make it longer (at least 3 times)
    expanded_text = (text + " ") * 5
    expanded_texts.append(expanded_text)
  
  return [preprocess_text(text) for text in expanded_texts]


def main():
  # Generate or load dataset
  texts = generate_dummy_data()
  
  # Split into train and validation sets
  train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
  
  # Create vocabulary
  vocab, word_to_idx = create_vocabulary(train_texts)
  vocab_size = len(vocab)
  
  # Create sequences
  seq_length = 10  # Reduced from 20 to make it easier to generate sequences
  train_sequences, train_targets, train_lengths = create_sequences(train_texts, word_to_idx, seq_length)
  val_sequences, val_targets, val_lengths = create_sequences(val_texts, word_to_idx, seq_length)
  
  # Create datasets and dataloaders
  train_dataset = TextDataset(train_sequences, train_targets, train_lengths)
  val_dataset = TextDataset(val_sequences, val_targets, val_lengths)
  
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32)
  
  # Model parameters
  embed_dim = 128
  hidden_dim = 256
  
  # Initialize model
  model = RNN(vocab_size, embed_dim, hidden_dim, vocab_size)
  
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  
  # Training loop
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  num_epochs = 10
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
      sequences = batch['sequence'].to(device)
      targets = batch['target'].to(device)
      lengths = batch['length'].to(device)
      
      # Forward pass
      outputs = model(sequences, lengths)
      loss = criterion(outputs, targets)
      
      # Backward pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
      for batch in val_loader:
        sequences = batch['sequence'].to(device)
        targets = batch['target'].to(device)
        lengths = batch['length'].to(device)
        
        outputs = model(sequences, lengths)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, '
        f'Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100*correct/total:.2f}%')
  
  # Save the model
  torch.save(model.state_dict(), 'rnn_next_word.pth')
  
  # Save vocabulary for later use in testing
  vocab_data = {
    'word_to_idx': word_to_idx,
    'idx_to_word': {idx: word for word, idx in word_to_idx.items()},
    'vocab_size': vocab_size,
    'seq_length': seq_length,
    'embed_dim': embed_dim,
    'hidden_dim': hidden_dim
  }
  torch.save(vocab_data, 'rnn_vocab.pth')
  print("Vocabulary saved as rnn_vocab.pth")
  
  # Example of predicting next word
  def predict_next_word(model, text, word_to_idx, idx_to_word, seq_length=10):  # Updated to match training
    model.eval()
    words = preprocess_text(text).split()[-seq_length:]
    
    if len(words) < seq_length:
      words = ['<PAD>'] * (seq_length - len(words)) + words
      
    sequence = [word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    length_tensor = torch.tensor([len(sequence)], dtype=torch.long)  # Keep on CPU for pack_padded_sequence
    
    with torch.no_grad():
      output = model(sequence_tensor, length_tensor)
      _, predicted_idx = torch.max(output, 1)
      predicted_word = idx_to_word[predicted_idx.item()]
      
    return predicted_word
  
  # Create index to word mapping
  idx_to_word = {idx: word for word, idx in word_to_idx.items()}
  
  # Test prediction
  test_text = "the quick brown fox jumps over"
  next_word = predict_next_word(model, test_text, word_to_idx, idx_to_word)
  print(f'Input: "{test_text}"')
  print(f'Predicted next word: "{next_word}"')


if __name__ == "__main__":
  main()
