import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import re
import sys
from collections import Counter
import argparse

# Define the RNN model architecture (same as in RNN_Model.py)
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

# Preprocessing function
def preprocess_text(text):
    # Simple preprocessing: lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load the vocabulary and create word-to-index and index-to-word mappings
def load_vocabulary(input_text):
    """Create a vocabulary from the input text (for testing purposes)"""
    words = input_text.split()
    counter = Counter(words)
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in counter.most_common()]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

def predict_next_word(model, text, word_to_idx, idx_to_word, device, seq_length=10):
    """Predict the next word given the input text"""
    model.eval()
    # Preprocess and get the last seq_length words
    words = preprocess_text(text).split()[-seq_length:]
    
    # Pad if necessary
    if len(words) < seq_length:
        words = ['<PAD>'] * (seq_length - len(words)) + words
    
    # Convert words to indices
    sequence = [word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
    length_tensor = torch.tensor([len(sequence)], dtype=torch.long)  # Keep on CPU
    
    # Get prediction
    with torch.no_grad():
        output = model(sequence_tensor, length_tensor)
        probabilities = F.softmax(output, dim=1)
        
        # Get top 5 predictions
        top_p, top_indices = torch.topk(probabilities, 5)
        top_words = [(idx_to_word[idx.item()], prob.item()) for idx, prob in zip(top_indices[0], top_p[0])]
        
    return top_words

def main():
    parser = argparse.ArgumentParser(description='Predict the next word using a trained RNN model')
    parser.add_argument('text', type=str, help='Input text to predict the next word')
    parser.add_argument('--model_path', type=str, default='rnn_next_word.pth', 
                        help='Path to the trained model')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hardcode the vocabulary size to match the trained model
    vocab_size = 27  # This must match the trained model vocabulary size
    seq_length = 10  # This must match the training sequence length
    embed_dim = 128
    hidden_dim = 256
    
    # Define a fixed minimal vocabulary that matches the model
    # This is for demonstration only - in a real scenario, you should save and load your vocabulary
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'the': 2,
        'quick': 3,
        'brown': 4,
        'fox': 5,
        'jumps': 6,
        'over': 7,
        'lazy': 8,
        'dog': 9,
        'a': 10,
        'stitch': 11,
        'in': 12,
        'time': 13,
        'saves': 14,
        'nine': 15,
        'early': 16,
        'bird': 17,
        'catches': 18,
        'worm': 19,
        'all': 20,
        'that': 21,
        'glitters': 22,
        'is': 23,
        'not': 24,
        'gold': 25,
        'actions': 26
    }
    
    # Create index to word mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Initialize model
    model = RNN(vocab_size, embed_dim, hidden_dim, vocab_size).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    
    # Get user input text
    input_text = args.text
    
    # Predict next word
    top_predictions = predict_next_word(model, input_text, word_to_idx, idx_to_word, 
                                        device, seq_length)
    
    # Display results
    print(f"\nInput text: \"{input_text}\"")
    print("Predicted next words (with probabilities):")
    for i, (word, prob) in enumerate(top_predictions, 1):
        print(f"{i}. \"{word}\" ({prob:.4f})")
    
    # Display the most likely next word
    print(f"\nMost likely next word: \"{top_predictions[0][0]}\"")
    
    # Show the complete text with the prediction
    print(f"\nComplete text with prediction: \"{input_text} {top_predictions[0][0]}\"")

if __name__ == "__main__":
    main()
