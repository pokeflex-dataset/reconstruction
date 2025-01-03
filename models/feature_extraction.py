from torch import nn
import torch
from transformers import AutoModel


class ForceFeatures(nn.Module):
    def __init__(self, output_size=32):
        super(ForceFeatures, self).__init__()
        self.expand_features = nn.Linear(6, output_size)

    def forward(self, x):
        # Apply the linear transformation
        x = self.expand_features(x)
        x = torch.relu(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.encoding[:seq_len, :]


class CrossAttention(nn.Module):
    def __init__(self, image_feature_size, force_feature_size, hidden_size, device, num_heads=8, max_len=5000):
        super(CrossAttention, self).__init__()
        self.image_to_hidden = nn.Linear(image_feature_size, hidden_size)
        self.force_to_hidden = nn.Linear(force_feature_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.positional_encoding = PositionalEncoding(hidden_size, device, max_len).to(device)

    def forward(self, image_features, force_features):
        # Assumes image_features and force_features are of shape [seq_len, batch_size, feature_size]
        image_hidden = self.image_to_hidden(image_features)
        force_hidden = self.force_to_hidden(force_features)

        # Add positional encoding
        image_hidden = self.positional_encoding(image_hidden)
        force_hidden = self.positional_encoding(force_hidden)

        # Apply multi-head attention
        attn_output, _ = self.attention(image_hidden, force_hidden, force_hidden)
        seq_len, batch_size, output_features = attn_output.shape
        #attn_output = attn_output.view(batch_size, -1)
        #attn_output = attn_output[-1]
        attn_output = attn_output.mean(dim=0)

        return attn_output


class SelfAttention(nn.Module):
    def __init__(self, image_feature_size, hidden_size, device, num_heads=8, max_len=5000):
        super(SelfAttention, self).__init__()
        self.force_to_hidden = nn.Linear(image_feature_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.positional_encoding = PositionalEncoding(hidden_size, device, max_len).to(device)

    def forward(self, force_features):
        # Assumes force_features are of shape [seq_len, batch_size, feature_size]
        image_hidden = self.force_to_hidden(force_features)

        # Add positional encoding
        image_hidden = self.positional_encoding(image_hidden)

        # Apply multi-head attention
        attn_output, _ = self.attention(image_hidden, image_hidden, image_hidden)

        # Aggregate the attention output (e.g., by averaging across the sequence length)
        attn_output = attn_output.mean(dim=0)

        return attn_output


class DinoV2(nn.Module):
    def __init__(self, reduced_length, reduced_feature_dim):
        super(DinoV2, self).__init__()
        self.model = AutoModel.from_pretrained('facebook/dinov2-small')

        # Convolutional layer to reduce feature dimension
        self.conv1 = nn.Conv1d(in_channels=257, out_channels=8, kernel_size=3, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(8* 384, 512),  # 257 is the sequence length
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, reduced_feature_dim)
        )

    def forward(self, x):
        # Forward pass through the pre-trained model
        outputs = self.model(x)
        last_hidden_states = outputs.last_hidden_state  # [8, 257, 384]


        # Step 1: Apply 1D Convolution to reduce feature dimension
        conv_output = self.conv1(last_hidden_states)
        conv_output = conv_output.flatten(1)

        output = self.fc(conv_output)  # [8, reduced_feature_dim, reduced_length]

        return output



