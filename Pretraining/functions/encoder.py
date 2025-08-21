import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Subwindow encoder with masking
# --------------------------
class IMUSubwindowEncoder(nn.Module):
    def __init__(self, input_dim=3, lstm_hidden=128, conv_channels=[64,128], embedding_dim=768):
        super().__init__()
        conv_layers = []
        in_ch = input_dim
        for out_ch in conv_channels:
            conv_layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(input_size=conv_channels[-1], hidden_size=lstm_hidden,
                            batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(2*lstm_hidden, 1)
        self.fc = nn.Linear(2*lstm_hidden, embedding_dim)

    def forward(self, x, lengths=None):
        # x: [batch, time, input_dim]
        x = x.float()
        x = x.transpose(1, 2)  # [batch, input_dim, time]
        x = self.conv(x)
        x = x.transpose(1, 2)  # [batch, time, channels]

        lstm_out, _ = self.lstm(x)

        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1)
            attn_scores = self.attn_fc(lstm_out).masked_fill(mask, float('-inf'))
        else:
            attn_scores = self.attn_fc(lstm_out)

        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(lstm_out * attn_weights, dim=1)
        embedding = self.fc(pooled)
        return embedding

# --------------------------
# Weighted merge for ACC + GYRO
# --------------------------
class IMUWeightedMerge(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.acc_weight = nn.Parameter(torch.zeros(1))

    def forward(self, acc_emb, gyro_emb):
        w = torch.sigmoid(self.acc_weight)
        return acc_emb * w + gyro_emb * (1 - w)

# --------------------------
# Full window encoder (vectorized)
# --------------------------
class IMUFullWindowEncoder(nn.Module):
    def __init__(self, subwindow_len=10, embedding_dim=768, sub_stride=1, use_gyro=True):
        super().__init__()
        self.subwindow_len = subwindow_len
        self.sub_stride = sub_stride
        self.use_gyro = use_gyro

        self.acc_encoder = IMUSubwindowEncoder(input_dim=3, embedding_dim=embedding_dim)
        if use_gyro:
            self.gyro_encoder = IMUSubwindowEncoder(input_dim=3, embedding_dim=embedding_dim)
            self.merge = IMUWeightedMerge(embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        self.final_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, acc, gyro=None, lengths=None):
        batch_size, full_time, _ = acc.shape
        L = self.subwindow_len
        S = self.sub_stride

        # Compute number of subwindows per sequence
        num_subs = (full_time - L) // S + 1

        # Vectorized subwindow extraction
        acc_subs = acc.unfold(dimension=1, size=L, step=S)  # [batch, num_subs, L, 3]
        acc_subs = acc_subs.contiguous().view(-1, L, 3)
        acc_emb = self.acc_encoder(acc_subs)
        acc_emb = acc_emb.view(batch_size, num_subs, -1)

        if self.use_gyro and gyro is not None:
            gyro_subs = gyro.unfold(dimension=1, size=L, step=S)
            gyro_subs = gyro_subs.contiguous().view(-1, L, 3)
            gyro_emb = self.gyro_encoder(gyro_subs)
            gyro_emb = gyro_emb.view(batch_size, num_subs, -1)
            merged_emb = self.merge(acc_emb, gyro_emb)
        else:
            merged_emb = acc_emb

        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sub_embs = torch.cat([cls_token, merged_emb], dim=1)

        # Mask for attention
        mask = None
        if lengths is not None:
            num_valid_subs = ((lengths - L) // S + 1).clamp(min=1)
            mask = torch.arange(sub_embs.size(1), device=acc.device).unsqueeze(0) >= (num_valid_subs + 1).unsqueeze(1)
            mask[:, 0] = False

        attn_out, _ = self.attn(sub_embs, sub_embs, sub_embs, key_padding_mask=mask)
        cls_out = attn_out[:, 0, :]
        final_embedding = self.final_fc(cls_out)
        return final_embedding


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# --------------------------
# Test
# --------------------------
if __name__ == "__main__":
    batch = 2
    full_time = 35
    acc = torch.randn(batch, full_time, 3)
    gyro = torch.randn(batch, full_time, 3)

    model = IMUFullWindowEncoder(subwindow_len=5, embedding_dim=768, sub_stride=1, use_gyro=True)
    out = model(acc, gyro)
    print("ACC+GYRO embedding:", out.shape)

