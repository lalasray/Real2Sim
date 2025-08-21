import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Subwindow encoder: DeepConv + LSTM with learnable attention
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
        self.lstm = nn.LSTM(input_size=conv_channels[-1], hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(2*lstm_hidden, 1)  # learnable attention
        self.fc = nn.Linear(2*lstm_hidden, embedding_dim)
    
    def forward(self, x):
        # x: batch, time, input_dim
        x = x.transpose(1,2)   # batch, input_dim, time
        x = self.conv(x)       # batch, channels, time
        x = x.transpose(1,2)   # batch, time, channels
        lstm_out, _ = self.lstm(x)
        # learnable attention pooling
        attn_scores = self.attn_fc(lstm_out)  # batch, time, 1
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
        self.acc_weight = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0.5

    def forward(self, acc_emb, gyro_emb):
        w = torch.sigmoid(self.acc_weight)
        return acc_emb * w + gyro_emb * (1-w)  # keep embedding_dim

# --------------------------
# Full window encoder
# --------------------------
class IMUFullWindowEncoder(nn.Module):
    def __init__(self, subwindow_len=10, embedding_dim=768, sub_stride = 1, use_gyro=True):
        super().__init__()
        self.subwindow_len = subwindow_len
        self.use_gyro = use_gyro
        self.sub_stride = sub_stride
        self.acc_encoder = IMUSubwindowEncoder(embedding_dim=embedding_dim)
        if use_gyro:
            self.gyro_encoder = IMUSubwindowEncoder(embedding_dim=embedding_dim)
            self.merge = IMUWeightedMerge(embedding_dim)
        # CLS token for attention over subwindows
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        self.final_fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, acc, gyro=None):
        batch_size, full_time, _ = acc.shape
        sub_embs = []
        for start in range(full_time - self.subwindow_len + self.sub_stride):
            end = start + self.subwindow_len
            acc_win = acc[:, start:end, :]
            acc_emb = self.acc_encoder(acc_win)
            if self.use_gyro and gyro is not None:
                gyro_win = gyro[:, start:end, :]
                gyro_emb = self.gyro_encoder(gyro_win)
                merged = self.merge(acc_emb, gyro_emb)
            else:
                merged = acc_emb
            sub_embs.append(merged.unsqueeze(1))
        sub_embs = torch.cat(sub_embs, dim=1)
        
        # prepend CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        sub_embs = torch.cat([cls_token, sub_embs], dim=1)
        
        attn_out, _ = self.attn(sub_embs, sub_embs, sub_embs)
        cls_out = attn_out[:,0,:]  # CLS token
        final_embedding = self.final_fc(cls_out)
        return final_embedding

class IdentityEncoder(nn.Module):
    def __init__(self):
        super(IdentityEncoder, self).__init__()
        
    def forward(self, x):
        # Just return the input as-is
        return x
# --------------------------
# Test
# --------------------------
if __name__ == "__main__":
    batch = 2
    full_time = 35
    acc = torch.randn(batch, full_time, 3)
    gyro = torch.randn(batch, full_time, 3)

    # ACC + GYRO
    model1 = IMUFullWindowEncoder(subwindow_len=5, embedding_dim=768, sub_stride = 1, use_gyro=True)
    out1 = model1(acc, gyro)
    print("ACC+GYRO embedding:", out1.shape)

    # ACC only
    model2 = IMUFullWindowEncoder(subwindow_len=5, embedding_dim=768, sub_stride = 1, use_gyro=False)
    out2 = model2(acc)
    print("ACC-only embedding:", out2.shape)
