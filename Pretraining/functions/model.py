import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import IMUFullWindowEncoder, IdentityEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# IMU Model
# ----------------------------
class IMUModel(nn.Module):
    def __init__(self, embedding_dim=768, subwindow_len=5, sub_stride=1, use_gyro=True):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "left_wrist": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "right_wrist": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "left_thigh": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro),
            "right_thigh": IMUFullWindowEncoder(subwindow_len, embedding_dim, sub_stride, use_gyro)
        })
        self.sentence_encoder = IdentityEncoder()

    def forward(self, imu_batch, sentence_emb):
        sentence_out = self.sentence_encoder(sentence_emb)
        if torch.isnan(sentence_out).any():
            print("NaN detected in sentence_out")
        imu_outs = {}
        for key, enc in self.encoders.items():
            acc, gyro, lengths = imu_batch[key]  # pass lengths, not masks
            imu_out = enc(acc, gyro, lengths)
            if torch.isnan(imu_out).any():
                print(f"NaN detected in {key} encoder output")
            imu_outs[key] = imu_out
        return sentence_out, imu_outs

# ----------------------------
# Helper: pad sequences and create masks
# ----------------------------
def stack_sensor(sensor_dict):
    accel_list = [a.squeeze(0) for a in sensor_dict["accel"]]
    gyro_list = [g.squeeze(0) for g in sensor_dict["gyro"]]

    lengths = [a.shape[0] for a in accel_list]
    max_len = max(lengths)

    accel = torch.stack([F.pad(a, (0, 0, 0, max_len - a.shape[0])) for a in accel_list], dim=0).to(device)
    gyro = torch.stack([F.pad(g, (0, 0, 0, max_len - g.shape[0])) for g in gyro_list], dim=0).to(device)

    lengths = torch.tensor(lengths, device=device)
    mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]

    # NaN check
    if torch.isnan(accel).any():
        print("NaN detected in accel")
    if torch.isnan(gyro).any():
        print("NaN detected in gyro")

    return accel, gyro, mask, lengths

