import os
import sys
import torch
from torch.utils.data import DataLoader
from model import IMUModel
from utd_mhad_eval import UTDMHAD_IMUDatasetSimOnly, collate_fn_no_pad
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Setup project path
# ----------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ----------------------------
# Dataset paths and params
# ----------------------------
root_sim = r"C:\Users\DFKILenovo\Desktop\UTD_MHAD"
window_size = 90
stride = 1

dataset = UTDMHAD_IMUDatasetSimOnly(root_sim, subjects=[1,2,3,4,5,6,7],
                                    window_size=window_size, stride=stride)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_no_pad)

# ----------------------------
# Device and model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = r"C:\Users\DFKILenovo\Downloads"
model_path = os.path.join(save_path, "best_model.pth")

model = IMUModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model from {model_path}")

# ----------------------------
# Action sentences
# ----------------------------
ACTIONS = [
    "swipe",                  
    "swipe",                 
    "wave",                              
    "clap",                          
    "throw",                              
    "cross arms",                      
    "shoot",                             
    "draw",                            
    "draw",           
    "draw",  
    "draw",                                
    "bowling",                         
    "boxing",                                 
    "swing",                    
    "swing",            
    "arm curl",                          
    "tennis serve",                                 
    "push",                                
    "knock",                     
    "catch",                   
    "pick up and throw",                 
    "jogging",                             
    "walking",                             
    "sit to stand",                                 
    "stand to sit",                                 
    "lunge",           
    "squat"                 
]

# ----------------------------
# Compute sentence embeddings and move to same device
# ----------------------------
sentence_model = SentenceTransformer('sentence-transformers/gtr-t5-base')
sentence_embeddings = sentence_model.encode(ACTIONS, convert_to_tensor=True, device=device)  # [27, 768]
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)  # normalize once

# ----------------------------
# Predict and collect labels
# ----------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for batch in loader:
        sample = batch[0]
        # Right wrist data
        acc_sim = sample['acc_sim'].unsqueeze(0).to(device)
        gyro_sim = sample['gyro_sim'].unsqueeze(0).to(device)
        lengths = torch.tensor([acc_sim.shape[1]]).to(device)

        # Compute embedding (changed to right_thigh in your snippet)
        wrist_emb = model.encoders["right_thigh"](acc_sim, gyro_sim, lengths)
        wrist_norm = F.normalize(wrist_emb, p=2, dim=1)

        # Cosine similarity
        cos_sim = torch.matmul(wrist_norm, sentence_embeddings.T).squeeze(0)  # [27]
        pred_index = torch.argmax(cos_sim).item() + 1  # 1-indexed

        # Store true and predicted labels
        y_true.append(sample['activity'])
        y_pred.append(pred_index)

# ----------------------------
# Compute Macro F1
# ----------------------------
macro_f1 = f1_score(y_true, y_pred, average='micro')
print(f"Macro F1 score on the dataset: {macro_f1:.4f}")

# ----------------------------
# Compute and plot confusion matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred, labels=np.arange(1, 28))  # classes 1-27

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(1, 28),
            yticklabels=np.arange(1, 28))
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()
