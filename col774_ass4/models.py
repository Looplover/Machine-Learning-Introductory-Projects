import numpy as np 
import pandas as pd 
import os
import re
import ast
import math
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

TRAIN_CSV = "/kaggle/input/maze-data/train_6x6_mazes.csv"   

# Parameters for training
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
EMBED_DIM = 128
HIDDEN_DIM = 512
NUM_RNN_LAYERS = 2
DROPOUT = 0.2
INITIAL_TF = 0.5
TF_MIN = 0.1
TF_DECAY = 0.02
GRAD_CLIP = 1.0

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Defining tokens
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"

# Functions for data processing and visualization
def safe_eval_list(s):
    if isinstance(s, list):
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return s.strip().split()

def extract_block(tag, full_text):
    patterns = [
        rf"<\s*{tag}\s*[_\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_\-\s]?\s*END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, full_text, re.S | re.I)
        if m:
            return m.group(1).strip()
    return ""

def parse_coord_pair(s):
    nums = re.findall(r"-?\d+", s)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None

def draw_maze_from_tokens(token_list, cell_size=1.0, rows=6, cols=6):
    text = " ".join(token_list)
    adj = extract_block("ADJLIST", text)
    path_block = extract_block("PATH", text)
    edges = []
    for m in re.finditer(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj):
        pair = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", m.group(0))
        a = parse_coord_pair(pair[0])
        b = parse_coord_pair(pair[1])
        edges.append((a, b))
    vertical = np.ones((rows, cols + 1), dtype=bool)
    horizontal = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2 and abs(c1 - c2) == 1:
            c_between = min(c1, c2) + 1
            vertical[r1, c_between] = False
        elif c1 == c2 and abs(r1 - r2) == 1:
            r_between = min(r1, r2) + 1
            horizontal[r_between, c1] = False

    path_coords = [parse_coord_pair(m) for m in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_block)]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_aspect('equal')

    for r in range(rows+1):
        ax.plot([0, cols], [r, r], color='lightgray', linewidth=1)
    for c in range(cols+1):
        ax.plot([c, c], [0, rows], color='lightgray', linewidth=1)

    for r in range(rows):
        for c in range(cols+1):
            if vertical[r, c]:
                x = c
                y0, y1 = rows - r - 1, rows - r
                ax.plot([x, x], [y0, y1], color='black', linewidth=4, solid_capstyle='butt')
    for r in range(rows+1):
        for c in range(cols):
            if horizontal[r, c]:
                y = r
                x0, x1 = c, c+1
                ax.plot([x0, x1], [rows - y - 1, rows - y - 1], color='black', linewidth=4, solid_capstyle='butt')

    if path_coords:
        xs = [c + 0.5 for (r, c) in path_coords]
        ys = [rows - r - 0.5 for (r, c) in path_coords]
        ax.plot(xs, ys, linestyle='--', linewidth=2, color='red', zorder=3)
        ax.scatter(xs[0], ys[0], marker='o', s=80, color='red', zorder=4)
        ax.scatter(xs[-1], ys[-1], marker='x', s=80, color='red', zorder=4)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Data Preparation
if not os.path.exists(TRAIN_CSV):
    print(f"WARNING: CSV not found at {TRAIN_CSV}. Please ensure data is uploaded.")
else:
    df_all = pd.read_csv(TRAIN_CSV)
    print("Columns:", df_all.columns.tolist())
    inp_col = "input_sequence" if "input_sequence" in df_all.columns else "input_seq"
    out_col = "output_path" if "output_path" in df_all.columns else "output_seq"
    df_all["_inp_tokens"] = df_all[inp_col].apply(safe_eval_list)
    df_all["_out_tokens"] = df_all[out_col].apply(safe_eval_list)
    token_set = set()
    max_in_len = 0
    max_out_len = 0

    for inp, out in zip(df_all["_inp_tokens"], df_all["_out_tokens"]):
        token_set.update(inp)
        token_set.update(out)
        max_in_len = max(max_in_len, len(inp))
        max_out_len = max(max_out_len, len(out))

    vocab_list = [PAD, SOS, EOS] + sorted(list(token_set))
    token2id = {t:i for i,t in enumerate(vocab_list)}
    id2token = {i:t for t,i in token2id.items()}

    PAD_ID = token2id[PAD]
    SOS_ID = token2id[SOS]
    EOS_ID = token2id[EOS]
    VOCAB_SIZE = len(token2id)

    print("Vocab size:", VOCAB_SIZE, "max_in:", max_in_len, "max_out:", max_out_len)
    MAX_OUT_LEN = max_out_len + 2
    MAX_IN_LEN = max_in_len


class MazeDatasetSimple(Dataset):
    def __init__(self, df, inp_col, out_col, token2id, max_in, max_out):
        self.rows = df.reset_index(drop=True)
        self.inp_col = inp_col
        self.out_col = out_col
        self.t2i = token2id
        self.max_in = max_in
        self.max_out = max_out

    def encode(self, tokens, add_sos_eos=False):
        ids = [self.t2i[t] for t in tokens]
        if add_sos_eos:
            ids = [SOS_ID] + ids + [EOS_ID]
        if len(ids) < self.max_out:
            ids = ids + [PAD_ID] * (self.max_out - len(ids))
        else:
            ids = ids[:self.max_out]
        return ids

    def pad_input(self, ids):
        if len(ids) < self.max_in:
            ids = ids + [PAD_ID] * (self.max_in - len(ids))
        else:
            ids = ids[:self.max_in]
        return ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        inp = self.rows.loc[idx, "_inp_tokens"]
        out = self.rows.loc[idx, "_out_tokens"]
        x = torch.tensor(self.pad_input([self.t2i[t] for t in inp]), dtype=torch.long)
        y = torch.tensor(self.encode(out, add_sos_eos=True), dtype=torch.long)
        return x, y

n = len(df_all)
perm = np.random.RandomState(42).permutation(n)
cut = int(0.9 * n)
train_idx, val_idx = perm[:cut], perm[cut:]

train_df = df_all.iloc[train_idx].reset_index(drop=True)
val_df = df_all.iloc[val_idx].reset_index(drop=True)

train_ds = MazeDatasetSimple(train_df, inp_col, out_col, token2id, MAX_IN_LEN, MAX_OUT_LEN)
val_ds = MazeDatasetSimple(val_df, inp_col, out_col, token2id, MAX_IN_LEN, MAX_OUT_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

# RNN classes
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)  
        self.W2 = nn.Linear(hidden_dim, hidden_dim)  
        self.V = nn.Linear(hidden_dim, 1)            

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1]
        enc_proj = self.W1(encoder_outputs)
        dec_proj = self.W2(decoder_hidden).unsqueeze(1)
        score = self.V(torch.tanh(enc_proj + dec_proj))

        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-1), -float('inf'))

        attention_weights = torch.softmax(score, dim=1)
        context = (attention_weights * encoder_outputs).sum(dim=1)
        
        return context, attention_weights.squeeze(-1)

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        emb = self.dropout(self.embed(src))
        outputs, hidden = self.rnn(emb) 
        return outputs, hidden

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.attn = BahdanauAttention(hid_dim)
        self.rnn = nn.RNN(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)

    def forward(self, token, hidden, enc_outputs, mask=None):
        emb = self.dropout(self.embed(token)).unsqueeze(1)
        context, attn_w = self.attn(enc_outputs, hidden, mask) 
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2) 
        out, new_hidden = self.rnn(rnn_in, hidden)
        logits = self.out(out.squeeze(1)) 
        
        return logits, new_hidden, attn_w

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, pad_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(self, src, tgt, use_teacher_forcing=False):
        B, T = tgt.shape
        V = VOCAB_SIZE
        device = src.device
        outputs = torch.zeros(B, T, V, device=device)
        mask = (src == self.pad_id)
        enc_out, hidden = self.encoder(src)
        dec_input = tgt[:, 0]
        
        for t in range(1, T):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out, mask)
            outputs[:, t] = logits

            if use_teacher_forcing:
                dec_input = tgt[:, t]
            else:
                dec_input = logits.argmax(dim=1)
                
        return outputs

# Functions for evaluation metrics
def token_accuracy_batch(logits, targets, pad_id):
    preds = logits.argmax(dim=-1)
    mask = (targets != pad_id)
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0.0

def seq_accuracy_batch(logits, targets, pad_id):
    preds = logits.argmax(dim=-1)
    mask = (targets != pad_id)
    eq = (preds == targets) | (~mask)
    per_seq = eq.all(dim=1)
    return per_seq.sum().item() / targets.size(0)

def token_micro_f1_batch(logits, targets, pad_id):
    preds = logits.argmax(dim=-1).reshape(-1)
    tr = targets.reshape(-1)
    
    valid = tr != pad_id
    if valid.sum().item() == 0:
        return 0.0
        
    p = preds[valid]
    t = tr[valid]
    
    TP = (p == t).sum().item()
    FP = (p != t).sum().item()
    FN = FP
    
    denom = 2*TP + FP + FN
    return (2*TP / denom) if denom > 0 else 0.0

def evaluate_model(model, loader, pad_id, criterion):
    model.eval()
    total_loss = 0.0
    tok_acc_sum = 0.0
    seq_acc_sum = 0.0
    f1_sum = 0.0
    batches = 0

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt, use_teacher_forcing=False)
            logits_flat = outputs[:, 1:].reshape(-1, outputs.size(-1))
            targets_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            logits_seq = outputs[:, 1:] 
            targets_seq = tgt[:, 1:]
            tok_acc = token_accuracy_batch(logits_seq, targets_seq, pad_id)
            seq_acc = seq_accuracy_batch(logits_seq, targets_seq, pad_id)
            f1 = token_micro_f1_batch(logits_seq, targets_seq, pad_id)
            tok_acc_sum += tok_acc
            seq_acc_sum += seq_acc
            f1_sum += f1
            batches += 1

    return (total_loss / max(1, batches),
            tok_acc_sum / max(1, batches),
            seq_acc_sum / max(1, batches),
            f1_sum / max(1, batches))

# RNN Training Loop
def train(model, train_loader, val_loader, epochs, lr, pad_id):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    history = {
        "train_loss": [], "train_tok_acc": [], "train_seq_acc": [],
        "val_loss": [],   "val_tok_acc": [],   "val_seq_acc": []
    }

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        epoch_tok_acc = 0.0
        epoch_seq_acc = 0.0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            use_tf = random.random() < 0.5 

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outputs = model(src, tgt, use_teacher_forcing=use_tf)
                logits_seq = outputs[:, 1:]
                targets_seq = tgt[:, 1:]
                loss = criterion(logits_seq.reshape(-1, VOCAB_SIZE), targets_seq.reshape(-1))
                batch_tok_acc = token_accuracy_batch(logits_seq.detach(), targets_seq, pad_id)
                batch_seq_acc = seq_accuracy_batch(logits_seq.detach(), targets_seq, pad_id)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_tok_acc += batch_tok_acc
            epoch_seq_acc += batch_seq_acc
            total_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_tok_acc:.2f}"})
        val_loss, val_tok_acc, val_seq_acc, val_f1 = evaluate_model(model, val_loader, PAD_ID, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss / total_batches)
        history["train_tok_acc"].append(epoch_tok_acc / total_batches)
        history["train_seq_acc"].append(epoch_seq_acc / total_batches)

        history["val_loss"].append(val_loss)
        history["val_tok_acc"].append(val_tok_acc)
        history["val_seq_acc"].append(val_seq_acc)

        print(f"\nEpoch {epoch} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Token Acc: {history['train_tok_acc'][-1]:.3f} | Val Token Acc: {val_tok_acc:.3f}")
        print(f"Train Seq Acc: {history['train_seq_acc'][-1]:.3f} | Val Seq Acc: {val_seq_acc:.3f}")
        
        if epoch % 10 == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "token2id": token2id,
                "id2token": id2token,
                "max_in_len": MAX_IN_LEN,
                "max_out_len": MAX_OUT_LEN,
                "vocab_size": VOCAB_SIZE
            }
            path = os.path.join(CHECKPOINT_DIR, f"rnn_epoch_{epoch}.pth")
            torch.save(ckpt, path)
            print(f"Saved: {path}")

    return history

print("\nStarting RNN training...")
enc = EncoderRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT)
dec = DecoderRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT)
model = Seq2SeqModel(enc, dec, PAD_ID).to(device)

print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
rnn_history = train(model, train_loader, val_loader, EPOCHS, LR, PAD_ID)
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, rnn_history["train_loss"], label="Train Loss")
plt.plot(epochs_range, rnn_history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("RNN Loss Curve")
plt.legend()
plt.show()

plt.figure(figsize= (8, 6))
plt.plot(epochs_range, rnn_history["val_tok_acc"], label="Val Token Accuracy")
plt.plot(epochs_range, rnn_history["train_tok_acc"], label="Train Token Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Token Accuracy")
plt.title("RNN Token Accuracy Comparison Curve")
plt.legend()
plt.show()

plt.figure(figsize= (8, 6))
plt.plot(epochs_range, rnn_history["val_seq_acc"], label="Val Sequence Accuracy")
plt.plot(epochs_range, rnn_history["train_seq_acc"], label="Train Sequence Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Sequence Accuracy")
plt.title("RNN Sequence Accuracy Comparison Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Function for greedy decoding
def greedy_decode(model, src_seq, max_len=None):
    model.eval()
    if max_len is None:
        max_len = MAX_OUT_LEN

    src = src_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        dummy = torch.full((1, max_len), PAD_ID, dtype=torch.long, device=device)
        dummy[0,0] = SOS_ID
        outputs = model(src, dummy, use_teacher_forcing=False)
        preds = outputs.argmax(dim=-1)[0].cpu().tolist()

    out_tokens = []
    for idx in preds[1:]:
        if idx in (EOS_ID, PAD_ID):
            break
        out_tokens.append(id2token[idx])

    return out_tokens

rng = np.random.RandomState(123)
sample_ids = rng.choice(len(val_ds), size=5, replace=False)

for sid in sample_ids:
    src, tgt = val_ds[sid]
    pred_tokens = greedy_decode(model, src)
    print("\nExample ID:", sid)
    print("Pred tokens:", pred_tokens)

    inp_tokens = val_df.loc[sid, "_inp_tokens"]
    combined = list(inp_tokens)

    injected = False
    for i, tok in enumerate(combined):
        if "PATH" in str(tok) and "START" in str(tok):
            combined = combined[:i+1] + pred_tokens + ["<PATH END>"] + combined[i+1:]
            injected = True
            break

    if not injected:
        combined += ["<PATH START>"] + pred_tokens + ["<PATH END>"]

    draw_maze_from_tokens(combined)


# Transformer classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                 num_decoder_layers, dim_feedforward, dropout, pad_id):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=MAX_IN_LEN+50)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1) 

        src_key_padding_mask = (src == self.pad_id).transpose(0, 1) 
        tgt_key_padding_mask = (tgt == self.pad_id).transpose(0, 1)
        
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)

        src_emb = self.embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer_decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        logits = self.out(output)
        return logits.transpose(0, 1)

# Function for greedy decoding with Transformer
def greedy_decode_transformer(model, src_seq, max_len=None):
    model.eval()
    if max_len is None:
        max_len = MAX_OUT_LEN
        
    src = src_seq.unsqueeze(0).to(device) 
    decoder_input = torch.tensor([[SOS_ID]], dtype=torch.long, device=device) 
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, decoder_input)
            last_token_logits = logits[:, -1, :] 
            next_token = last_token_logits.argmax(dim=-1).unsqueeze(1) 
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            if next_token.item() == EOS_ID:
                break
                
    out_ids = decoder_input.squeeze().tolist()
    if isinstance(out_ids, int): out_ids = [out_ids]
    
    tokens = []
    for idx in out_ids:
        if idx == SOS_ID: continue
        if idx == EOS_ID: break
        tokens.append(id2token.get(idx, ""))
    return tokens

# Transformer training loop
def evaluate_transformer_metrics(model, loader, pad_id, criterion):
    model.eval()
    total_loss = 0
    total_tok_correct = 0
    total_tokens = 0
    total_seq_acc_sum = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            logits = model(src, tgt_inp)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            mask = (tgt_out != pad_id)
            correct_tok = ((preds == tgt_out) & mask).sum().item()
            total_tok_correct += correct_tok
            total_tokens += mask.sum().item()
            
            batch_seq_acc = seq_accuracy_batch(logits, tgt_out, pad_id)
            total_seq_acc_sum += batch_seq_acc
            num_batches += 1
            
    avg_loss = total_loss / max(1, num_batches)
    avg_tok_acc = total_tok_correct / total_tokens if total_tokens > 0 else 0.0
    avg_seq_acc = total_seq_acc_sum / max(1, num_batches)
    
    return avg_loss, avg_tok_acc, avg_seq_acc

def train_transformer_with_history(model, train_loader, val_loader, epochs, lr, pad_id):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    history = {
        "train_loss": [], "train_tok_acc": [], "train_seq_acc": [],
        "val_loss": [],   "val_tok_acc": [],   "val_seq_acc": []
    }

    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        epoch_tok_acc = 0.0
        epoch_seq_acc = 0.0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
                
                batch_tok_acc = token_accuracy_batch(logits.detach(), tgt_output, pad_id)
                batch_seq_acc = seq_accuracy_batch(logits.detach(), tgt_output, pad_id)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_tok_acc += batch_tok_acc
            epoch_seq_acc += batch_seq_acc
            total_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "seq_acc": f"{batch_seq_acc:.2f}"})

        val_loss, val_tok_acc, val_seq_acc = evaluate_transformer_metrics(model, val_loader, pad_id, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(epoch_loss / total_batches)
        history["train_tok_acc"].append(epoch_tok_acc / total_batches)
        history["train_seq_acc"].append(epoch_seq_acc / total_batches)
        
        history["val_loss"].append(val_loss)
        history["val_tok_acc"].append(val_tok_acc)
        history["val_seq_acc"].append(val_seq_acc)

        print(f"\nEpoch {epoch} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Tok Acc: {history['train_tok_acc'][-1]:.3f} | Val Tok Acc: {val_tok_acc:.3f}")
        print(f"Train Seq Acc: {history['train_seq_acc'][-1]:.3f} | Val Seq Acc: {val_seq_acc:.3f}")

        if epoch % 5 == 0 or epoch == epochs:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "token2id": token2id,
                "id2token": id2token,
                "max_in_len": MAX_IN_LEN,
                "max_out_len": MAX_OUT_LEN,
                "vocab_size": VOCAB_SIZE,
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "dropout": DROPOUT
            }
            path = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch}.pth")
            torch.save(ckpt, path)
            print(f"Saved: {path}")

    final_ckpt = {
        "epoch": epochs,
        "model_state": model.state_dict(),
        "token2id": token2id,
        "id2token": id2token,
        "max_in_len": MAX_IN_LEN,
        "max_out_len": MAX_OUT_LEN,
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_layers": NUM_LAYERS,
        "dim_feedforward": DIM_FEEDFORWARD,
        "dropout": DROPOUT
    }
    final_path = os.path.join(CHECKPOINT_DIR, "transformer_final.pth")
    torch.save(final_ckpt, final_path)
    print(f"Training Complete. Saved Final Model to: {final_path}")

    return history

print("\nStarting transformer training...")
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

transformer_model = MazeTransformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_encoder_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    pad_id=PAD_ID
).to(device)

print(f"Transformer Parameters: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad):,}")
transformer_history = train_transformer_with_history(transformer_model, train_loader, val_loader, epochs=EPOCHS, lr=LR, pad_id=PAD_ID)
print("\nGenerating Transformer Visualizations...")
rng = np.random.RandomState(42)
sample_ids = rng.choice(len(val_ds), size=min(5, len(val_ds)), replace=False)

for sid in sample_ids:
    src, tgt = val_ds[sid]
    pred_tokens = greedy_decode_transformer(transformer_model, src)
    print(f"\nExample {sid} | Pred Len: {len(pred_tokens)}")
    
    inp_tokens = list(val_df.loc[sid, "_inp_tokens"])
    combined = list(inp_tokens)
    injected = False
    for i, tok in enumerate(combined):
        if "PATH" in str(tok) and "START" in str(tok):
            combined = combined[:i+1] + pred_tokens + ["<PATH END>"] + combined[i+1:]
            injected = True
            break
    if not injected:
        combined += ["<PATH START>"] + pred_tokens + ["<PATH END>"]
    draw_maze_from_tokens(combined)

# Visualization function for metrics
def plot_metrics(history, model_name="Model"):
    if not history:
        print("No history to plot.")
        return

    n_epochs = len(history["train_loss"])
    epochs_range = range(1, n_epochs + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='.')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='.')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history["train_tok_acc"], label="Train Token Accuracy", marker='.')
    plt.plot(epochs_range, history["val_tok_acc"], label="Val Token Accuracy", marker='.')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Token Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    if "train_seq_acc" in history:
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, history["train_seq_acc"], label="Train Sequence Accuracy", marker='.')
        
        if "val_seq_acc" in history and len(history["val_seq_acc"]) > 0:
            plt.plot(epochs_range, history["val_seq_acc"], label="Val Sequence Accuracy", marker='.')
            
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"{model_name} Sequence Accuracy Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

plot_metrics(transformer_history, "Transformer")