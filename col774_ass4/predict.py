import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import ast
import re
import os
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Function to evaluate string representations of lists
def safe_eval_list(s):
    if isinstance(s, list): return s
    try: return ast.literal_eval(s)
    except: return s.strip().split()

# RNN classes
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        if decoder_hidden.dim() == 3: decoder_hidden = decoder_hidden[-1]
        enc_proj = self.W1(encoder_outputs)
        dec_proj = self.W2(decoder_hidden).unsqueeze(1)
        score = self.V(torch.tanh(enc_proj + dec_proj))
        if mask is not None:
            score = score.masked_fill(mask.unsqueeze(-1), -float('inf'))
        weights = torch.softmax(score, dim=1)
        context = (weights * encoder_outputs).sum(dim=1)
        return context, weights.squeeze(-1)

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
    def forward(self, src):
        emb = self.dropout(self.embed(src))
        return self.rnn(emb)

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout, pad_id):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attn = BahdanauAttention(hid_dim)
        self.rnn = nn.RNN(emb_dim + hid_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hid_dim, vocab_size)
    def forward(self, token, hidden, enc_outputs, mask=None):
        emb = self.dropout(self.embed(token)).unsqueeze(1)
        context, _ = self.attn(enc_outputs, hidden, mask)
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2)
        out, new_hidden = self.rnn(rnn_in, hidden)
        logits = self.out(out.squeeze(1))
        return logits, new_hidden, None

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder, pad_id):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
    def forward(self, src, tgt, use_teacher_forcing=False):
        B, T = tgt.shape
        V = self.decoder.out.out_features
        outputs = torch.zeros(B, T, V, device=src.device)
        mask = (src == self.pad_id)
        enc_out, hidden = self.encoder(src)
        dec_input = tgt[:, 0]
        for t in range(1, T):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out, mask)
            outputs[:, t] = logits
            if use_teacher_forcing: dec_input = tgt[:, t]
            else: dec_input = logits.argmax(dim=1)
        return outputs

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
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, pad_id, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        src_mask = (src == self.pad_id).transpose(0, 1)
        tgt_pad_mask = (tgt == self.pad_id).transpose(0, 1)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        src_emb = self.pos_encoder(self.embed(src) * math.sqrt(self.d_model))
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_mask)
        tgt_emb = self.pos_encoder(self.embed(tgt) * math.sqrt(self.d_model))
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_pad_mask,
                                          memory_key_padding_mask=src_mask)
        return self.out(output).transpose(0, 1)

# Functions and classes for dataset and metrics
class MazeDatasetSimple(Dataset):
    def __init__(self, df, token2id, max_in, max_out):
        self.rows = df.reset_index(drop=True)
        self.t2i = token2id
        self.max_in = max_in
        self.max_out = max_out
        self.inp_col = "input_sequence" if "input_sequence" in df.columns else "input_seq"
        self.out_col = "output_path" if "output_path" in df.columns else "output_seq"

    def encode(self, tokens, add_sos_eos=False):
        ids = [self.t2i.get(t, self.t2i.get('<PAD>', 0)) for t in tokens]
        if add_sos_eos:
            sos = self.t2i.get('<SOS>', 0)
            eos = self.t2i.get('<EOS>', 0)
            ids = [sos] + ids + [eos]
        if len(ids) < self.max_out:
            pad = self.t2i.get('<PAD>', 0)
            ids = ids + [pad] * (self.max_out - len(ids))
        else:
            ids = ids[:self.max_out]
        return ids

    def pad_input(self, ids):
        pad = self.t2i.get('<PAD>', 0)
        if len(ids) < self.max_in:
            ids = ids + [pad] * (self.max_in - len(ids))
        else:
            ids = ids[:self.max_in]
        return ids

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        inp = safe_eval_list(self.rows.loc[idx, self.inp_col])
        out = safe_eval_list(self.rows.loc[idx, self.out_col])
        inp_ids = [self.t2i.get(t, self.t2i.get('<PAD>', 0)) for t in inp]
        x = torch.tensor(self.pad_input(inp_ids), dtype=torch.long)
        y = torch.tensor(self.encode(out, add_sos_eos=True), dtype=torch.long)
        return x, y

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
    if valid.sum().item() == 0: return 0.0
    p = preds[valid]
    t = tr[valid]
    TP = (p == t).sum().item()
    FP = (p != t).sum().item()
    FN = FP 
    denom = 2*TP + FP + FN
    return (2*TP / denom) if denom > 0 else 0.0

# Batched Evaluation Functions
def evaluate_rnn_batched(model, loader, pad_id, sos_id, eos_id, device):
    model.eval()
    total_tok_acc = 0; total_seq_acc = 0; total_f1 = 0; total_samples = 0
    print("Evaluating RNN...")
    
    with torch.no_grad():
        for src, tgt in tqdm(loader):
            src, tgt = src.to(device), tgt.to(device)
            bs = src.size(0)
            outputs = model(src, tgt, use_teacher_forcing=False)
            outputs_sliced = outputs[:, 1:, :] 
            tgt_sliced = tgt[:, 1:]
            tok_acc = token_accuracy_batch(outputs_sliced, tgt_sliced, pad_id)
            seq_acc = seq_accuracy_batch(outputs_sliced, tgt_sliced, pad_id)
            f1 = token_micro_f1_batch(outputs_sliced, tgt_sliced, pad_id)
            total_tok_acc += tok_acc * bs
            total_seq_acc += seq_acc * bs
            total_f1 += f1 * bs
            total_samples += bs
            
    if total_samples == 0: return 0,0,0
    return total_tok_acc/total_samples, total_seq_acc/total_samples, total_f1/total_samples


def evaluate_transformer_batched(model, loader, pad_id, sos_id, eos_id, device):
    model.eval()
    total_tok_acc = 0; total_seq_acc = 0; total_f1 = 0; total_samples = 0
    print("Evaluating Transformer...")
    
    with torch.no_grad():
        for src, tgt in tqdm(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            bs = src.size(0)
            decoder_input = torch.full((bs, 1), sos_id, device=device, dtype=torch.long)
            max_len = tgt.size(1) + 5
            finished = torch.zeros(bs, dtype=torch.bool, device=device)

            for _ in range(max_len):
                logits = model(src, decoder_input) 
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                finished |= (next_tokens == eos_id)
                if finished.all(): break

            pred_len = decoder_input.size(1)
            tgt_len = tgt.size(1)
            preds = decoder_input

            if pred_len < tgt_len:
                pad_tensor = torch.full((bs, tgt_len - pred_len), pad_id, device=device, dtype=torch.long)
                preds_aligned = torch.cat([preds, pad_tensor], dim=1)
            else:
                preds_aligned = preds[:, :tgt_len]

            V = model.out.out_features
            logits_fake = torch.zeros((bs, tgt_len, V), device=device)
            logits_fake.scatter_(2, preds_aligned.unsqueeze(-1), 1.0)
            tok_acc = token_accuracy_batch(logits_fake, tgt, pad_id)
            seq_acc = seq_accuracy_batch(logits_fake, tgt, pad_id)
            f1 = token_micro_f1_batch(logits_fake, tgt, pad_id)

            total_tok_acc += tok_acc * bs
            total_seq_acc += seq_acc * bs
            total_f1 += f1 * bs
            total_samples += bs
            
    if total_samples == 0: return 0,0,0
    return total_tok_acc/total_samples, total_seq_acc/total_samples, total_f1/total_samples

def main():
    parser = argparse.ArgumentParser(description="Predict and Calculate Metrics")
    parser.add_argument("model_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("data_path", type=str, help="Path to csv dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model_path):
        print("Error: Model path not found.")
        return
    print(f"Loading model: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location=device)
    token2id = ckpt['token2id']
    vocab_size = ckpt['vocab_size']
    pad_id = token2id.get('<PAD>', 0)
    sos_id = token2id.get('<SOS>', 1)
    eos_id = token2id.get('<EOS>', 2)
    model = None
    model_type = "RNN"

    if 'd_model' in ckpt:
        model_type = "Transformer"
        pe_weight = ckpt['model_state'].get('pos_encoder.pe')
        if pe_weight is not None:
            trained_max_len = pe_weight.shape[0]
            print(f"Detected trained max_len from checkpoint: {trained_max_len}")
        else:
            trained_max_len = 1000 
            print(f"Could not detect max_len, using fallback: {trained_max_len}")

        model = MazeTransformer(
            vocab_size=vocab_size,
            d_model=ckpt['d_model'],
            nhead=ckpt['nhead'],
            num_encoder_layers=ckpt['num_layers'],
            num_decoder_layers=ckpt['num_layers'],
            dim_feedforward=ckpt['dim_feedforward'],
            dropout=ckpt['dropout'],
            pad_id=pad_id,
            max_len=trained_max_len 
        )
    else:
        model_type = "RNN"
        emb_dim = ckpt.get('embed_dim', 128)
        hid_dim = ckpt.get('hidden_dim', 512)
        n_layers = ckpt.get('num_layers', 2)
        dropout = ckpt.get('dropout', 0.2)
        
        enc = EncoderRNN(vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_id)
        dec = DecoderRNN(vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_id)
        model = Seq2SeqModel(enc, dec, pad_id)

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Model loaded successfully. Architecture: {model_type}")

    if not os.path.exists(args.data_path):
        print("Error: Data path not found.")
        return
    print(f"Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    max_in = ckpt.get('max_in_len', 100)
    max_out = ckpt.get('max_out_len', 100)
    
    ds = MazeDatasetSimple(df, token2id, max_in, max_out)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    print(f"Data loaded. Samples: {len(ds)}")

    if model_type == "RNN":
        tok_acc, seq_acc, f1 = evaluate_rnn_batched(model, loader, pad_id, sos_id, eos_id, device)
    else:
        tok_acc, seq_acc, f1 = evaluate_transformer_batched(model, loader, pad_id, sos_id, eos_id, device)

    print(f"Results for {model_type.upper()}...")
    print(f"Sequence Accuracy (Exact Match): {seq_acc*100:.2f}%")
    print(f"Token Accuracy:                {tok_acc*100:.2f}%")
    print(f"F1 Score:                        {f1*100:.2f}%")

if __name__ == "__main__":
    main()