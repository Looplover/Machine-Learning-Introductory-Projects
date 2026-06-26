import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import ast
import sys
import os
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parse and visualize maze from token sequences
def safe_eval_list(s):
    """Safely parse list string."""
    if isinstance(s, list): return s
    s = str(s).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list): return val
        if isinstance(val, str) and val.strip().startswith('['): return ast.literal_eval(val)
        return val
    except:
        return s.split()

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
        pass 

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


class InferenceDataset(Dataset):
    def __init__(self, df, token2id, max_in):
        self.df = df.reset_index(drop=True)
        self.t2i = token2id
        self.max_in = max_in
        self.inp_col = "input_sequence" if "input_sequence" in df.columns else "input_seq"

    def pad_input(self, ids):
        pad = self.t2i.get('<PAD>', 0)
        if len(ids) > self.max_in:
            ids = ids[:self.max_in]
        elif len(ids) < self.max_in:
            ids = ids + [pad] * (self.max_in - len(ids))
        return ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_seq = self.df.loc[idx, self.inp_col]
        tokens = safe_eval_list(raw_seq)
        ids = [self.t2i.get(t, self.t2i.get('<PAD>', 0)) for t in tokens]
        return torch.tensor(self.pad_input(ids), dtype=torch.long)

# Main evaluation functions
def load_model(model_path, model_type_arg, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    ckpt = torch.load(model_path, map_location=device)
    token2id = ckpt['token2id']
    id2token = ckpt['id2token']
    vocab_size = ckpt['vocab_size']
    pad_id = token2id.get('<PAD>', 0)
    
    max_out_len = ckpt.get('max_out_len', 100) 
    max_in_len = ckpt.get('max_in_len', 1000)

    model = None
    
    if model_type_arg == 'transformer':
        pe_weight = ckpt['model_state'].get('pos_encoder.pe')
        if pe_weight is not None:
            trained_max_len = pe_weight.shape[0]
            max_in_len = trained_max_len
        else:
            trained_max_len = max_in_len
        
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
        emb_dim = ckpt.get('embed_dim', 128)
        hid_dim = ckpt.get('hidden_dim', 512)
        n_layers = ckpt.get('num_layers', 2)
        dropout = ckpt.get('dropout', 0.2)
        
        enc = EncoderRNN(vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_id)
        dec = DecoderRNN(vocab_size, emb_dim, hid_dim, n_layers, dropout, pad_id)
        model = Seq2SeqModel(enc, dec, pad_id)

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    return model, token2id, id2token, pad_id, max_out_len, max_in_len

def predict(model, loader, model_type, device, token2id, id2token, pad_id, max_len):
    results = []
    sos_id = token2id.get('<SOS>', 1)
    eos_id = token2id.get('<EOS>', 2)
    
    print(f"Generating predictions using {model_type}...")
    
    with torch.no_grad():
        for src in tqdm(loader):
            src = src.to(device)
            bs = src.size(0)
            
            finished = torch.zeros(bs, dtype=torch.bool, device=device)
            
            if model_type == 'rnn':
                enc_out, hidden = model.encoder(src)
                mask = (src == pad_id)
                dec_input = torch.full((bs,), sos_id, device=device, dtype=torch.long)
                seqs = [[] for _ in range(bs)]
                
                for _ in range(max_len):
                    logits, hidden, _ = model.decoder(dec_input, hidden, enc_out, mask)
                    next_tokens = logits.argmax(dim=1) 
                    dec_input = next_tokens
                    current_tokens = next_tokens.cpu().tolist()

                    for i, tok in enumerate(current_tokens):
                        if not finished[i]:
                            if tok == eos_id or tok == pad_id:
                                finished[i] = True
                            else:
                                seqs[i].append(tok)
                    if finished.all(): break
                
                for seq in seqs:
                    tokens = [id2token.get(idx, "") for idx in seq]
                    results.append(tokens)

            else:
                decoder_input = torch.full((bs, 1), sos_id, device=device, dtype=torch.long)
                
                for _ in range(max_len):
                    logits = model(src, decoder_input)
                    next_tokens = logits[:, -1, :].argmax(dim=-1)
                    decoder_input = torch.cat([decoder_input, next_tokens.unsqueeze(1)], dim=1)
                    finished |= (next_tokens == eos_id)
                    if finished.all(): break
                
                output_seqs = decoder_input.cpu().tolist()
                for seq in output_seqs:
                    tokens = []
                    for idx in seq[1:]: 
                        if idx == eos_id: break
                        if idx == pad_id: continue 
                        tokens.append(id2token.get(idx, ""))
                    results.append(tokens)
                    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    try:
        model, t2i, i2t, pad_id, max_out_len, max_in_len = load_model(args.model_path, args.model_type.lower(), device)
        print(f"Model loaded. Input Max Len: {max_in_len}, Output Max Len: {max_out_len}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load input
    try:
        df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Error reading data CSV: {e}")
        sys.exit(1)
        
    ds = InferenceDataset(df, t2i, max_in_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    predicted_tokens = predict(model, loader, args.model_type.lower(), device, t2i, i2t, pad_id, max_out_len)

    df['output_path'] = predicted_tokens
    df['output_path'] = df['output_path'].apply(str)

    out_cols = ['id', 'input_sequence', 'maze_type', 'output_path']
    for c in out_cols:
        if c not in df.columns: df[c] = ""
            
    final_df = df[out_cols]
    final_df.to_csv(args.output_path, index=False)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()