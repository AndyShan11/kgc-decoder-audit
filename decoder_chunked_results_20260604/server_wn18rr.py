"""
Self-contained WN18RR training script for server.
Exactly matches local wn18rr_fix.py that achieved Test MRR=0.470.

No external dependencies on pcden_v2_experiment.py.
Compatible with old PyTorch (torch.cuda.amp).

Usage:
  python server_wn18rr.py --seed 42
  python server_wn18rr.py --seed 123
  python server_wn18rr.py --seed 456
"""
import sys, os, argparse, json, time, random, math, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

# PyTorch compat: new API (torch.amp) vs old API (torch.cuda.amp)
_new_amp = hasattr(torch.amp, 'GradScaler')

def _make_scaler(enabled=True):
    return torch.amp.GradScaler('cuda', enabled=enabled) if _new_amp else torch.cuda.amp.GradScaler(enabled=enabled)

def _autocast(enabled=True):
    return torch.amp.autocast('cuda', enabled=enabled) if _new_amp else torch.cuda.amp.autocast(enabled=enabled)


# ============================================================
# Chunked distance-decoder scoring
# ============================================================

def _score_candidates(model, h_ids, r_ids, ent_full, cand_ent):
    """Score h,r against a candidate-tail chunk.

    Do not call model.score(h, r, cand_ent): h_ids index into the full entity
    matrix, not the candidate chunk.
    """
    h = ent_full[h_ids]

    if model.scorer_type == "TransE":
        r = model.rel_emb(r_ids)
        hr = h + r
        score = -torch.cdist(hr.float(), cand_ent.float(), p=1)
        return model.gamma + score

    if model.scorer_type == "RotatE":
        h_re, h_im = h[:, :model.half], h[:, model.half:]
        t_re, t_im = cand_ent[:, :model.half], cand_ent[:, model.half:]
        phase = model.rel_phase(r_ids)
        cos_p, sin_p = torch.cos(phase), torch.sin(phase)
        rot_re = h_re * cos_p - h_im * sin_p
        rot_im = h_re * sin_p + h_im * cos_p
        diff_re = rot_re.unsqueeze(1) - t_re.unsqueeze(0)
        diff_im = rot_im.unsqueeze(1) - t_im.unsqueeze(0)
        dist = torch.sqrt((diff_re ** 2 + diff_im ** 2).sum(-1) + 1e-9)
        return model.gamma - dist

    raise ValueError(f"Chunked scoring only supports RotatE/TransE, got {model.scorer_type}")


def _chunked_score_no_grad(model, h_ids, r_ids, ent):
    """Return full [batch, num_entities] scores by candidate-tail chunks."""
    chunk = int(getattr(model, "score_chunk_size", 0) or 0)
    if torch.is_grad_enabled() or chunk <= 0 or ent is None or ent.size(0) <= chunk:
        return None
    if getattr(model, "scorer_type", "") not in ("RotatE", "TransE"):
        return None

    outs = []
    for start in range(0, int(ent.size(0)), chunk):
        outs.append(_score_candidates(model, h_ids, r_ids, ent, ent[start:start + chunk]))
    return torch.cat(outs, dim=1)


def _streaming_ce_backward(model, h, r, t, ent, label_smooth, scaler, score_chunk_size):
    """Exact smoothed softmax CE over entity chunks.

    This computes logsumexp and the CE gradient without materialising the full
    [batch, num_entities, dim] distance tensor. It performs backward internally
    and returns a detached scalar loss for logging.
    """
    chunk = int(score_chunk_size or 0)
    if chunk <= 0 or ent.size(0) <= chunk:
        scores = model.score(h, r, ent)
        loss = F.cross_entropy(scores.float(), t, label_smoothing=label_smooth)
        scaler.scale(loss).backward()
        return loss.detach()

    n_ent = int(ent.size(0))
    batch = int(h.size(0))
    dev = h.device
    log_z = None
    sum_scores = torch.zeros(batch, device=dev, dtype=torch.float32)
    target_scores = torch.empty(batch, device=dev, dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, n_ent, chunk):
            end = min(start + chunk, n_ent)
            scores = _score_candidates(model, h, r, ent, ent[start:end]).float()
            part_lse = torch.logsumexp(scores, dim=1)
            log_z = part_lse if log_z is None else torch.logaddexp(log_z, part_lse)
            sum_scores.add_(scores.sum(dim=1))
            mask = (t >= start) & (t < end)
            if mask.any():
                target_scores[mask] = scores[mask, t[mask] - start]

        loss_value = (
            log_z
            - (1.0 - float(label_smooth)) * target_scores
            - (float(label_smooth) / float(n_ent)) * sum_scores
        ).mean()

    scale = 1.0 / float(batch)
    eps_over_n = float(label_smooth) / float(n_ent)
    hard_coeff = 1.0 - float(label_smooth)
    n_chunks = (n_ent + chunk - 1) // chunk

    for chunk_idx, start in enumerate(range(0, n_ent, chunk)):
        end = min(start + chunk, n_ent)
        scores = _score_candidates(model, h, r, ent, ent[start:end]).float()
        with torch.no_grad():
            coeff = torch.exp(scores.detach() - log_z[:, None]) - eps_over_n
            mask = (t >= start) & (t < end)
            if mask.any():
                coeff[mask, t[mask] - start] -= hard_coeff
            coeff.mul_(scale)
        pseudo_loss = (scores * coeff).sum()
        retain = chunk_idx < n_chunks - 1
        scaler.scale(pseudo_loss).backward(retain_graph=retain)

    return loss_value.detach()


# ============================================================
# Data loading
# ============================================================

class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}

        self.train_triples = self._load(os.path.join(data_dir, "train.txt"))
        self.valid_triples = self._load(os.path.join(data_dir, "valid.txt"))
        self.test_triples = self._load(os.path.join(data_dir, "test.txt"))

        self.N = len(self.entity2id)
        num_orig = len(self.relation2id)
        self.num_relations_with_inv = num_orig * 2

        src, tgt, rel = [], [], []
        for h, r, t in self.train_triples:
            src.append(h); tgt.append(t); rel.append(r)
            src.append(t); tgt.append(h); rel.append(r + num_orig)

        self.M = len(src)
        self.edge_index = torch.tensor([src, tgt], dtype=torch.long)
        self.edge_type = torch.tensor(rel, dtype=torch.long)

        self.all_true = defaultdict(set)
        for h, r, t in self.train_triples + self.valid_triples + self.test_triples:
            self.all_true[(h, r)].add(t)

        print(f"  KG: {self.N} ent, {self.num_relations_with_inv} rel, {self.M} edges")
        print(f"  Train/Val/Test: {len(self.train_triples)}/{len(self.valid_triples)}/{len(self.test_triples)}")

    def _load(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3: continue
                h, r, t = parts
                for e in [h, t]:
                    if e not in self.entity2id:
                        idx = len(self.entity2id)
                        self.entity2id[e] = idx
                        self.id2entity[idx] = e
                if r not in self.relation2id:
                    idx = len(self.relation2id)
                    self.relation2id[r] = idx
                    self.id2relation[idx] = r
                triples.append((self.entity2id[h], self.relation2id[r], self.entity2id[t]))
        return triples


# ============================================================
# CompGCN layer (simple, EXACT copy from pcden_v2_experiment.py)
# ============================================================

class CompGCNLayer(nn.Module):
    """CompGCN: msg = W(x_src ⊙ rel_emb). Simple, no gates."""
    def __init__(self, in_dim, out_dim, num_relations, dropout=0.1):
        super().__init__()
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_embed = nn.Embedding(num_relations, in_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.rel_embed.weight)

    def forward(self, x, edge_index, edge_type):
        N = x.size(0)
        src, tgt = edge_index

        x_src = x[src]
        r_emb = self.rel_embed(edge_type)
        composed = x_src * r_emb
        msg = self.W_msg(composed)

        out = torch.zeros(N, msg.size(1), device=x.device, dtype=msg.dtype)
        out.scatter_add_(0, tgt.unsqueeze(1).expand_as(msg), msg)

        deg = torch.zeros(N, device=x.device, dtype=msg.dtype)
        deg.scatter_add_(0, tgt, torch.ones(tgt.size(0), device=x.device, dtype=msg.dtype))
        deg = deg.clamp(min=1)
        out = out / deg.unsqueeze(1)

        out = out + self.W_self(x) + self.bias
        out_f32 = out.float()
        out = self.dropout(F.relu(self.bn(out_f32))).to(x.dtype)
        return out


class CompGCNEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, d_h, num_layers=2, dropout=0.1):
        super().__init__()
        self.init_embed = nn.Embedding(num_entities, d_h)
        nn.init.xavier_uniform_(self.init_embed.weight)

        self.layers = nn.ModuleList([
            CompGCNLayer(d_h, d_h, num_relations, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, edge_index, edge_type):
        x = self.init_embed.weight
        for layer in self.layers:
            x = layer(x, edge_index, edge_type) + x  # residual
        return self.layer_norm(x)


# ============================================================
# Full model: CompGCN encoder + DistMult scoring
# ============================================================

class CompGCNDistMult(nn.Module):
    """CompGCN encoder + pluggable scorer: DistMult / ComplEx / TransE / RotatE."""
    def __init__(self, N, R, d_h=200, gcn_layers=2, gcn_dropout=0.2, scorer='DistMult'):
        super().__init__()
        self.N = N
        self.scorer_type = scorer
        self.graph_encoder = CompGCNEncoder(N, R, d_h, num_layers=gcn_layers,
                                             dropout=gcn_dropout)

        if scorer == 'DistMult':
            self.rel_emb = nn.Embedding(R, d_h)
            nn.init.xavier_uniform_(self.rel_emb.weight)
        elif scorer == 'ComplEx':
            # split d_h into real/imag halves
            assert d_h % 2 == 0
            self.half = d_h // 2
            self.rel_re = nn.Embedding(R, self.half)
            self.rel_im = nn.Embedding(R, self.half)
            nn.init.xavier_uniform_(self.rel_re.weight)
            nn.init.xavier_uniform_(self.rel_im.weight)
        elif scorer == 'TransE':
            self.rel_emb = nn.Embedding(R, d_h)
            nn.init.xavier_uniform_(self.rel_emb.weight)
            self.gamma = 12.0  # margin
        elif scorer == 'RotatE':
            # Entity embeddings split into re/im; relation = rotation phase
            assert d_h % 2 == 0
            self.half = d_h // 2
            self.rel_phase = nn.Embedding(R, self.half)
            nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
            self.gamma = 12.0
        else:
            raise ValueError(f"Unknown scorer: {scorer}")

        self._ei = None
        self._et = None
        self._cached = None

    def set_graph(self, ei, et):
        self._ei = ei.to(next(self.parameters()).device)
        self._et = et.to(next(self.parameters()).device)

    def encode(self):
        self._cached = self.graph_encoder(self._ei, self._et)
        return self._cached

    def score(self, h_ids, r_ids, ent=None):
        chunked = _chunked_score_no_grad(self, h_ids, r_ids, ent)
        if chunked is not None:
            return chunked
        if ent is None:
            ent = self._cached if self._cached is not None else self.encode()
        h = ent[h_ids]

        if self.scorer_type == 'DistMult':
            r = self.rel_emb(r_ids)
            return torch.mm(h * r, ent.t())

        elif self.scorer_type == 'ComplEx':
            # score(h,r,t) = Re(<h, r, conj(t)>)
            h_re, h_im = h[:, :self.half], h[:, self.half:]
            t_re, t_im = ent[:, :self.half], ent[:, self.half:]
            r_re = self.rel_re(r_ids)
            r_im = self.rel_im(r_ids)
            # Compute: h_re*r_re*t_re + h_im*r_re*t_im + h_re*r_im*t_im - h_im*r_im*t_re
            q_re = h_re * r_re - h_im * r_im   # real part of h*r
            q_im = h_re * r_im + h_im * r_re   # imag part of h*r
            return torch.mm(q_re, t_re.t()) + torch.mm(q_im, t_im.t())

        elif self.scorer_type == 'TransE':
            # score = gamma - ||h + r - t||_1 (higher is better)
            r = self.rel_emb(r_ids)
            hr = h + r  # (batch, d)
            # score for all t: -(||hr - t||_1) per (b, N)
            # Use L1: expanding out via broadcast
            # This is memory-heavy; use chunked if needed
            score = -torch.cdist(hr.float(), ent.float(), p=1)
            return self.gamma + score  # gamma constant shift

        elif self.scorer_type == 'RotatE':
            # Rotation in complex plane: t = h ⊙ e^{i*r}
            h_re, h_im = h[:, :self.half], h[:, self.half:]
            t_re, t_im = ent[:, :self.half], ent[:, self.half:]
            phase = self.rel_phase(r_ids)  # (batch, half)
            cos_p, sin_p = torch.cos(phase), torch.sin(phase)
            # Rotate h by phase
            rot_re = h_re * cos_p - h_im * sin_p  # (batch, half)
            rot_im = h_re * sin_p + h_im * cos_p
            # Distance to all t: ||rot - t||
            # Use chunked computation for memory
            b = rot_re.size(0)
            N = ent.size(0)
            diff_re = rot_re.unsqueeze(1) - t_re.unsqueeze(0)  # (b, N, half)
            diff_im = rot_im.unsqueeze(1) - t_im.unsqueeze(0)
            dist = torch.sqrt((diff_re**2 + diff_im**2).sum(-1) + 1e-9)  # (b, N)
            return self.gamma - dist


    def n3_reg(self, h_ids, r_ids, t_ids, ent):
        """Lacroix-2018 N3 nuclear-3-norm regulariser. Penalty on (h, r, t)."""
        h_emb = ent[h_ids]
        t_emb = ent[t_ids]
        if self.scorer_type == "ComplEx":
            half = self.half
            h_re, h_im = h_emb[:, :half], h_emb[:, half:]
            t_re, t_im = t_emb[:, :half], t_emb[:, half:]
            r_re = self.rel_re(r_ids)
            r_im = self.rel_im(r_ids)
            h_n3 = ((h_re**2 + h_im**2 + 1e-12) ** 1.5).sum(dim=1)
            t_n3 = ((t_re**2 + t_im**2 + 1e-12) ** 1.5).sum(dim=1)
            r_n3 = ((r_re**2 + r_im**2 + 1e-12) ** 1.5).sum(dim=1)
            return (h_n3 + r_n3 + t_n3).mean() / 3.0
        elif self.scorer_type == "DistMult":
            r_emb = self.rel_emb(r_ids)
            h_n3 = h_emb.abs().pow(3).sum(dim=1)
            t_n3 = t_emb.abs().pow(3).sum(dim=1)
            r_n3 = r_emb.abs().pow(3).sum(dim=1)
            return (h_n3 + r_n3 + t_n3).mean() / 3.0
        else:
            return torch.tensor(0.0, device=h_emb.device)


# ============================================================
# Training
# ============================================================

@torch.no_grad()
def evaluate(model, graph, triples, batch_size=128):
    model.eval()
    ent = model.encode()
    ranks = []
    for s in range(0, len(triples), batch_size):
        batch = triples[s:s+batch_size]
        h = torch.tensor([x[0] for x in batch], device=device)
        r = torch.tensor([x[1] for x in batch], device=device)
        scores = model.score(h, r, ent)
        for i, (hi, ri, ti) in enumerate(batch):
            st = scores[i, ti].clone()
            for kt in graph.all_true.get((hi, ri), set()):
                if kt != ti:
                    scores[i, kt] = float('-inf')
            ranks.append(max(int((scores[i] >= st).sum()), 1))
    ranks = np.array(ranks, dtype=np.float64)
    return {
        'MRR': float(np.mean(1.0 / ranks)),
        'Hits@1': float(np.mean(ranks <= 1) * 100),
        'Hits@3': float(np.mean(ranks <= 3) * 100),
        'Hits@10': float(np.mean(ranks <= 10) * 100),
        'MR': float(np.mean(ranks)),
    }


def train(model, graph, epochs=300, lr=5e-4, batch_size=4096,
          label_smooth=0.3, val_every=5, val_size=500, ckpt_path=None,
          loss_type="ce", neg_per_pos=50, reg="none", reg_lambda=5e-3,
          score_chunk_size=0):
    scaler = _make_scaler(enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # MultiStepLR: exactly matches wn18rr_fix.py
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs*0.6), int(epochs*0.8)], gamma=0.3)

    model.set_graph(graph.edge_index.to(device), graph.edge_type.to(device))
    triples = graph.train_triples
    max_steps = max(1, len(triples) // batch_size)

    best_mrr = 0.0
    patience = 0
    patience_limit = 30

    print(f"  Training: epochs={epochs}, lr={lr}, batch={batch_size}, ls={label_smooth}")
    print(f"  Steps/epoch: {max_steps}")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        random.shuffle(triples)
        total_loss = 0
        n = 0

        for step in range(max_steps):
            s = step * batch_size
            batch = triples[s:s+batch_size]
            if not batch: break

            optimizer.zero_grad()
            streaming_ce = False
            with _autocast(enabled=use_amp):
                ent = model.encode()
                h = torch.tensor([x[0] for x in batch], device=device)
                r = torch.tensor([x[1] for x in batch], device=device)
                t = torch.tensor([x[2] for x in batch], device=device)
                if (loss_type == "ce"
                        and model.scorer_type in ("RotatE", "TransE")
                        and int(score_chunk_size or 0) > 0):
                    loss = _streaming_ce_backward(
                        model, h, r, t, ent, label_smooth, scaler, score_chunk_size)
                    streaming_ce = True
                else:
                    scores = model.score(h, r, ent)
                    if loss_type == "ce":
                        loss = F.cross_entropy(scores.float(), t, label_smoothing=label_smooth)
                    else:  # logistic with sampled negatives (Trouillon-2016 / Lacroix-2018)
                        B = scores.size(0)
                        pos_scores = scores.gather(1, t.unsqueeze(1)).squeeze(1).float()
                        neg_idx = torch.randint(0, scores.size(1), (B, neg_per_pos), device=scores.device)
                        neg_scores = scores.gather(1, neg_idx).float()
                        pos_loss = F.softplus(-pos_scores)              # [B]
                        neg_loss = F.softplus(neg_scores).sum(dim=1)    # [B]: sum over N negatives per positive
                        loss = (pos_loss + neg_loss).mean()
                if reg == "n3":
                    loss = loss + reg_lambda * model.n3_reg(h, r, t, ent)

            if not streaming_ce:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n += 1

        scheduler.step()
        et = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{epochs} Loss:{total_loss/n:.3f} "
                  f"lr:{optimizer.param_groups[0]['lr']:.2e} {et:.0f}s")

        if val_every > 0 and (epoch + 1) % val_every == 0:
            model.eval()
            with torch.no_grad():
                model._cached = model.encode().detach()
            vr = evaluate(model, graph, graph.valid_triples[:val_size])
            vm = vr['MRR']
            if vm > best_mrr:
                best_mrr = vm
                patience = 0
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"  >> Val MRR:{vm:.4f} H@1:{vr['Hits@1']:.1f}% "
                      f"H@10:{vr['Hits@10']:.1f}% ** BEST **")
            else:
                patience += 1
                if patience <= 3 or patience % 10 == 0:
                    print(f"  >> Val MRR:{vm:.4f} (best:{best_mrr:.4f} p:{patience}/{patience_limit})")
            if patience >= patience_limit:
                print(f"  Early stop at epoch {epoch+1}")
                break

    return best_mrr


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="WN18RR",
                        choices=["WN18RR", "FB15k-237", "YAGO3-10", "Kinship", "UMLS", "codex-m", "codex-s", "codex-l"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_h", type=int, default=200)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--label_smooth", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scorer", type=str, default="DistMult",
                        choices=["DistMult", "ComplEx", "TransE", "RotatE"])
    parser.add_argument("--score_chunk_size", type=int, default=0,
                        help="Entity chunk size for RotatE/TransE 1-vs-all scoring; 0 disables")
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--skip_test", action="store_true",
                        help="skip full test evaluation (for debug)")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "logistic"],
                        help="ce = 1-vs-all softmax CE (current default); logistic = sampled-negative logistic loss (Trouillon-2016)")
    parser.add_argument("--neg_per_pos", type=int, default=50,
                        help="negatives per positive when --loss logistic")
    parser.add_argument("--reg", type=str, default="none", choices=["none", "n3"],
                        help="N3 = Lacroix-2018 nuclear-3-norm regulariser")
    parser.add_argument("--reg_lambda", type=float, default=5e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Device: {device}, AMP: {use_amp}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("=" * 70)
    print(f"CompGCN+DistMult | {args.dataset} | seed={args.seed}")
    print(f"d_h={args.d_h} L={args.layers} lr={args.lr} ls={args.label_smooth} "
          f"epochs={args.epochs}")
    if args.score_chunk_size:
        print(f"score_chunk_size={args.score_chunk_size}")
    print("=" * 70)

    # Data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try local data/ first, then parent dir data/
    for candidate in [os.path.join(script_dir, "data", args.dataset),
                       os.path.join(script_dir, "..", "data", args.dataset),
                       os.path.join(os.getcwd(), "data", args.dataset)]:
        if os.path.isdir(candidate):
            data_dir = candidate
            break
    else:
        raise FileNotFoundError(f"Cannot find data/{args.dataset}/")

    graph = KnowledgeGraph(data_dir)

    model = CompGCNDistMult(graph.N, graph.num_relations_with_inv,
                             d_h=args.d_h, gcn_layers=args.layers,
                             gcn_dropout=args.dropout,
                             scorer=args.scorer).to(device)
    model.score_chunk_size = int(args.score_chunk_size or 0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()

    _stag = f"_{args.scorer}" if args.scorer != "DistMult" else ""
    ckpt = os.path.join(script_dir,
                         f"best_{args.dataset}_d{args.d_h}_L{args.layers}_ls{args.label_smooth}{_stag}_seed{args.seed}.pt")
    t0 = time.time()
    best_val = train(model, graph, epochs=args.epochs, lr=args.lr,
                     batch_size=args.batch_size, label_smooth=args.label_smooth,
                     val_every=args.val_every, val_size=args.val_size,
                     ckpt_path=ckpt,
                     loss_type=args.loss, neg_per_pos=args.neg_per_pos,
                     reg=args.reg, reg_lambda=args.reg_lambda,
                     score_chunk_size=args.score_chunk_size)
    train_time = time.time() - t0
    print(f"  Training done: {train_time:.0f}s, best_val_MRR={best_val:.4f}")

    # Load best and evaluate on full test
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    model.set_graph(graph.edge_index.to(device), graph.edge_type.to(device))
    with torch.no_grad():
        model._cached = model.encode().detach()

    if args.skip_test:
        print("  Skipping full test eval")
        results = {'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0, 'MR': 0}
    else:
        print(f"  Full test evaluation ({len(graph.test_triples)} triples)...")
        results = evaluate(model, graph, graph.test_triples)

    print(f"\n  FINAL TEST: MRR={results['MRR']:.4f} H@1={results['Hits@1']:.1f}% "
          f"H@3={results['Hits@3']:.1f}% H@10={results['Hits@10']:.1f}%")

    # Save
    output = {
        'dataset': args.dataset,
        'seed': args.seed,
        'config': {
            'd_h': args.d_h, 'layers': args.layers, 'lr': args.lr,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'score_chunk_size': args.score_chunk_size,
            'label_smooth': args.label_smooth, 'dropout': args.dropout,
        },
        'params': n_params,
        'train_time_s': train_time,
        'best_val_mrr': best_val,
        'test': results,
    }
    # Filename: include scorer only if non-default, to keep compatibility with existing files
    scorer_tag = f"_{args.scorer}" if args.scorer != "DistMult" else ""
    out_path = os.path.join(
        script_dir,
        f"{args.dataset}_d{args.d_h}_L{args.layers}_ls{args.label_smooth}{scorer_tag}_seed{args.seed}.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")
