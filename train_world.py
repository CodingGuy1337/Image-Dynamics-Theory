import json
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from geometric_mind import GeometricMind, D_INPUT, D_ACT
from world_text import Vocab, TextEncoder, pad_sequences, save_world_assets

DATA_PATH = "world_mind_dataset_5k.jsonl"
ASSETS_PATH = "world_assets.json"
CKPT_PATH = "mind_world.pt"

# ----------------- Goal label "argument" helpers -----------------


def classify_goal_argument(init_label_idx, revised_label_idx, pred_label_idx) -> str:
    """
    Classify how the predicted goal embedding relates to the annotated
    initial and revised goal labels.
    """
    if init_label_idx is None or revised_label_idx is None:
        return "no_goal"
    if pred_label_idx is None:
        return "ambiguous"

    if pred_label_idx == init_label_idx == revised_label_idx:
        return "aligned_stable"

    if pred_label_idx == init_label_idx and init_label_idx != revised_label_idx:
        return "clings_to_initial"

    if pred_label_idx == revised_label_idx and init_label_idx != revised_label_idx:
        return "updates_to_revised"

    if pred_label_idx not in (init_label_idx, revised_label_idx):
        return "rebels_against_both"

    return "ambiguous"


def describe_goal_argument(init_label, revised_label, pred_label, relation: str) -> str:
    """
    Short natural-language explanation of the relation between predicted goal
    and the initial/revised labels. Used only for logging.
    """
    if relation == "no_goal":
        return "No usable initial/revised goal labels for this step."

    if relation == "aligned_stable":
        return (
            f"Prediction aligns with both initial and revised goal label: '{pred_label}'. "
            f"The model treats the stated goal as stable."
        )

    if relation == "clings_to_initial":
        return (
            f"Prediction matches the initial goal '{init_label}' but not the revised goal '{revised_label}'. "
            f"The model stays loyal to the original framing."
        )

    if relation == "updates_to_revised":
        return (
            f"Prediction matches the revised goal '{revised_label}' rather than the initial goal '{init_label}'. "
            f"The model accepts the post-outcome reinterpretation of priority."
        )

    if relation == "rebels_against_both":
        return (
            f"Prediction '{pred_label}' matches neither initial '{init_label}' nor revised '{revised_label}'. "
            f"The model is effectively proposing a third framing of what the step was really about."
        )

    return "Goal-argument relationship ambiguous for this step."


# keep a short history of previous suggestions for boredom / novelty
LAST_SUGGESTIONS: List[str] = []


def load_world_data(path: str) -> List[Dict]:
    """Permissive JSONL loader (no strict schema enforcement)."""
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    return items


def build_vocab_and_maps(items: List[Dict]) -> Tuple[Vocab, Dict[str, int], Dict[str, int]]:
    """Build vocab over text fields and domain/goal mappings."""
    texts: List[str] = []
    domains = set()
    goal_labels = set()

    for it in items:
        for key in ("situation", "action", "expected_outcome", "actual_outcome", "reflection"):
            val = it.get(key, "")
            if isinstance(val, str) and val.strip():
                texts.append(val)

        d = it.get("domain")
        if isinstance(d, str):
            domains.add(d)

        ig = it.get("initial_goal_label")
        rg = it.get("revised_goal_label")
        if isinstance(ig, str) and ig.strip():
            goal_labels.add(ig)
        if isinstance(rg, str) and rg.strip():
            goal_labels.add(rg)

    vocab = Vocab.build(texts, min_freq=1)
    domain2idx = {d: i for i, d in enumerate(sorted(domains))}
    goal2idx = {g: i for i, g in enumerate(sorted(goal_labels))}
    return vocab, domain2idx, goal2idx


def encode_batch_text(vocab: Vocab, encoder: TextEncoder, texts: List[str], device):
    seqs = [vocab.encode(t) for t in texts]
    token_ids, lengths = pad_sequences(seqs, pad_id=vocab.token_to_id["<pad>"])
    token_ids = token_ids.to(device)
    lengths = lengths.to(device)
    emb = encoder(token_ids, lengths)  # (B, D_INPUT)
    return emb


def prepare_decoder_batch(vocab: Vocab, texts: List[str], device):
    """Prepare (input_ids, target_ids, pad_id) for teacher forcing."""
    bos_id = vocab.token_to_id.get("<bos>", 2)
    eos_id = vocab.token_to_id.get("<eos>", 3)
    pad_id = vocab.token_to_id["<pad>"]

    seqs_in = []
    seqs_tgt = []

    for t in texts:
        ids = vocab.encode(t)
        seq = [bos_id] + ids + [eos_id]
        seqs_in.append(seq[:-1])
        seqs_tgt.append(seq[1:])

    input_ids, _ = pad_sequences(seqs_in, pad_id=pad_id)
    target_ids, _ = pad_sequences(seqs_tgt, pad_id=pad_id)

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    return input_ids, target_ids, pad_id


def compute_boredom(new_suggestion: str, history: List[str], max_history: int = 20) -> float:
    """Very dumb Jaccard overlap boredom score in [0,1]."""
    if not history:
        return 0.0

    new_tokens = set(new_suggestion.lower().split())
    if not new_tokens:
        return 0.0

    best_overlap = 0.0
    for h in history[-max_history:]:
        h_tokens = set(h.lower().split())
        if not h_tokens:
            continue
        inter = len(new_tokens & h_tokens)
        union = len(new_tokens | h_tokens)
        if union == 0:
            continue
        overlap = inter / union
        if overlap > best_overlap:
            best_overlap = overlap
    return best_overlap


def make_supervised_items(items: List[Dict]) -> List[Dict]:
    """Keep only items that have full supervision fields."""
    out = []
    for it in items:
        if not isinstance(it.get("situation"), str):
            continue
        if not isinstance(it.get("actual_outcome"), str):
            continue
        if not isinstance(it.get("initial_goal_label"), str):
            continue
        if not isinstance(it.get("revised_goal_label"), str):
            continue
        if not isinstance(it.get("reflection"), str):
            continue
        out.append(it)
    return out


def make_episode_texts(batch: List[Dict]) -> List[str]:
    """Concatenate situation + action + actual_outcome to form episode context text."""
    texts = []
    for b in batch:
        s = b.get("situation", "")
        a = b.get("action", "")
        o = b.get("actual_outcome", "")
        texts.append(f"{s} {a} {o}".strip())
    return texts


def train_world(
    num_epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    decoder_weight: float = 1.0,
    goal_weight: float = 3.0,
    device: str | None = None,
):
    """
    Two-phase training:

      Phase 1 (unsupervised-ish):
        - Uses the supervised subset, but only trains:
            * world prediction (L_world)
            * decoder (L_dec)
        - Goal loss is turned off (L_goal = 0)

      Phase 2 (supervised):
        - Same subset, but now all three losses are active:
            * L_world
            * L_dec
            * L_goal (upweighted by goal_weight)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] {DATA_PATH} not found. Create it with your transitions and re-run.")
        return

    print(f"Loading data from {DATA_PATH} ...")
    items = load_world_data(DATA_PATH)
    print(f"Loaded {len(items)} raw transitions.")

    train_items = make_supervised_items(items)
    if not train_items:
        print("[ERROR] No usable items with situation / outcome / goal labels.")
        return

    print(f"Using {len(train_items)} items with full supervision.")

    vocab, domain2idx, goal2idx = build_vocab_and_maps(items)
    idx2goal = {v: k for k, v in goal2idx.items()}
    print(f"Vocab size = {vocab.size}, domains = {len(domain2idx)}, goals = {len(goal2idx)}")

    # Save vocab & domain/goal mapping for the live model
    save_world_assets(ASSETS_PATH, vocab, domain2idx)
    print(f"Saved world assets to {ASSETS_PATH}")

    encoder = TextEncoder(vocab_size=vocab.size, d_model=128, d_out=D_INPUT).to(device)
    mind = GeometricMind(vocab_size=vocab.size).to(device)

    params = list(encoder.parameters()) + list(mind.parameters())
    opt = optim.Adam(params, lr=lr)

    # Domain and goal embeddings
    n_domains = len(domain2idx)
    n_goals = len(goal2idx)
    domain_emb = nn.Embedding(n_domains, D_ACT).to(device)
    goal_emb = nn.Embedding(n_goals, D_ACT).to(device)

    def batches():
        idxs = torch.randperm(len(train_items))
        for start in range(0, len(train_items), batch_size):
            end = min(start + batch_size, len(train_items))
            sel = idxs[start:end].tolist()
            if not sel:
                continue
            batch = [train_items[i] for i in sel]
            yield batch

    phase1_epochs = num_epochs // 2
    phase2_epochs = num_epochs - phase1_epochs

    global_epoch = 0

    # -----------------
    # Phase 1: unsupervised-ish (no goal loss)
    # -----------------
    for _ in range(1, phase1_epochs + 1):
        global_epoch += 1
        mind.train()
        encoder.train()
        total_loss = 0.0
        total_world = 0.0
        total_dec = 0.0
        total_goal = 0.0
        n_batches = 0

        for batch in batches():
            n = len(batch)
            situations = [b["situation"] for b in batch]
            outcomes = [b["actual_outcome"] for b in batch]
            reflections = [b["reflection"] for b in batch]

            domains = [b["domain"] for b in batch]
            init_goals = [b["initial_goal_label"] for b in batch]
            revised_goals = [b["revised_goal_label"] for b in batch]

            x_t = encode_batch_text(vocab, encoder, situations, device)   # (B, D_INPUT)
            x_tp1 = encode_batch_text(vocab, encoder, outcomes, device)   # (B, D_INPUT)

            # Episode context embedding
            ep_texts = make_episode_texts(batch)
            x_ep = encode_batch_text(vocab, encoder, ep_texts, device)    # (B, D_INPUT)

            dom_ids = torch.tensor(
                [domain2idx[d] for d in domains],
                dtype=torch.long,
                device=device,
            )
            init_goal_ids = torch.tensor(
                [goal2idx[g] for g in init_goals],
                dtype=torch.long,
                device=device,
            )
            revised_goal_ids = torch.tensor(
                [goal2idx[g] for g in revised_goals],
                dtype=torch.long,
                device=device,
            )

            y_domain = domain_emb(dom_ids)             # (B, D_ACT)
            y_goal_init = goal_emb(init_goal_ids)      # (B, D_ACT)
            y_goal_revised = goal_emb(revised_goal_ids)  # (B, D_ACT)

            y_intended = y_domain + y_goal_init

            Z = mind.init_state(batch_size=n, device=device)
            Z_new, E = mind.step(
                Z,
                x_t=x_t,
                x_tp1=x_tp1,
                y_intended=y_intended,
                dt=0.05,
                noise_std=0.0,
            )

            # world prediction loss
            x_hat, _ = mind.world_predict(Z_new, y_actual=y_intended)
            L_world = ((x_hat - x_tp1) ** 2).mean()

            # reflection decoding conditioned on episode context
            L_dec = torch.tensor(0.0, device=device)
            if mind.thought_decoder is not None and decoder_weight > 0.0:
                dec_in, dec_tgt, pad_id = prepare_decoder_batch(vocab, reflections, device)
                logits = mind.thought_decoder(Z_new["sem"], dec_in, ctx=x_ep)   # (B, T, V)
                B, T, V = logits.shape
                L_dec = F.cross_entropy(
                    logits.view(B * T, V),
                    dec_tgt.view(B * T),
                    ignore_index=pad_id,
                )

            L_goal = torch.tensor(0.0, device=device)  # off in phase 1

            loss = L_world + decoder_weight * L_dec + goal_weight * L_goal

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_world += L_world.item()
            total_dec += L_dec.item()
            total_goal += L_goal.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_world = total_world / max(1, n_batches)
        avg_dec = total_dec / max(1, n_batches)
        avg_goal = total_goal / max(1, n_batches)

        print()
        print("=" * 72)
        print(f"[PHASE 1 / EPOCH {global_epoch:03d}] TRAINING STATS (WINDOW A)")
        print(f"  total_loss   = {avg_loss:.4f}")
        print(f"  L_world      = {avg_world:.6f}")
        print(f"  L_dec        = {avg_dec:.6f}")
        print(f"  L_goal       = {avg_goal:.6f} (off in this phase)")
        print("=" * 72)

        _debug_window(
            epoch=global_epoch,
            phase="PHASE 1",
            train_items=train_items,
            vocab=vocab,
            encoder=encoder,
            mind=mind,
            domain2idx=domain2idx,
            goal2idx=goal2idx,
            idx2goal=idx2goal,
            domain_emb=domain_emb,
            goal_emb=goal_emb,
            device=device,
        )

    # -----------------
    # Phase 2: supervised (with goal loss)
    # -----------------
    for _ in range(1, phase2_epochs + 1):
        global_epoch += 1
        mind.train()
        encoder.train()
        total_loss = 0.0
        total_world = 0.0
        total_dec = 0.0
        total_goal = 0.0
        n_batches = 0

        for batch in batches():
            n = len(batch)
            situations = [b["situation"] for b in batch]
            outcomes = [b["actual_outcome"] for b in batch]
            reflections = [b["reflection"] for b in batch]

            domains = [b["domain"] for b in batch]
            init_goals = [b["initial_goal_label"] for b in batch]
            revised_goals = [b["revised_goal_label"] for b in batch]

            x_t = encode_batch_text(vocab, encoder, situations, device)   # (B, D_INPUT)
            x_tp1 = encode_batch_text(vocab, encoder, outcomes, device)   # (B, D_INPUT)

            ep_texts = make_episode_texts(batch)
            x_ep = encode_batch_text(vocab, encoder, ep_texts, device)    # (B, D_INPUT)

            dom_ids = torch.tensor(
                [domain2idx[d] for d in domains],
                dtype=torch.long,
                device=device,
            )
            init_goal_ids = torch.tensor(
                [goal2idx[g] for g in init_goals],
                dtype=torch.long,
                device=device,
            )
            revised_goal_ids = torch.tensor(
                [goal2idx[g] for g in revised_goals],
                dtype=torch.long,
                device=device,
            )

            y_domain = domain_emb(dom_ids)             # (B, D_ACT)
            y_goal_init = goal_emb(init_goal_ids)      # (B, D_ACT)
            y_goal_revised = goal_emb(revised_goal_ids)  # (B, D_ACT)

            y_intended = y_domain + y_goal_init

            Z = mind.init_state(batch_size=n, device=device)
            Z_new, E = mind.step(
                Z,
                x_t=x_t,
                x_tp1=x_tp1,
                y_intended=y_intended,
                dt=0.05,
                noise_std=0.0,
            )

            x_hat, _ = mind.world_predict(Z_new, y_actual=y_intended)
            L_world = ((x_hat - x_tp1) ** 2).mean()

            L_dec = torch.tensor(0.0, device=device)
            if mind.thought_decoder is not None and decoder_weight > 0.0:
                dec_in, dec_tgt, pad_id = prepare_decoder_batch(vocab, reflections, device)
                logits = mind.thought_decoder(Z_new["sem"], dec_in, ctx=x_ep)
                B, T, V = logits.shape
                L_dec = F.cross_entropy(
                    logits.view(B * T, V),
                    dec_tgt.view(B * T),
                    ignore_index=pad_id,
                )

            y_goal_pred = mind.propose_goal_update(Z_new, y_goal_init)
            L_goal = ((y_goal_pred - y_goal_revised) ** 2).mean()

            loss = L_world + decoder_weight * L_dec + goal_weight * L_goal

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_world += L_world.item()
            total_dec += L_dec.item()
            total_goal += L_goal.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_world = total_world / max(1, n_batches)
        avg_dec = total_dec / max(1, n_batches)
        avg_goal = total_goal / max(1, n_batches)

        print()
        print("=" * 72)
        print(f"[PHASE 2 / EPOCH {global_epoch:03d}] TRAINING STATS (WINDOW A)")
        print(f"  total_loss   = {avg_loss:.4f}")
        print(f"  L_world      = {avg_world:.6f}")
        print(f"  L_dec        = {avg_dec:.6f}")
        print(f"  L_goal       = {avg_goal:.6f}")
        print("=" * 72)

        _debug_window(
            epoch=global_epoch,
            phase="PHASE 2",
            train_items=train_items,
            vocab=vocab,
            encoder=encoder,
            mind=mind,
            domain2idx=domain2idx,
            goal2idx=goal2idx,
            idx2goal=idx2goal,
            domain_emb=domain_emb,
            goal_emb=goal_emb,
            device=device,
        )

    torch.save(
        {
            "mind_state_dict": mind.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "domain2idx": domain2idx,
            "goal2idx": goal2idx,
            "token_to_id": vocab.token_to_id,
        },
        CKPT_PATH,
    )
    print(f"\nSaved world-model checkpoint to {CKPT_PATH}")


def _debug_window(
    epoch: int,
    phase: str,
    train_items: List[Dict],
    vocab: Vocab,
    encoder: TextEncoder,
    mind: GeometricMind,
    domain2idx: Dict[str, int],
    goal2idx: Dict[str, int],
    idx2goal: Dict[int, str],
    domain_emb: nn.Embedding,
    goal_emb: nn.Embedding,
    device: str,
):
    """Single-sample debug printout."""
    mind.eval()
    encoder.eval()

    with torch.no_grad():
        idx = torch.randint(len(train_items), (1,)).item()
        sample = train_items[idx]

        x_t = encode_batch_text(vocab, encoder, [sample["situation"]], device)
        x_tp1 = encode_batch_text(vocab, encoder, [sample["actual_outcome"]], device)

        ep_texts = [
            f"{sample.get('situation','')} {sample.get('action','')} {sample.get('actual_outcome','')}"
        ]
        x_ep = encode_batch_text(vocab, encoder, ep_texts, device)

        dom_id = torch.tensor(
            [domain2idx[sample["domain"]]],
            dtype=torch.long,
            device=device,
        )
        init_goal_id = torch.tensor(
            [goal2idx[sample["initial_goal_label"]]],
            dtype=torch.long,
            device=device,
        )
        revised_goal_id = torch.tensor(
            [goal2idx[sample["revised_goal_label"]]],
            dtype=torch.long,
            device=device,
        )

        y_domain = domain_emb(dom_id)
        y_goal_init = goal_emb(init_goal_id)
        y_goal_revised = goal_emb(revised_goal_id)
        y_intended = y_domain + y_goal_init

    Z = mind.init_state(batch_size=1, device=device)
    with torch.enable_grad():
        Z, E = mind.step(
            Z,
            x_t=x_t,
            x_tp1=x_tp1,
            y_intended=y_intended,
            dt=0.05,
            noise_std=0.0,
        )

    with torch.no_grad():
        _, speech_tokens = mind.read_speech(Z)
        mode = speech_tokens[0]

        y_goal_pred = mind.propose_goal_update(Z, y_goal_init)
        cos = torch.nn.functional.cosine_similarity
        sim_init = cos(y_goal_pred, y_goal_init, dim=-1).item()
        sim_revised = cos(y_goal_pred, y_goal_revised, dim=-1).item()

        all_goal_vecs = goal_emb.weight
        sims_all = torch.nn.functional.cosine_similarity(
            y_goal_pred.expand_as(all_goal_vecs), all_goal_vecs, dim=-1
        )
        pred_goal_idx = int(torch.argmax(sims_all).item()) if sims_all.numel() > 0 else None
        pred_goal_label = idx2goal[pred_goal_idx] if pred_goal_idx is not None else None

        init_idx_dbg = goal2idx.get(sample["initial_goal_label"]) if sample.get("initial_goal_label") is not None else None
        revised_idx_dbg = goal2idx.get(sample["revised_goal_label"]) if sample.get("revised_goal_label") is not None else None

        relation = classify_goal_argument(init_idx_dbg, revised_idx_dbg, pred_goal_idx)
        relation_comment = describe_goal_argument(
            sample.get("initial_goal_label"),
            sample.get("revised_goal_label"),
            pred_goal_label,
            relation,
        )

        suggestion = "<no decoder>"
        boredom = 0.0

        if getattr(mind, "thought_decoder", None) is not None:
            bos_id = vocab.token_to_id.get("<bos>", 2)
            eos_id = vocab.token_to_id.get("<eos>", 3)

            gen_ids = mind.thought_decoder.generate(
                Z["sem"],
                ctx=x_ep,
                max_len=60,
                start_id=bos_id,
                stop_id=eos_id,
                temperature=0.9,
                top_k=30,
                rep_penalty=0.3,
            )
            toks = gen_ids[0].tolist()
            if eos_id in toks:
                eos_pos = toks.index(eos_id)
                toks = toks[:eos_pos]
            suggestion = vocab.decode(toks)

            boredom = compute_boredom(suggestion, LAST_SUGGESTIONS)
            if boredom > 0.6:
                gen_ids = mind.thought_decoder.generate(
                    Z["sem"],
                    ctx=x_ep,
                    max_len=60,
                    start_id=bos_id,
                    stop_id=eos_id,
                    temperature=1.2,
                    top_k=50,
                    rep_penalty=0.5,
                )
                toks = gen_ids[0].tolist()
                if eos_id in toks:
                    eos_pos = toks.index(eos_id)
                    toks = toks[:eos_pos]
                suggestion = vocab.decode(toks)

            LAST_SUGGESTIONS.append(suggestion)
            if len(LAST_SUGGESTIONS) > 50:
                LAST_SUGGESTIONS.pop(0)

    print(f"[{phase} / EPOCH {epoch:03d}] THOUGHT & CONTEXT (WINDOW B)")
    print("-" * 72)
    print(f"domain:             {sample['domain']}")
    print(f"episode_id / step:  {sample['episode_id']} / {sample['step']}")
    print(f"mode (speech head): {mode}")
    print(f"energy E:           {E.item():.4f}")
    print(f"goal_init:          {sample['initial_goal_label']}")
    print(f"goal_revised:       {sample['revised_goal_label']}")
    print(f"pred_goal_label:    {pred_goal_label}")
    print(f"goal_relation:      {relation}")
    print(f"goal_comment:       {relation_comment}")
    print(f"cos(pred, init):    {sim_init:.3f}")
    print(f"cos(pred, revised): {sim_revised:.3f}")
    print(f"boredom≈            {boredom:.2f}")
    print("-" * 72)
    print("SITUATION:")
    print(f"  {sample['situation']}")
    print("ACTION:")
    print(f"  {sample['action']}")
    print("EXPECTED OUTCOME:")
    print(f"  {sample['expected_outcome']}")
    print("ACTUAL OUTCOME:")
    print(f"  {sample['actual_outcome']}")
    print("REFLECTION (TARGET):")
    print(f"  {sample['reflection']}")
    print("GENERATED THOUGHT:")
    print(f"  {suggestion}")
    print("=" * 72)


if __name__ == "__main__":
    train_world()
