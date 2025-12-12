import torch
import torch.nn as nn
import torch.nn.functional as F

# Public constants used by training code
D_INPUT = 64   # dimension of x_t, x_tp1 (semantic embedding)
D_ACT   = 16   # dimension of action / motor / goal field


class ThoughtDecoder(nn.Module):
    """
    GRU decoder that turns Z['sem'] (and optional episode context)
    into a sequence of tokens. Trained with teacher forcing; sampled
    with top-k + repetition penalty.
    """

    def __init__(
        self,
        vocab_size: int,
        d_sem: int,
        d_hidden: int = 256,
        ctx_dim: int | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden
        self.ctx_dim = ctx_dim

        self.embedding = nn.Embedding(vocab_size, d_hidden)
        self.init_proj = nn.Linear(d_sem, d_hidden)
        self.ctx_proj = nn.Linear(ctx_dim, d_hidden) if ctx_dim is not None else None
        self.gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        self.proj = nn.Linear(d_hidden, vocab_size)

    def _init_hidden(
        self,
        z_sem: torch.Tensor,
        ctx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        z_sem: (B, d_sem)
        ctx:   (B, ctx_dim) or None
        """
        h0 = self.init_proj(z_sem)  # (B, H)
        if ctx is not None and self.ctx_proj is not None:
            h0 = h0 + self.ctx_proj(ctx)
        return h0.unsqueeze(0)      # (1, B, H)

    def forward(
        self,
        z_sem: torch.Tensor,
        input_ids: torch.Tensor,
        ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        z_sem:     (B, d_sem)
        input_ids: (B, T_in) tokens used as decoder input (e.g. BOS + tokens[:-1])
        ctx:       (B, ctx_dim) episode context embedding (e.g. enc(situation+action+outcome))
        returns:
          logits:  (B, T_in, vocab_size) for predicting next token at each step.
        """
        h0 = self._init_hidden(z_sem, ctx=ctx)  # (1, B, H)

        emb = self.embedding(input_ids)          # (B, T, H)
        out, _ = self.gru(emb, h0)              # (B, T, H)
        logits = self.proj(out)                 # (B, T, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        z_sem: torch.Tensor,
        ctx: torch.Tensor | None = None,
        max_len: int = 40,
        start_id: int = 2,
        stop_id: int = 3,
        temperature: float = 0.9,
        top_k: int | None = 30,
        rep_penalty: float = 0.3,
    ) -> torch.Tensor:
        """
        Sample a sentence from z_sem using top-k sampling + repetition penalty.

        z_sem: (B, d_sem)
        ctx:   (B, ctx_dim) episode context embedding (optional)
        returns:
          token_ids: (B, max_len) sampled ids (may contain EOS somewhere).
        """
        device = z_sem.device
        B = z_sem.size(0)

        h = self._init_hidden(z_sem, ctx=ctx)   # (1, B, H)
        inputs = torch.full((B, 1), start_id, dtype=torch.long, device=device)
        outputs = torch.zeros(B, max_len, dtype=torch.long, device=device)

        # track which tokens have already appeared, to discourage spam
        used = torch.zeros(B, self.vocab_size, device=device)

        # don't penalize these “special” tokens
        special_ids = {start_id, stop_id}

        for t in range(max_len):
            emb = self.embedding(inputs)         # (B, 1, H)
            out, h = self.gru(emb, h)           # (B, 1, H)
            logits = self.proj(out[:, -1, :])   # (B, V)

            # mild repetition penalty
            if t > 0 and rep_penalty > 0.0:
                logits = logits - rep_penalty * used

            # temperature
            if temperature != 1.0:
                logits = logits / temperature

            # top-k sampling
            if top_k is not None and 0 < top_k < logits.size(-1):
                values, indices = torch.topk(logits, top_k, dim=-1)  # (B, k)
                probs = F.softmax(values, dim=-1)
                idx_in_topk = torch.multinomial(probs, num_samples=1)     # (B, 1)
                next_ids = indices.gather(-1, idx_in_topk).squeeze(-1)    # (B,)
            else:
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            outputs[:, t] = next_ids
            inputs = next_ids.unsqueeze(1)

            # update used tokens
            for b in range(B):
                tok_id = int(next_ids[b].item())
                if tok_id not in special_ids:
                    used[b, tok_id] = 1.0

        return outputs


class GeometricMind(nn.Module):
    """
    Energy-based geometric "mind" with a structured latent state Z.

    Z has components:
      per   : perceptual field     (B, D_PER)
      sem   : semantic field       (B, D_SEM)
      rel   : relational heads     (B, N_REL_HEADS, D_REL)
      act   : action / motor       (B, D_ACT)
      slow  : slow context         (B, D_SLOW)
      fast  : fast fluctuations    (B, D_FAST)
      phi   : phase per head       (B, N_REL_HEADS)
    """

    def __init__(self, vocab_size: int | None = None):
        super().__init__()

        # Latent dimensions
        self.D_PER = D_INPUT       # align with x_t
        self.D_SEM = D_INPUT       # align with x_tp1
        self.N_REL_HEADS = 4
        self.D_REL = 16
        self.D_ACT = D_ACT
        self.D_SLOW = 32
        self.D_FAST = 32
        self.D_PHI = self.N_REL_HEADS

        # For prediction / world model
        self._init_flat_dim()

        # Small projection from semantic -> relational space
        self.sem_to_rel = nn.Linear(self.D_SEM, self.D_REL)

        # World model: predicts semantic "next" embedding from full Z and action/goal
        self.world_mlp = nn.Sequential(
            nn.Linear(self.flat_dim + self.D_ACT, 128),
            nn.ReLU(),
            nn.Linear(128, D_INPUT),
        )

        # Tiny speech head (for quick diagnostic / "mode" labels)
        self.speech_head = nn.Linear(self.D_SEM, 8)
        self.speech_tokens = [
            "YES", "NO", "MAYBE", "SEEK",
            "ALERT", "CALM", "CONFUSED", "STUCK",
        ]

        # Goal-update head: propose revised goal given state + current goal
        # (current goal will be added externally; we use Z here)
        self.goal_update = nn.Linear(self.D_SEM + self.D_SLOW, self.D_ACT)

        # Optional decoder for text; now context-aware (episode embedding)
        if vocab_size is not None:
            self.thought_decoder = ThoughtDecoder(
                vocab_size=vocab_size,
                d_sem=self.D_SEM,
                d_hidden=256,
                ctx_dim=D_INPUT,   # same dim as encoder output
            )
        else:
            self.thought_decoder = None

    # -------------------------------
    # Internal helpers
    # -------------------------------

    def _init_flat_dim(self):
        B = 1
        dummy_Z = {
            "per": torch.zeros(B, self.D_PER),
            "sem": torch.zeros(B, self.D_SEM),
            "rel": torch.zeros(B, self.N_REL_HEADS, self.D_REL),
            "act": torch.zeros(B, self.D_ACT),
            "slow": torch.zeros(B, self.D_SLOW),
            "fast": torch.zeros(B, self.D_FAST),
            "phi": torch.zeros(B, self.N_REL_HEADS),
        }
        flat = self._pack(dummy_Z)
        self.flat_dim = flat.size(1)

    def _pack(self, Z: dict) -> torch.Tensor:
        """
        Flatten Z into (B, D_total).
        """
        per = Z["per"]
        sem = Z["sem"]
        rel = Z["rel"].reshape(Z["rel"].size(0), -1)
        act = Z["act"]
        slow = Z["slow"]
        fast = Z["fast"]
        phi = Z["phi"]
        return torch.cat([per, sem, rel, act, slow, fast, phi], dim=-1)

    def _unpack(self, flat: torch.Tensor) -> dict:
        """
        Unpack a flat tensor of shape (B, D_total) into Z dict.
        """
        B = flat.size(0)
        idx = 0

        per = flat[:, idx: idx + self.D_PER]; idx += self.D_PER
        sem = flat[:, idx: idx + self.D_SEM]; idx += self.D_SEM

        rel_size = self.N_REL_HEADS * self.D_REL
        rel_flat = flat[:, idx: idx + rel_size]; idx += rel_size
        rel = rel_flat.view(B, self.N_REL_HEADS, self.D_REL)

        act = flat[:, idx: idx + self.D_ACT]; idx += self.D_ACT
        slow = flat[:, idx: idx + self.D_SLOW]; idx += self.D_SLOW
        fast = flat[:, idx: idx + self.D_FAST]; idx += self.D_FAST

        phi = flat[:, idx: idx + self.N_REL_HEADS]; idx += self.N_REL_HEADS

        return {
            "per": per,
            "sem": sem,
            "rel": rel,
            "act": act,
            "slow": slow,
            "fast": fast,
            "phi": phi,
        }

    # -------------------------------
    # Public API
    # -------------------------------

    def init_state(self, batch_size: int, device: str | torch.device):
        """
        Initialize Z at t=0.
        """
        Z = {
            "per": torch.zeros(batch_size, self.D_PER, device=device),
            "sem": torch.zeros(batch_size, self.D_SEM, device=device),
            "rel": torch.zeros(batch_size, self.N_REL_HEADS, self.D_REL, device=device),
            "act": torch.zeros(batch_size, self.D_ACT, device=device),
            "slow": torch.zeros(batch_size, self.D_SLOW, device=device),
            "fast": torch.zeros(batch_size, self.D_FAST, device=device),
            "phi": torch.zeros(batch_size, self.N_REL_HEADS, device=device),
        }
        return Z

    def world_predict(self, Z: dict, y_actual: torch.Tensor):
        """
        Predict next semantic embedding x_hat from Z and action/goal y_actual.
        """
        flat = self._pack(Z)
        inp = torch.cat([flat, y_actual], dim=-1)
        x_hat = self.world_mlp(inp)
        return x_hat, None

    def energy(
        self,
        Z: dict,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        y_intended: torch.Tensor
    ) -> torch.Tensor:
        """
        Energy functional E(Z, x_t, x_tp1, y_intended) per batch element.
        Returns tensor of shape (B,).
        """
        per = Z["per"]
        sem = Z["sem"]
        rel = Z["rel"]
        act = Z["act"]
        slow = Z["slow"]
        fast = Z["fast"]
        phi = Z["phi"]

        # Perceptual alignment: per ~ x_t
        E_per = ((per - x_t) ** 2).mean(dim=1)

        # Semantic alignment: sem ~ x_tp1
        E_sem = ((sem - x_tp1) ** 2).mean(dim=1)

        # Intention / goal alignment: act ~ y_intended
        E_act = ((act - y_intended) ** 2).mean(dim=1)

        # Slow-fast coherence
        E_sf = ((slow - fast) ** 2).mean(dim=1)

        # Relational heads align to projected semantics
        sem_rel = self.sem_to_rel(sem)          # (B, D_REL)
        sem_rel_exp = sem_rel.unsqueeze(1)      # (B, 1, D_REL)
        E_rel = ((rel - sem_rel_exp) ** 2).mean(dim=(1, 2))

        # Phase coherence across heads
        phi_centered = phi - phi.mean(dim=1, keepdim=True)
        E_phi = (phi_centered ** 2).mean(dim=1)

        # Small L2 regularizer to keep magnitudes sane
        E_reg = (
            per.pow(2).mean(dim=1)
            + sem.pow(2).mean(dim=1)
            + rel.pow(2).mean(dim=(1, 2))
            + act.pow(2).mean(dim=1)
            + slow.pow(2).mean(dim=1)
            + fast.pow(2).mean(dim=1)
            + phi.pow(2).mean(dim=1)
        )

        # Weights for each term
        w_per = 1.0
        w_sem = 1.0
        w_act = 0.5
        w_sf  = 0.3
        w_rel = 0.3
        w_phi = 0.1
        w_reg = 0.01

        E = (
            w_per * E_per
            + w_sem * E_sem
            + w_act * E_act
            + w_sf  * E_sf
            + w_rel * E_rel
            + w_phi * E_phi
            + w_reg * E_reg
        )
        return E  # (B,)

    def step(
        self,
        Z: dict,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
        y_intended: torch.Tensor,
        mode: str = "real",
        dt: float = 0.05,
        noise_std: float = 0.0
    ):
        """
        One gradient-descent step on the energy landscape.

        mode: "real" or "imagine" (currently only affects noise; your trainer decides what x_t is).
        Returns:
          Z_next: updated state dict
          energy_last: scalar (mean over batch)
        """
        # Pack & require grad for the latent state
        flat = self._pack(Z)
        flat = flat.detach().clone().requires_grad_(True)

        Z_view = self._unpack(flat)
        E_batch = self.energy(Z_view, x_t, x_tp1, y_intended)  # (B,)

        grad_flat, = torch.autograd.grad(E_batch.sum(), flat, create_graph=False)

        # Simple constant mobility G(L) = 1
        flat_next = flat - dt * grad_flat

        if noise_std > 0.0:
            flat_next = flat_next + noise_std * torch.randn_like(flat_next)

        Z_next = self._unpack(flat_next.detach())
        energy_last = E_batch.mean().detach()
        return Z_next, energy_last

    def propose_goal_update(self, Z: dict, y_goal: torch.Tensor) -> torch.Tensor:
        """
        Given current state Z and current goal embedding y_goal,
        propose a revised goal embedding y_goal_new.

        This is trained to match the embedding of revised_goal_label.
        """
        sem = Z["sem"]
        slow = Z["slow"]
        h = torch.cat([sem, slow], dim=-1)  # (B, D_SEM + D_SLOW)
        delta = self.goal_update(h)         # (B, D_ACT)
        return y_goal + delta

    @torch.no_grad()
    def read_speech(self, Z: dict):
        """
        Tiny diagnostic head: maps semantic state to one of 8 labels.
        Not used in training, just for peeking at the mind.
        """
        sem = Z["sem"]
        logits = self.speech_head(sem)
        idx = logits.argmax(dim=-1)
        tokens = [self.speech_tokens[i] for i in idx.tolist()]
        return idx, tokens


if __name__ == "__main__":
    # Minimal demo so you can run: python geometric_mind.py
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mind = GeometricMind(vocab_size=None).to(device)

    B = 2
    Z = mind.init_state(batch_size=B, device=device)
    x_t = torch.randn(B, D_INPUT, device=device)
    x_tp1 = torch.randn(B, D_INPUT, device=device)
    y_intended = torch.zeros(B, D_ACT, device=device)

    for step in range(10):
        Z, E = mind.step(
            Z,
            x_t=x_t,
            x_tp1=x_tp1,
            y_intended=y_intended,
            mode="real",
            dt=0.05,
            noise_std=0.01,
        )
        _, speech = mind.read_speech(Z)
        print(f"step {step:02d} | E={E.item():.4f} | say: {speech}")
