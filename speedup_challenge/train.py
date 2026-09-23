import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


import os
import urllib.request
from tqdm import tqdm
import time
import copy
import math

BATCH_SIZE = 64
AMP = True
COMPILE_OPTIONS = {"realize_acc_reads_threshold": 64, "realize_opcount_threshold": 200}


class TromptCell(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model):
        super().__init__()
        # Embeddings (Figure 3.2)
        self.feature_emb_weight = nn.Parameter(torch.empty(n_columns, d_model))
        self.feature_emb_bias = nn.Parameter(torch.empty(n_columns, d_model))
        self.ln_emb = nn.LayerNorm(d_model)

        # Importance Getter (Figure 3.1)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_prompt = nn.LayerNorm(d_model)
        self.dense_imp = nn.Linear(2 * d_model, d_model)

        self.emb_column = nn.Parameter(torch.empty(n_columns, d_model))
        self.emb_prompt = nn.Parameter(torch.empty(n_prompts, d_model))

        # Modified expansion block (Figure 3.3)
        # Without non-linearities! This is important to make significant speed-ups possible.
        self.dense_expand = nn.Linear(1, n_prompts)

        self.reset_parameters()

    def reset_parameters(self):
        d_rsqrt = self.feature_emb_weight.shape[1] ** -0.5
        nn.init.uniform_(self.feature_emb_weight, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.feature_emb_bias, -d_rsqrt, d_rsqrt)
        nn.init.normal_(self.emb_column, std=0.01)
        nn.init.normal_(self.emb_prompt, std=0.01)


class TromptDownstream(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dense0 = nn.Linear(d_model, 1)
        self.dense1 = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dense_out = nn.Linear(d_model, 1)

    def forward(self, xnew: torch.Tensor) -> torch.Tensor:
        return self.dense_out(self.ln(F.relu(self.dense1(xnew))))


class EmbedPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b, v, M, logit_bias):
        with torch.autocast('cuda', enabled=False):
            h = F.relu(x[None, :, :, None] * W[:, None] + b[:, None])
            var, mu = torch.var_mean(h, dim=-1, correction=0)
            rstd = torch.rsqrt(var + 1e-5)
            s = rstd * ((h * v[:, None, None]).sum(-1) - mu * v.sum(-1)[:, None, None])
            pw = torch.softmax(torch.baddbmm(logit_bias[:, None], s, M.transpose(1, 2)), dim=-1)
            a = pw @ M
            an = ((a * rstd)[..., None] * h).sum(2) - (a * rstd * mu).sum(-1, keepdim=True)
        ctx.save_for_backward(x, W, b, v, M, mu, rstd, s, pw, a)
        return an, pw

    @staticmethod
    def backward(ctx, g_An, g_pw):
        x, W, b, v, M, mu, rstd, s, pw, a = ctx.saved_tensors
        with torch.autocast('cuda', enabled=False):
            h = F.relu(x[None, :, :, None] * W[:, None] + b[:, None])
            g_a = rstd * ((h * g_An[:, :, None]).sum(-1) - mu * g_An.sum(-1, keepdim=True))
            g_pw = g_pw + g_a @ M.transpose(1, 2)
            g_logit = pw * (g_pw - (pw * g_pw).sum(-1, keepdim=True))
            g_s = g_logit @ M
            g_M = torch.cat([pw, g_logit], 1).transpose(1, 2) @ torch.cat([g_a, s], 1)
            g_v = ((g_s * rstd)[..., None] * h).sum((1, 2)) - (g_s * rstd * mu).sum((1, 2))[:, None]
            g_n = a[..., None] * g_An[:, :, None] + g_s[..., None] * v[:, None, None]
            m1 = a * g_An.mean(-1, keepdim=True) + g_s * v.mean(-1)[:, None, None]
            m2 = rstd * rstd * (a * g_a + g_s * s) / W.shape[-1]
            g_z = (rstd[..., None] * (g_n - m1[..., None]) - (h - mu[..., None]) * m2[..., None]) * (h > 0)
            g_Wb = (g_z[:, None] * torch.stack([x, torch.ones_like(x)])[None, :, :, :, None]).sum(2)
        return None, g_Wb[:, 0], g_Wb[:, 1], g_v, g_M, g_logit.sum(1)


class Trompt(nn.Module):
    def __init__(self, n_columns, n_prompts, d_model, n_cycles):
        super().__init__()
        self.tcells = nn.ModuleList([TromptCell(n_columns, n_prompts, d_model) for _ in range(n_cycles)])
        self.tdown = TromptDownstream(d_model)
        self.prompt = nn.Parameter(torch.empty(n_prompts, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.prompt, std=0.01)

    def forward(self, x: torch.Tensor, cells=None) -> torch.Tensor:
        cells = [self.tcells[i] for i in (cells if cells is not None else range(len(self.tcells)))]
        st = lambda get: torch.stack([get(c) for c in cells])
        d = self.prompt.shape[-1]
        emb_prompt = st(lambda c: c.emb_prompt)
        x_prompt = F.layer_norm(emb_prompt, (d,)) * st(lambda c: c.ln_prompt.weight)[:, None] + st(lambda c: c.ln_prompt.bias)[:, None]
        x_prompt = torch.cat([x_prompt, self.prompt.expand_as(emb_prompt)], dim=-1)
        x_prompt = torch.baddbmm(st(lambda c: c.dense_imp.bias)[:, None], x_prompt, st(lambda c: c.dense_imp.weight).transpose(1, 2)) + emb_prompt
        x_column = F.layer_norm(st(lambda c: c.emb_column), (d,)) * st(lambda c: c.ln_col.weight)[:, None] + st(lambda c: c.ln_col.bias)[:, None]
        mask = torch.softmax(x_prompt @ x_column.transpose(1, 2), dim=-1)
        scale = 1 + st(lambda c: c.dense_expand.weight)[..., 0]
        mask_sum = mask.sum(-1)
        bias = st(lambda c: c.dense_expand.bias) * mask_sum
        g, beta = st(lambda c: c.ln_emb.weight), st(lambda c: c.ln_emb.bias)
        w0 = self.tdown.dense0.weight[0]
        logit_bias = (beta @ w0)[:, None] * scale * mask_sum + bias * w0.sum() + self.tdown.dense0.bias
        An, pw = EmbedPool.apply(x, st(lambda c: c.feature_emb_weight), st(lambda c: c.feature_emb_bias), g * w0, mask * scale[..., None], logit_bias)
        pw_sums = pw @ torch.stack([scale * mask_sum, bias], dim=-1)
        xnew = g[:, None] * An + beta[:, None] * pw_sums[..., :1] + pw_sums[..., 1:]
        return self.tdown(xnew).squeeze(-1).t()


class GpuSplit:
    def __init__(self, model, cells, device, X, Y):
        self.model, self.cells, self.device = model.to(device), cells, device
        self.X, self.Y = X.to(device), Y.to(device)
        self.steps = math.ceil(len(X) / BATCH_SIZE)
        self.perm = torch.zeros(self.steps, BATCH_SIZE, dtype=torch.long, device=device)
        self.w = torch.zeros(self.steps, BATCH_SIZE, device=device)
        self.i = torch.zeros((), dtype=torch.long, device=device)

        self.shared = [model.prompt, *model.tdown.parameters()]
        self.params = [p for i in cells for p in model.tcells[i].parameters()] + self.shared
        self.optimizer = torch.optim.AdamW(self.params, lr=3e-4, weight_decay=1e-5, fused=True, capturable=True)

        self.scale = torch.tensor(2.0 ** 16 if AMP else 1.0, device=device)
        self.growth = torch.zeros((), dtype=torch.int32, device=device)
        self.found_inf = torch.zeros((), device=device)
        self.optimizer.found_inf = self.found_inf

        sizes = [p.numel() for p in self.shared]
        self.send = torch.zeros(sum(sizes) + 1, device=device)
        self.recv = torch.zeros(sum(sizes) + 1, device=device)
        self.recv_grads = [t.view_as(p) for t, p in zip(self.recv[:-1].split(sizes), self.shared)]

        self.loss = torch.compile(self.loss_fn, fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
        self.predict = torch.compile(lambda x: self.model(x, self.cells).sum(-1), dynamic=False, options=COMPILE_OPTIONS)
        self.graphs = None

    def new_epoch(self, perm):
        pad = self.steps * BATCH_SIZE - len(perm)
        self.perm.copy_(torch.cat([perm, perm[:pad]]).view(self.steps, BATCH_SIZE))
        self.w.copy_(torch.cat([torch.ones(len(perm)), torch.zeros(pad)]).view(self.steps, BATCH_SIZE))
        self.i.zero_()

    def loss_fn(self, X, Y, perm, w, i):
        idx, w = perm.index_select(0, i.view(1)).view(-1), w.index_select(0, i.view(1)).view(-1)
        with torch.autocast('cuda', dtype=torch.float16, enabled=AMP):
            pred = self.model(X.index_select(0, idx), self.cells).float()
        return ((pred - Y.index_select(0, idx)[:, None]) ** 2 * w[:, None]).sum() / (w.sum() * 6)

    def forward_backward(self):
        self.optimizer.zero_grad()
        loss = self.loss(self.X, self.Y, self.perm, self.w, self.i)
        (loss * self.scale).backward()
        self.found_inf.zero_()
        torch._amp_foreach_non_finite_check_and_unscale_([p.grad for p in self.params], self.found_inf, 1 / self.scale)
        torch.cat([p.grad.flatten() for p in self.shared] + [self.found_inf.view(1)], out=self.send)

    def update(self):
        torch._foreach_add_([p.grad for p in self.shared], self.recv_grads)
        torch.maximum(self.found_inf, self.recv[-1], out=self.found_inf)
        self.optimizer.step()
        if AMP:
            torch._amp_update_scale_(self.scale, self.growth, self.found_inf, 2.0, 0.5, 2000)
        self.i += 1

    def capture(self):
        self.optimizer.zero_grad()
        stream = torch.cuda.Stream(self.device)
        self.graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
        with torch.cuda.device(self.device):
            with torch.cuda.graph(self.graphs[0], stream=stream):
                self.forward_backward()
            with torch.cuda.graph(self.graphs[1], stream=stream):
                self.update()

    def run(self, k):
        self.graphs[k].replay() if self.graphs else [self.forward_backward, self.update][k]()


def train_step(halves):
    for h in halves:
        h.run(0)
    halves[0].recv.copy_(halves[1].send)
    halves[1].recv.copy_(halves[0].send)
    for h in halves:
        h.run(1)


def load_from_url(url, cache_dir='.'):
    filename = os.path.join(cache_dir, url.split('/')[-1])
    if not os.path.exists(filename):
        with tqdm(unit='B', unit_scale=True, desc=filename) as pbar:
            urllib.request.urlretrieve(url, filename, reporthook=lambda _, b, t: pbar.update(b))
    return torch.load(filename, map_location=torch.device('cpu'), weights_only=True)


TRAIN_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/train_dataset.pt"
VAL_DATA = "https://huggingface.co/datasets/puhsu/hw01-data/resolve/main/val_dataset.pt"

if __name__ == "__main__":
    torch.manual_seed(0)

    train_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(TRAIN_DATA)))
    val_dataset = torch.utils.data.TensorDataset(*map(torch.nan_to_num, load_from_url(VAL_DATA)))

    Y_mean = train_dataset.tensors[1].mean()
    Y_std = train_dataset.tensors[1].std()
    train_dataset.tensors = (train_dataset.tensors[0], (train_dataset.tensors[1] - Y_mean) / Y_std)

    model = Trompt(n_columns=train_dataset.tensors[0].shape[1], n_prompts=128, d_model=128, n_cycles=6)
    devices = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() > 1 else ['cuda:0', 'cuda:0']
    halves = [GpuSplit(copy.deepcopy(model), cells, device, *train_dataset.tensors) for cells, device in zip([[0, 1, 2], [3, 4, 5]], devices)]

    val_dl = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=1024)

    EPOCHS = 5

    for e in range(1, EPOCHS + 1):
        perm = torch.randperm(len(train_dataset))
        for h in halves:
            h.new_epoch(perm)
        steps = halves[0].steps

        if e == 1:
            sides = [torch.cuda.Stream(d) for d in devices]
            for s, d in zip(sides, devices):
                s.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(sides[0]), torch.cuda.stream(sides[1]):
                for _ in range(3):
                    train_step(halves)
            for s, d in zip(sides, devices):
                torch.cuda.current_stream(d).wait_stream(s)
            for h in halves:
                h.capture()
            steps -= 3

        for d in devices:
            torch.cuda.synchronize(d)
        start = time.time()
        for _ in tqdm(range(steps), unit=' samples', unit_scale=BATCH_SIZE):
            train_step(halves)
        for d in devices:
            torch.cuda.synchronize(d)
        print(f'{steps * BATCH_SIZE / (time.time() - start):.0f} samples/sec')

        mae = 0
        with torch.no_grad():
            for batch in val_dl:
                x, y = batch
                pred = sum(h.predict(x.to(h.device)).to(devices[0]) for h in halves) / 6  # mean over 6 cells
                mae += (pred * Y_std + Y_mean - y.to(devices[0])).abs().sum().item()

            mae = mae / len(val_dataset)

            print(f'>>> Epoch {e:>02}')
            print(f'Validation MAE = {mae:.5f}')
            print('>>>\n')
