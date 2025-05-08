import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (FNS, REGRESSION_DICT, masked_mean, masked_mean_var, masked_quantile)


class SILog(nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        input_fn: str = "log",
        output_fn: str = "sqrt",
        integrated: float = 0.15,
        dims: list[int] = [-2, -1],
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        
        self.dims = dims
        self.input_fn = FNS[input_fn]
        self.output_fn = FNS[output_fn]
        self.eps: float = eps
        self.integrated = integrated

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
        # si: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        error = self.input_fn(inputs.float()) - self.input_fn(target.float())
        if mask is not None:
            mask = mask.bool()
        mean_error, var_error = masked_mean_var(data=error, mask=mask, dim=self.dims, keepdim=False)
        if var_error.ndim > 1:
            var_error = var_error.mean(dim=-1)

        if self.integrated > 0.0:
            scale_error = mean_error**2
            var_error = var_error#  + self.integrated * scale_error * (1 - si.int())

        out_loss = self.output_fn(var_error)
        return out_loss.mean()

    @classmethod
    def build(cls, config):
        obj = cls(
            weight=config["weight"],
            dims=config["dims"],
            output_fn=config["output_fn"],
            input_fn=config["input_fn"],
            integrated=config.get("integrated", 0.15),
        )
        return obj

def depth_triplet_loss(fake_embed, real_embed, depth_diff_threshold=0.1):
    # Compute depth differences (if depth maps are available)
    depth_diff = torch.abs(fake_depth - real_depth).mean(dim=(1,2))
    # Select hard negatives (where depth differs significantly)
    mask = (depth_diff > depth_diff_threshold)
    # Apply triplet loss only on hard negatives
    triplet_loss = F.triplet_margin_loss(
        anchor=fake_embed[mask],
        positive=real_embed[mask],
        negative=real_embed[~mask],  # Random negatives
        margin=0.5
    )
    return triplet_loss

def contrastive_depth_loss(fake_embed, real_embed):
    # Normalize embeddings
    fake_embed = F.normalize(fake_embed, p=2, dim=1)
    real_embed = F.normalize(real_embed, p=2, dim=1)
    # Pairwise cosine similarity
    sim = torch.matmul(fake_embed, real_embed.t())
    # Contrastive loss (positive pairs should be close)
    loss = torch.mean(1 - sim.diag())  # Pull real/fake pairs together
    return loss

def contrastive_loss(
    emb_view1: torch.Tensor,  # [B, T, D]
    emb_view2: torch.Tensor,  # [B, T, D] (same scene, different view)
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Assumes emb_view1 and emb_view2 are paired (same scene).
    """
    B, T, D = emb_view1.shape
    emb_view1 = F.normalize(emb_view1, p=2, dim=-1)
    emb_view2 = F.normalize(emb_view2, p=2, dim=-1)
    
    # Flatten timesteps and compute similarity
    emb1_flat = emb_view1.reshape(-1, D)  # [B*T, D]
    emb2_flat = emb_view2.reshape(-1, D)  # [B*T, D]
    sim = torch.mm(emb1_flat, emb2_flat.T) / temperature
    
    # Positives are same scene (diagonal blocks)
    labels = torch.arange(B * T, device=emb1_flat.device)
    return F.cross_entropy(sim, labels)


if __name__ == "__main__":
    # easy optimization example to use SILog loss
    import torch
    import torch.nn as nn
    from .utils import masked_mean_var

    # Define a model: forward (B, 1, H, W) to (B, T, D)
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            # light weight model
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, 7, stride=4),
                nn.ReLU(),
                nn.Conv2d(64, 128, 7, stride=4),
                nn.Flatten(-2),
                nn.ReLU(),
                nn.Linear(169, 169),
            )

        def forward(self, x):
            return self.net(x)
        
    todepth = lambda x: x.abs().clip(0, 1).detach()
    totarget = lambda x: todepth(2 * x ** 0.5 - 1)
    
    inputs = todepth(torch.randn(64, 1, 224, 224).cuda())
    target_inputs = totarget(inputs)
    eval_inputs = todepth(torch.randn(64, 1, 224, 224).cuda())
    eval_target = totarget(eval_inputs)
    
    model = Encoder().cuda()
    
    mse = nn.MSELoss()
    # loss_fn = lambda x, y: mse(x, y)
    loss_fn = temporal_hard_contrastive_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(1000):
        depth_diff = torch.abs(inputs - target_inputs)
        mask = depth_diff > depth_diff.median()
        loss = loss_fn(model(inputs), model(target_inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            eval_loss = mse(model(eval_inputs), model(eval_target))
            print(f'{i} | eval_loss: {eval_loss.item()}')
        


