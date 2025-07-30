from typing import List, Tuple

import open_clip
import torch
import torch.nn as nn
from torch import Tensor

from clip.transform import image_transform


class CLIPEmbed(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        if isinstance(clip_model.visual.image_size, int):
            image_size = clip_model.visual.image_size
        else:
            image_size = clip_model.visual.image_size[0]
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with torch.no_grad(), torch.autocast("cuda", enabled=torch.cuda.is_available()):
            x = self.transform(x)  # [batch, 3, 244, 244]
            x = self.clip_model.encode_image(x, normalize=True)  # [batch, 1024]
        return x


class CLIPReward(nn.Module):
    """
    CLIPReward is a PyTorch module for computing reward signals using CLIP-based embeddings for various vision-language model (VLM) reward types.
    Args:
        model (CLIPEmbed): The CLIP embedding module used to encode prompts and images.
        alpha (float): Interpolation parameter for projection computation.
        target_prompts (torch.Tensor): Tokenized or embedded target prompts.
        baseline_prompts (torch.Tensor): Tokenized or embedded baseline prompts.
    Attributes:
        clip_embed_module (CLIPEmbed): The CLIP embedding module.
        targets (torch.Tensor): Embedded target prompts.
        baselines (torch.Tensor): Embedded baseline prompts (if provided).
        direction (torch.Tensor): Direction vector between targets and baselines (if baselines provided).
        alpha (float): Interpolation parameter for projection computation.
        projection (torch.Tensor): Projection matrix computed from direction and alpha (if baselines provided).
    Methods:
        compute_projection(alpha, direction):
            Computes a projection matrix interpolated between the direction outer product and the identity matrix.
        forward(x, vlm_reward_type):
            Computes the reward for the given input tensor `x` according to the specified reward type.
        forward_vlm_rl(x):
            Computes the VLM-RL reward as the difference between the similarity to the first and second target.
        forward_lord(x):
            Computes the LORD reward as one minus the similarity to the targets.
        forward_vlm_rm(x):
            Computes the VLM-RM reward using the projection matrix and the distance to the targets.
        forward_vlm_sr(x):
            Computes the VLM-SR or RoboCLIP reward as the similarity to the targets.
        get_pos_neg(x):
            Returns the positive and negative similarities to the targets.
        tokenize_prompts(x):
            Tokenizes a list of prompt strings using OpenCLIP.
        embed_prompts(x):
            Embeds a list of prompts using the CLIP model.
        embed_images(x):
            Embeds images using the CLIP model.
    """
    def __init__(
            self,
            *,
            model: CLIPEmbed,
            alpha: float,
            target_prompts: torch.Tensor,
            baseline_prompts: torch.Tensor,
    ) -> None:
        super().__init__()
        self.clip_embed_module = model
        targets = self.embed_prompts(target_prompts)
        self.register_buffer("targets", targets)
        if len(baseline_prompts) > 0:
            baselines = self.embed_prompts(baseline_prompts)
            direction = targets - baselines  # [1, 1024]
            # Register them as buffers so they are automatically moved around.
            self.register_buffer("baselines", baselines)
            self.register_buffer("direction", direction)
            self.alpha = alpha
            projection = self.compute_projection(alpha, self.direction)  # [1024, 1024]
            self.register_buffer("projection", projection)

    def compute_projection(self, alpha: float, direction) -> torch.Tensor:
        projection = direction.T @ direction / torch.norm(direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, vlm_reward_type: str) -> torch.Tensor:
        if vlm_reward_type == "VLM-RL":
            return self.forward_vlm_rl(x)
        elif "LORD" in vlm_reward_type:
            return self.forward_lord(x)
        elif vlm_reward_type == "VLM-RM":
            return self.forward_vlm_rm(x)
        elif vlm_reward_type in ["VLM-SR", "RoboCLIP"]:
            return self.forward_vlm_sr(x)
        else:
            raise NotImplementedError

    @torch.inference_mode()
    def forward_vlm_rl(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        z = y[:, 0] - y[:, 1]
        return z

    @torch.inference_mode()
    def forward_lord(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        z = 1 - y
        return z.squeeze()

    @torch.inference_mode()
    def forward_vlm_rm(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.targets) @ self.projection, dim=-1) ** 2) / 2
        return y

    @torch.inference_mode()
    def forward_vlm_sr(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        return y.squeeze()

    @torch.inference_mode()
    def get_pos_neg(self, x: torch.Tensor):
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = (x @ self.targets.T)
        return y[:, 0], y[:, 1]

    @staticmethod
    def tokenize_prompts(x: List[str]) -> torch.Tensor:
        """Tokenize a list of prompts."""
        return open_clip.tokenize(x)

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.clip_embed_module.clip_model.encode_text(x).float()
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.clip_embed_module.forward(x)


def compute_rewards(
        model: CLIPReward,
        frames: torch.Tensor,
        batch_size: int,
        vlm_reward_type: str = "VLM-RL",
) -> Tensor:
    assert frames.device == torch.device("cpu")
    n_samples = len(frames)
    basic_rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i: i + batch_size].to(next(model.parameters()).device)
            with torch.no_grad():
                embeddings = model.clip_embed_module(frames_batch)
                rewards_batch = model(embeddings, vlm_reward_type=vlm_reward_type)
            basic_rewards[i: i + batch_size] = rewards_batch

    return basic_rewards


