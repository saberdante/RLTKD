import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm
import math

from ._base import Distiller


class RunningMeanStd:
    def __init__(self, shape, device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def update(self, x):
        device = x.device
        if self.mean.device != device:
            self.to(device)
        batch_mean, batch_var = x.mean(dim=0), x.var(dim=0)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.pow(delta, 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        device = x.device
        if self.mean.device != device:
            self.to(device)
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class RLTemperatureAgent(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, t_min=1.0, t_max=10.0):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.initial_LOG_STD_MAX = 0.0
        self.LOG_STD_MIN = -5.0
        self.current_LOG_STD_MAX = self.initial_LOG_STD_MAX
        self.total_epochs = 1
        self.decay_start_epoch = 0
        self.min_log_std_max_ratio = 0.1
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, 4)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def update_exploration_config(self, total_epochs, decay_start_epoch, min_log_std_max_ratio):
        self.total_epochs = total_epochs
        self.decay_start_epoch = decay_start_epoch
        self.min_log_std_max_ratio = min_log_std_max_ratio
        print(f"[RL Agent Config] Total Epochs: {total_epochs}, Decay Start Epoch: {decay_start_epoch},"
              f" Min LOG_STD_MAX Ratio: {min_log_std_max_ratio}")

    def update_exploration_params(self, current_epoch):
        if self.total_epochs <= 0:
            self.current_LOG_STD_MAX = self.initial_LOG_STD_MAX
            return
        decay_progress = 0.0
        if current_epoch >= self.decay_start_epoch:
            effective_epochs = self.total_epochs - self.decay_start_epoch
            if effective_epochs > 0:
                decay_progress = (current_epoch - self.decay_start_epoch) / effective_epochs
                decay_progress = max(0.0, min(1.0, decay_progress))
        decay_factor = 1.0 - decay_progress * (1.0 - self.min_log_std_max_ratio)
        self.current_LOG_STD_MAX = self.initial_LOG_STD_MAX * decay_factor
        self.current_LOG_STD_MAX = max(self.current_LOG_STD_MAX, self.LOG_STD_MIN + 0.1)

    def _get_dist(self, state):
        feat = self.actor_net(state)
        mean_raw, log_std_raw = self.head(feat).chunk(2, dim=-1)
        mean = torch.sigmoid(mean_raw)
        log_std = torch.tanh(log_std_raw)
        log_std = self.LOG_STD_MIN + 0.5 * (self.current_LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        return Normal(mean, std)

    def forward(self, state):
        dist = self._get_dist(state)
        action = dist.mean
        t_teacher = self.t_min + (self.t_max - self.t_min) * action[:, 0].clamp(0.0, 1.0)
        t_student = self.t_min + (self.t_max - self.t_min) * action[:, 1].clamp(0.0, 1.0)
        return t_teacher.clamp(min=1e-6), t_student.clamp(min=1e-6)

    def sample_action(self, state):
        dist = self._get_dist(state)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action = raw_action.clamp(0.0, 1.0)
        t_teacher = self.t_min + (self.t_max - self.t_min) * action[:, 0]
        t_student = self.t_min + (self.t_max - self.t_min) * action[:, 1]
        value = self.critic_net(state).squeeze(-1)
        return t_teacher.clamp(min=1e-6), t_student.clamp(min=1e-6), log_prob, value, action

    def evaluate_actions(self, state, action):
        dist = self._get_dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_net(state).squeeze(-1)
        return log_prob, entropy, value


def kd_loss(logits_student, logits_teacher, temperature_student, temperature_teacher):
    t_s = temperature_student.view(-1, 1).clamp(min=1e-6)
    t_t = temperature_teacher.view(-1, 1).clamp(min=1e-6)
    pred_teacher = F.softmax(logits_teacher / t_t, dim=1)
    log_pred_student = F.log_softmax(logits_student / t_s, dim=1)
    kl_div_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    scaled_loss = kl_div_loss * (t_s.squeeze()**2 + t_t.squeeze()**2) / 2.0
    return scaled_loss.mean()


class KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.initial_device = next(student.parameters()).device
        self.total_epochs = getattr(cfg.SOLVER, "EPOCHS", 240)
        self.delay_epochs_ratio = cfg.KD.DELAY_EPOCHS_RATIO
        self.top_k = cfg.KD.RL_STATE_TOP_K
        state_dim = self.top_k * 2 + 8 + 7
        print(f"[RL Agent] Using optimized state representation with Top-K={self.top_k}. State dimension is {state_dim}.")
        self.agent = RLTemperatureAgent(state_dim)
        self.state_norm = RunningMeanStd(shape=(state_dim,), device=self.initial_device)
        decay_start_epoch = getattr(cfg.KD, "RL_STD_DECAY_START_EPOCH", 0)
        min_log_std_max_ratio = getattr(cfg.KD, "RL_STD_DECAY_MIN_RATIO", 0.1)
        self.agent.update_exploration_config(self.total_epochs, decay_start_epoch, min_log_std_max_ratio)
        self.curriculum_start_epoch = getattr(cfg.KD, "CURRICULUM_START_EPOCH", 0)
        self.curriculum_end_epoch = getattr(cfg.KD, "CURRICULUM_END_EPOCH", int(self.total_epochs * self.delay_epochs_ratio))
        self.curriculum_start_weight = getattr(cfg.KD, "CURRICULUM_START_WEIGHT", 0.1)
        self.kd_weight_factor = self.curriculum_start_weight
        print(f"[Curriculum KD] Enabled. Weight will increase from {self.curriculum_start_weight:.2f} to 1.0 "
              f"between epoch {self.curriculum_start_epoch} and {self.curriculum_end_epoch}.")
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.beta = getattr(cfg.KD.LOSS, "FEAT_WEIGHT", 0.1)
        self.rl_buffer = []
        self.optim_agent = optim.Adam(self.agent.parameters(), lr=getattr(cfg.KD, "RL_LR", 0.0001))
        self.ppo_eps = getattr(cfg.KD, "PPO_CLIP", 0.2)
        self.ppo_epochs = getattr(cfg.KD, "PPO_EPOCHS", 10)
        self.gamma = getattr(cfg.KD, "RL_GAMMA", 0.99)
        self.lam = getattr(cfg.KD, "GAE_LAMBDA", 0.95)
        self.agent_updated = False
        self.default_T = getattr(cfg.KD, "DEFAULT_T", 4.0)
        self.transition_steps = getattr(cfg.KD, "RL_TEMPERATURE_TRANSITION_STEPS", int(self.total_epochs * self.delay_epochs_ratio))
        self.transition_count = 0

    def update_curriculum_factor(self, current_epoch):
        if current_epoch < self.curriculum_start_epoch:
            self.kd_weight_factor = self.curriculum_start_weight
        elif current_epoch > self.curriculum_end_epoch:
            self.kd_weight_factor = 1.0
        else:
            progress = (current_epoch - self.curriculum_start_epoch) / \
                       (self.curriculum_end_epoch - self.curriculum_start_epoch)
            self.kd_weight_factor = self.curriculum_start_weight + \
                                   (1.0 - self.curriculum_start_weight) * \
                                   (0.5 * (1.0 - math.cos(math.pi * progress)))

    def build_state(self, logits_student, logits_teacher, features_student, features_teacher, target):
        ce_teacher = F.cross_entropy(logits_teacher, target, reduction='none')
        p_teacher = F.softmax(logits_teacher, dim=1)
        entropy_teacher = -(p_teacher * torch.log(p_teacher.clamp(min=1e-12))).sum(dim=1)
        ce_student = F.cross_entropy(logits_student, target, reduction='none')
        p_student = F.softmax(logits_student, dim=1)
        entropy_student = -(p_student * torch.log(p_student.clamp(min=1e-12))).sum(dim=1)
        kl = F.kl_div(F.log_softmax(logits_student, dim=1), p_teacher, reduction='none').sum(1)
        fs = features_student['feats'][-1]
        ft = features_teacher['feats'][-1]
        fs_pool = F.adaptive_avg_pool2d(fs, 1).flatten(1)
        ft_pool = F.adaptive_avg_pool2d(ft, 1).flatten(1)
        if fs_pool.shape[1] != ft_pool.shape[1]:
            s_dim, t_dim = fs_pool.shape[1], ft_pool.shape[1]
            if s_dim < t_dim:
                repeats = t_dim // s_dim
                remainder = t_dim % s_dim
                aligned_fs = fs_pool.repeat(1, repeats)
                if remainder > 0:
                    aligned_fs = torch.cat([aligned_fs, fs_pool[:, :remainder]], dim=1)
                fs_pool = aligned_fs
            else:
                repeats = s_dim // t_dim
                remainder = s_dim % t_dim
                aligned_ft = ft_pool.repeat(1, repeats)
                if remainder > 0:
                    aligned_ft = torch.cat([aligned_ft, ft_pool[:, :remainder]], dim=1)
                ft_pool = aligned_ft
        cos_sim = F.cosine_similarity(fs_pool, ft_pool, dim=1)
        mse = ((logits_student - logits_teacher) ** 2).mean(dim=1)
        with torch.no_grad():
            top_k_teacher_logits = torch.topk(logits_teacher, self.top_k, dim=1)[0]
            stats_teacher = torch.cat([
                top_k_teacher_logits.mean(dim=1, keepdim=True), top_k_teacher_logits.std(dim=1, keepdim=True),
                top_k_teacher_logits.max(dim=1, keepdim=True)[0], top_k_teacher_logits.min(dim=1, keepdim=True)[0],
            ], dim=1)
            top_k_student_logits = torch.topk(logits_student, self.top_k, dim=1)[0]
            stats_student = torch.cat([
                top_k_student_logits.mean(dim=1, keepdim=True), top_k_student_logits.std(dim=1, keepdim=True),
                top_k_student_logits.max(dim=1, keepdim=True)[0], top_k_student_logits.min(dim=1, keepdim=True)[0],
            ], dim=1)
        state = torch.cat([
            top_k_teacher_logits, stats_teacher, ce_teacher.unsqueeze(1), entropy_teacher.unsqueeze(1),
            top_k_student_logits, stats_student, ce_student.unsqueeze(1), entropy_student.unsqueeze(1),
            kl.unsqueeze(1), cos_sim.unsqueeze(1), mse.unsqueeze(1),
        ], dim=1)
        return state.detach()

    def compute_reward(self, logits_student, logits_teacher, target,
                    t_student, t_teacher, features_student, features_teacher):
        with torch.no_grad():
            device = logits_teacher.device
            t_teacher, t_student = t_teacher.to(device), t_student.to(device)
            ce_term = F.cross_entropy(logits_student, target, reduction='none')
            t_s, t_t = t_student.view(-1, 1).clamp(min=1e-6), t_teacher.view(-1, 1).clamp(min=1e-6)
            pred_teacher = F.softmax(logits_teacher / t_t, dim=1)
            log_pred_student = F.log_softmax(logits_student / t_s, dim=1)
            kl_div_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
            scaled_kl_term = kl_div_loss * (t_s.squeeze()**2 + t_t.squeeze()**2) / 2.0
            feat_s, feat_t = features_student['feats'][-1], features_teacher['feats'][-1]
            if feat_s.shape[1] != feat_t.shape[1]:
                s_C, t_C = feat_s.shape[1], feat_t.shape[1]
                if s_C < t_C:
                    repeats = t_C // s_C
                    remainder = t_C % s_C
                    aligned_s = feat_s.repeat(1, repeats, 1, 1)
                    if remainder > 0:
                        aligned_s = torch.cat([aligned_s, feat_s[:, :remainder, :, :]], dim=1)
                    feat_s = aligned_s
                else:
                    repeats = s_C // t_C
                    remainder = s_C % t_C
                    aligned_t = feat_t.repeat(1, repeats, 1, 1)
                    if remainder > 0:
                        aligned_t = torch.cat([aligned_t, feat_t[:, :remainder, :, :]], dim=1)
                    feat_t = aligned_t
            if feat_s.shape[2:] != feat_t.shape[2:]:
                feat_s = F.adaptive_avg_pool2d(feat_s, feat_t.shape[2:])
            feat_loss = ((feat_s - feat_t) ** 2).mean(dim=[1, 2, 3])
            reward = -(self.ce_loss_weight * ce_term + 
                       self.kd_weight_factor * self.kd_loss_weight * scaled_kl_term + 
                       self.beta * feat_loss)
        return reward

    def forward_train(self, image, target, **kwargs):
        device = image.device
        logits_s, feats_s = self.student(image)
        with torch.no_grad():
            logits_t, feats_t = self.teacher(image)
        state_raw = self.build_state(logits_s.detach(), logits_t.detach(), feats_s, feats_t, target)
        self.state_norm.update(state_raw)
        state_normalized = self.state_norm.normalize(state_raw)
        if self.agent_updated:
            t_teacher_rl, t_student_rl, log_prob, value, action = self.agent.sample_action(state_normalized)
            if self.transition_count < self.transition_steps:
                alpha = self.transition_count / float(self.transition_steps)
                t_teacher = ((1 - alpha) * self.default_T + alpha * t_teacher_rl).clamp(min=1e-6)
                t_student = ((1 - alpha) * self.default_T + alpha * t_student_rl).clamp(min=1e-6)
            else:
                t_teacher, t_student = t_teacher_rl, t_student_rl
        else:
            t_teacher = torch.full((logits_s.size(0),), self.default_T, device=device)
            t_student = torch.full((logits_s.size(0),), self.default_T, device=device)
            log_prob = torch.zeros(logits_s.size(0), device=device)
            value = torch.zeros(logits_s.size(0), device=device)
            action = torch.zeros(logits_s.size(0), 2, device=device)
        reward = self.compute_reward(logits_s, logits_t, target, t_student, t_teacher, feats_s, feats_t)
        self.rl_buffer.append({
            "state": state_normalized, "action": action.detach(), "log_prob": log_prob.detach(),
            "value": value.detach(), "reward": reward.detach()
        })
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_s, target)
        loss_kd = self.kd_weight_factor * self.kd_loss_weight * kd_loss(logits_s, logits_t, t_student, t_teacher)
        return logits_s, {
            "loss_ce": loss_ce, "loss_kd": loss_kd,
            "t_teacher_mean": t_teacher.mean(), "t_student_mean": t_student.mean(),
            "kd_weight_factor": self.kd_weight_factor
        }

    def update_agent(self, epoch=None, pbar=None):
        if not self.rl_buffer:
            return None
        start_time = time.time()
        update_device = next(self.parameters()).device
        self.agent.to(update_device)
        if epoch is not None:
            self.agent.update_exploration_params(epoch)
            self.update_curriculum_factor(epoch)
        states = torch.cat([x["state"].to(update_device) for x in self.rl_buffer])
        actions = torch.cat([x["action"].to(update_device) for x in self.rl_buffer])
        old_log_probs = torch.cat([x["log_prob"].to(update_device) for x in self.rl_buffer])
        rewards = torch.cat([x["reward"].to(update_device) for x in self.rl_buffer])
        values = torch.cat([x["value"].to(update_device) for x in self.rl_buffer])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages, returns = [], []
        gae = 0
        next_value = torch.tensor(0.0, device=update_device)
        for step in reversed(range(len(rewards))):
            next_value = values[step + 1] if step < len(rewards) - 1 else torch.tensor(0.0, device=update_device)
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        advantages = torch.stack(advantages)
        returns = torch.stack(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_actor_loss, total_critic_loss, total_entropy = 0.0, 0.0, 0.0
        original_desc, original_bar_format = (pbar.desc, pbar.bar_format) if pbar else (None, None)
        if pbar: pbar.bar_format = '{desc} {postfix}'
        for ppo_step in range(self.ppo_epochs):
            log_probs, entropy, value_pred = self.agent.evaluate_actions(states, actions)
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_eps, 1 + self.ppo_eps) * advantages
            actor_loss, critic_loss, entropy_bonus = -torch.min(surr1, surr2).mean(), F.mse_loss(value_pred, returns), entropy.mean()
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_bonus
            self.optim_agent.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optim_agent.step()
            total_actor_loss, total_critic_loss, total_entropy = total_actor_loss + actor_loss.item(), total_critic_loss + critic_loss.item(), total_entropy + entropy_bonus.item()
            if pbar:
                pbar.set_description(f"[TRAIN][RL] Epoch {epoch or -1} PPO {ppo_step+1}/{self.ppo_epochs}")
                pbar.set_postfix({"actor_loss": f"{actor_loss.item():.4f}", "critic_loss": f"{critic_loss.item():.4f}", "entropy": f"{entropy_bonus.item():.4f}"})
                pbar.refresh()
        if pbar and original_desc:
            pbar.bar_format, pbar.desc = original_bar_format, original_desc
            pbar.set_postfix({})
            pbar.refresh()
        with torch.no_grad():
            t_teacher_m, t_student_m = self.agent(states)
        self.rl_buffer = []
        if not self.agent_updated: self.agent_updated = True
        elif self.transition_count < self.transition_steps: self.transition_count += 1
        avg_actor_loss, avg_critic_loss, avg_entropy = total_actor_loss / self.ppo_epochs, total_critic_loss / self.ppo_epochs, total_entropy / self.ppo_epochs
        log_msg = (
            f"Epoch:{epoch or -1}|RL_Update|Time:{time.time()-start_time:.2f}s|"
            f"Actor_Loss:{avg_actor_loss:.4f}|Critic_Loss:{avg_critic_loss:.4f}|Entropy:{avg_entropy:.4f}|"
            f"T_teacher:{t_teacher_m.mean().item():.4f}|T_student:{t_student_m.mean().item():.4f}|"
            f"LOG_STD_MAX:{self.agent.current_LOG_STD_MAX:.4f}|"
            f"KD_Factor:{self.kd_weight_factor:.4f}"
        )
        return f"\033[33m{log_msg}\033[0m"
