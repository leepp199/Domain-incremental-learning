'''
domain_net_v4.py – MCnn14 with anti-forgetting BN stats protection.

Built on domain_net_v2.py. Key changes vs v2:
  1. set_task_bn_train_only(current_task): ensures only current task's BN layers
     are in train mode; all other tasks' BN layers stay in eval mode to prevent
     running_mean/running_var corruption during auxiliary loss forward passes.
  2. save_bn_snapshot(task) / restore_bn_snapshot(task): save and restore
     BN running statistics for a specific task as a safety net.
'''
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ---------------------------------------------------------------------------
# Task-specific SE (Squeeze-and-Excitation) attention
# ---------------------------------------------------------------------------
class TaskSEBlock(nn.Module):
    """Lightweight per-task channel attention."""

    def __init__(self, channels, nb_tasks, reduction=16):
        super(TaskSEBlock, self).__init__()
        mid = max(channels // reduction, 4)
        self.se_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels, bias=False),
                nn.Sigmoid(),
            )
            for _ in range(nb_tasks)
        ])
        for se in self.se_layers:
            nn.init.kaiming_normal_(se[0].weight, nonlinearity='relu')
            nn.init.xavier_normal_(se[2].weight)
            se[2].weight.data *= 0.01

    def forward(self, x, task):
        b, c, _, _ = x.size()
        y = x.mean(dim=[2, 3])  # (B, C)
        y = self.se_layers[task](y).view(b, c, 1, 1)
        return x * (y * 2.0)


# ---------------------------------------------------------------------------
# Task-specific adapter (residual bottleneck)
# ---------------------------------------------------------------------------
class TaskAdapter(nn.Module):
    """Lightweight per-task feature adapter with residual connection."""

    def __init__(self, dim, nb_tasks, bottleneck=256):
        super(TaskAdapter, self).__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, bottleneck),
                nn.ReLU(inplace=True),
                nn.Linear(bottleneck, dim),
            )
            for _ in range(nb_tasks)
        ])
        for adapter in self.adapters:
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)

    def forward(self, x, task):
        return x + self.adapters[task](x)


# ---------------------------------------------------------------------------
# ConvBlock with per-task BN + SE attention
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])

        self.se = TaskSEBlock(out_channels, nb_tasks)

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1, use_se=True):
        x = input
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if use_se:
            x = self.se(x, task)

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


# ---------------------------------------------------------------------------
# MCnn14 v4 – main model with BN stats protection
# ---------------------------------------------------------------------------
class MCnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):

        super(MCnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center,
            pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin,
            top_db=top_db, freeze_parameters=True)

        self.bn0 = nn.ModuleList([nn.BatchNorm2d(64) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64, nb_tasks=nb_tasks)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128, nb_tasks=nb_tasks)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256, nb_tasks=nb_tasks)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512, nb_tasks=nb_tasks)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024, nb_tasks=nb_tasks)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048, nb_tasks=nb_tasks)

        self.adapter = TaskAdapter(dim=2048, nb_tasks=nb_tasks, bottleneck=256)

        self.fc_heads = nn.ModuleList([
            nn.Linear(2048, classes_num) for _ in range(nb_tasks)
        ])

        self.fc_route = nn.Linear(2048, classes_num)

        # Storage for BN running stats snapshots
        self._bn_snapshots = {}

    # -------------------------------------------------------------------
    # V4 NEW: BN stats protection
    # -------------------------------------------------------------------
    def _get_task_bn_layers(self, task):
        """Return all BN layers belonging to a specific task."""
        layers = [self.bn0[task]]
        for block_name in ['conv_block1', 'conv_block2', 'conv_block3',
                           'conv_block4', 'conv_block5', 'conv_block6']:
            block = getattr(self, block_name)
            layers.append(block.bnF[task])
            layers.append(block.bnS[task])
        return layers

    def set_task_bn_train_only(self, current_task):
        """Set ONLY current task's BN layers to train mode; all others to eval.

        This prevents running_mean/running_var corruption when forwarding
        through other tasks' BN paths during auxiliary loss computation.
        Must be called AFTER model.train() at each training epoch.
        """
        nb_tasks = len(self.bn0)
        for task in range(nb_tasks):
            mode_train = (task == current_task)
            for bn in self._get_task_bn_layers(task):
                if mode_train:
                    bn.train()
                else:
                    bn.eval()

    def save_bn_snapshot(self, task):
        """Save a copy of BN running stats for a specific task."""
        snapshot = {}
        for bn in self._get_task_bn_layers(task):
            key = id(bn)
            snapshot[key] = {
                'running_mean': bn.running_mean.clone(),
                'running_var': bn.running_var.clone(),
                'num_batches_tracked': bn.num_batches_tracked.clone(),
            }
        self._bn_snapshots[task] = snapshot

    def restore_bn_snapshot(self, task):
        """Restore BN running stats for a specific task from saved snapshot."""
        snapshot = self._bn_snapshots.get(task)
        if snapshot is None:
            return
        for bn in self._get_task_bn_layers(task):
            key = id(bn)
            if key in snapshot:
                bn.running_mean.copy_(snapshot[key]['running_mean'])
                bn.running_var.copy_(snapshot[key]['running_var'])
                bn.num_batches_tracked.copy_(snapshot[key]['num_batches_tracked'])

    # -------------------------------------------------------------------
    # Standard methods (same as v2)
    # -------------------------------------------------------------------
    def get_output_dim(self):
        return self.fc_heads[0].out_features

    def change_output_dim(self, new_dim, second_iter=False):
        for i, fc in enumerate(self.fc_heads):
            in_features = fc.in_features
            out_features = fc.out_features
            new_fc = nn.Linear(in_features, new_dim)
            new_fc.weight.data[:out_features] = fc.weight.data
            new_fc.bias.data[:out_features] = fc.bias.data
            self.fc_heads[i] = new_fc

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_weight_conv(self):
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.conv_block2.parameters():
            param.requires_grad = False
        for param in self.conv_block3.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        m.bias.data.fill_(0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.fill_(0.)
                m.weight.data.fill_(1.)

    # -------------------------------------------------------------------
    # Checkpoint compatibility: load v1 (original) state dict into v4
    # -------------------------------------------------------------------
    def load_pretrained_state_dict(self, state_dict, strict=False):
        new_state = {}
        for key, value in state_dict.items():
            if key.startswith('fc.'):
                new_key = key.replace('fc.', 'fc_heads.0.')
                new_state[new_key] = value
                route_key = key.replace('fc.', 'fc_route.')
                new_state[route_key] = value.clone()
            else:
                new_state[key] = value

        missing, unexpected = self.load_state_dict(new_state, strict=False)

        with torch.no_grad():
            for i in range(1, len(self.fc_heads)):
                self.fc_heads[i].weight.data.copy_(self.fc_heads[0].weight.data)
                self.fc_heads[i].bias.data.copy_(self.fc_heads[0].bias.data)

        if missing:
            print(f'[load_pretrained] Missing keys (using default init): {len(missing)} keys')
        if unexpected:
            print(f'[load_pretrained] Unexpected keys (ignored): {unexpected}')

    # -------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------
    def _extract_features_with_task_list(self, input, task_list, use_se=True):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task_list[0]](x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task_list[1], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task_list[2], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task_list[3], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task_list[4], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task_list[5], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task_list[6], use_se=use_se)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def _extract_adapted_features(self, input, task_list, adapter_task=None):
        features = self._extract_features_with_task_list(input, task_list)
        if adapter_task is None:
            adapter_task = task_list[0]
        return self.adapter(features, adapter_task)

    def _forward_with_task_list(self, input, task_list, head_task=None):
        adapted = self._extract_adapted_features(input, task_list)
        if head_task is None:
            head_task = task_list[0]
        return self.fc_heads[head_task](adapted)

    def forward(self, input, task=1):
        task_list = [task] * 7
        return self._forward_with_task_list(input, task_list, head_task=task)

    def forward_route(self, input, task=1):
        task_list = [task] * 7
        features = self._extract_features_with_task_list(input, task_list, use_se=False)
        return self.fc_route(features)

    def extract_embedding(self, input, task=1):
        task_list = [task] * 7
        return self._extract_adapted_features(input, task_list, adapter_task=task)

    def forward_split(self, input, early_task=0, late_task=0, split_after=4):
        task_list = [early_task] * 5 + [late_task] * 2
        return self._forward_with_task_list(input, task_list, head_task=late_task)

    def extract_embedding_split(self, input, early_task=0, late_task=0, split_after=4):
        task_list = [early_task] * 5 + [late_task] * 2
        return self._extract_adapted_features(input, task_list, adapter_task=late_task)
