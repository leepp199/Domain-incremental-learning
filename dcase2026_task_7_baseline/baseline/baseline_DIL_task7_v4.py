"""
baseline_DIL_task7_v4.py – V4 anti-forgetting incremental learning baseline.

Key changes vs V2:
  1. BN stats protection: during training, only the current task's BN layers are
     in train mode. All other tasks' BN layers stay in eval mode, preventing
     running_mean/running_var corruption when computing auxiliary losses.
  2. BN stats snapshot: after each task's training, BN running stats are
     snapshotted and can be restored as a safety net.
  3. Eval mode during prototype refresh to prevent BN stats drift.
"""
import pandas as pd

import os
import sys

import torch

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import config_task7 as config

import torch.nn.functional as F

from config_task7 import (sample_rate, mel_bins, fmin, fmax, window_size,
                    hop_size)

import torch.optim as optim
from domain_net_v4 import *

from datasetfactory_task7 import DILDatasetInc as DILDataset
from sklearn import metrics
from tqdm import tqdm
from utilities import *
import time

timestr = time.strftime("%Y%m%d-%H%M%S")


def _enable_bn_adapt_only(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def _compute_task_predictions(model, inputs, nb_tasks):
    """Classification predictions using per-task FC heads."""
    outputs_uncs = []
    for task in range(nb_tasks):
        with torch.no_grad():
            outputs = model(inputs, task)
        outputs = torch.softmax(outputs, dim=1)
        outputs_uncs.append(outputs.detach())
    return torch.stack(outputs_uncs, dim=0)


def _compute_task_routing_scores(model, inputs, nb_tasks):
    """Routing scores using shared fc_route (comparable across tasks)."""
    outputs_uncs = []
    for task in range(nb_tasks):
        with torch.no_grad():
            outputs = model.forward_route(inputs, task)
        outputs = torch.softmax(outputs, dim=1)
        outputs_uncs.append(outputs.detach())
    return torch.stack(outputs_uncs, dim=0)


def _crop_or_pad_waveform(waveform, clip_samples, random_crop=True):
    if len(waveform) < clip_samples:
        return np.concatenate((waveform, np.zeros(clip_samples - len(waveform), dtype=waveform.dtype)))
    if len(waveform) == clip_samples:
        return waveform

    if random_crop:
        start = np.random.randint(0, len(waveform) - clip_samples + 1)
    else:
        start = 0
    return waveform[start:start + clip_samples]


def _train_collate_fn(batch):
    audio_list, label_list, file_list = zip(*batch)
    cropped_audio = [
        _crop_or_pad_waveform(audio, config.clip_samples, random_crop=True)
        for audio in audio_list
    ]
    return (
        torch.tensor(np.stack(cropped_audio), dtype=torch.float32),
        torch.tensor(np.stack(label_list), dtype=torch.float32),
        list(file_list),
    )


def _prototype_collate_fn(batch):
    audio_list, label_list, file_list = zip(*batch)
    padded_audio = [
        _crop_or_pad_waveform(audio, config.clip_samples, random_crop=False)
        for audio in audio_list
    ]
    return (
        torch.tensor(np.stack(padded_audio), dtype=torch.float32),
        torch.tensor(np.stack(label_list), dtype=torch.float32),
        list(file_list),
    )


def _resolve_save_checkpoint_dir(args):
    if args.checkpoint_dir:
        return args.checkpoint_dir

    base_dir = os.path.dirname(os.path.normpath(config.save_resume_path))
    return os.path.join(base_dir, 'BN_research', args.experiment_name)


def _resolve_resume_checkpoint_dir(args):
    if args.resume_checkpoint_dir:
        return args.resume_checkpoint_dir
    return config.save_resume_path


def _should_resume_current_task(args, cur_task):
    if not args.resume:
        return False

    if args.resume_mode == 'all':
        return True

    if args.resume_mode == 'd1_only':
        return cur_task == 0

    if args.resume_mode == 'd1_d2':
        return cur_task <= 1

    return False


def _copy_bn_module_state(source_bn, target_bn):
    target_bn.weight.data.copy_(source_bn.weight.data)
    target_bn.bias.data.copy_(source_bn.bias.data)
    target_bn.running_mean.data.copy_(source_bn.running_mean.data)
    target_bn.running_var.data.copy_(source_bn.running_var.data)
    target_bn.num_batches_tracked.data.copy_(source_bn.num_batches_tracked.data)


def _clone_task_bn_parameters(model, source_task, target_task):
    _copy_bn_module_state(model.bn0[source_task], model.bn0[target_task])
    for block_name in ['conv_block1', 'conv_block2', 'conv_block3', 'conv_block4', 'conv_block5', 'conv_block6']:
        block = getattr(model, block_name)
        _copy_bn_module_state(block.bnF[source_task], block.bnF[target_task])
        _copy_bn_module_state(block.bnS[source_task], block.bnS[target_task])
        if hasattr(block, 'se'):
            source_se = block.se.se_layers[source_task]
            target_se = block.se.se_layers[target_task]
            target_se.load_state_dict(source_se.state_dict())
    if hasattr(model, 'adapter'):
        source_adapter = model.adapter.adapters[source_task]
        target_adapter = model.adapter.adapters[target_task]
        target_adapter.load_state_dict(source_adapter.state_dict())
    if hasattr(model, 'fc_heads'):
        model.fc_heads[target_task].weight.data.copy_(model.fc_heads[source_task].weight.data)
        model.fc_heads[target_task].bias.data.copy_(model.fc_heads[source_task].bias.data)


def _compute_task_embeddings(model, inputs, nb_tasks):
    embedding_list = []
    for task in range(nb_tasks):
        with torch.no_grad():
            embedding = model.extract_embedding(inputs, task)
        embedding_list.append(embedding.detach())
    return torch.stack(embedding_list, dim=0)


def _compute_prototype_distances(task_embeddings, prototype_bank, nb_tasks):
    distance_list = []
    for task in range(nb_tasks):
        prototype = prototype_bank.get(task)
        if prototype is None:
            distance = torch.full(
                (task_embeddings.shape[1],),
                1e6,
                device=task_embeddings.device,
                dtype=task_embeddings.dtype,
            )
        else:
            prototype = prototype.to(task_embeddings.device)
            distance = torch.sum((task_embeddings[task] - prototype.unsqueeze(0)) ** 2, dim=-1)
        distance_list.append(distance)
    return torch.stack(distance_list, dim=0)


def _sparse_topk_weights(raw_scores, topk):
    num_tasks = raw_scores.shape[0]
    if topk <= 0 or topk >= num_tasks:
        return torch.softmax(raw_scores, dim=0)

    top_values, top_indices = torch.topk(raw_scores, k=topk, dim=0)
    sparse_scores = torch.full_like(raw_scores, float('-inf'))
    sparse_scores.scatter_(0, top_indices, top_values)
    return torch.softmax(sparse_scores, dim=0)


def _parse_tta_shift_ratios(tta_shifts):
    if isinstance(tta_shifts, str):
        ratios = []
        for token in tta_shifts.split(','):
            token = token.strip()
            if not token:
                continue
            ratios.append(float(token))
    else:
        ratios = [float(x) for x in tta_shifts]

    if not ratios:
        ratios = [0.0]
    return ratios


def _shift_waveform_batch(inputs, shift_ratio):
    if abs(shift_ratio) < 1e-12:
        return inputs
    shift = int(round(inputs.shape[-1] * shift_ratio))
    if shift == 0:
        return inputs
    return torch.roll(inputs, shifts=shift, dims=-1)


def _sanitize_entropy(entropy, num_classes):
    """Replace NaN/inf entropy with max entropy penalty."""
    max_entropy = float(np.log(num_classes))
    return torch.where(torch.isfinite(entropy), entropy,
                       torch.full_like(entropy, max_entropy))


def _compute_routing_probabilities(outputs_uncs, entropy, prototype_distances, args):
    routing_mode = args.routing_mode
    routing_temp = max(float(args.routing_temp), 1e-6)
    hard_fallback_thresh = float(getattr(args, 'hard_fallback_thresh', 0.0))
    routing_start_task = int(getattr(args, 'routing_start_task', 0))

    # V4 FIX: sanitize entropy to handle NaN from degenerate routing
    num_classes = outputs_uncs.shape[-1]
    entropy = _sanitize_entropy(entropy, num_classes)

    def _apply_task_mask(scores):
        """Mask out excluded tasks by setting their scores to -inf."""
        if routing_start_task > 0:
            scores[:routing_start_task] = float('-inf')
        return scores

    if routing_mode == 'soft':
        raw_scores = _apply_task_mask(-entropy / routing_temp)
        weights = _sparse_topk_weights(raw_scores, int(args.routing_topk)).unsqueeze(-1)
        if hard_fallback_thresh > 0:
            weights = _apply_hard_fallback(weights, hard_fallback_thresh)
        return torch.sum(weights * outputs_uncs, dim=0)

    if routing_mode == 'prototype' and prototype_distances is not None:
        raw_scores = _apply_task_mask(-prototype_distances / routing_temp)
        weights = _sparse_topk_weights(raw_scores, int(args.routing_topk)).unsqueeze(-1)
        if hard_fallback_thresh > 0:
            weights = _apply_hard_fallback(weights, hard_fallback_thresh)
        return torch.sum(weights * outputs_uncs, dim=0)

    if routing_mode == 'hybrid':
        entropy_term = -entropy / routing_temp
        confidence_term = torch.max(outputs_uncs, dim=-1).values / routing_temp
        alpha = float(args.hybrid_entropy_weight)
        beta = float(args.hybrid_conf_weight)
        combined_scores = alpha * entropy_term + beta * confidence_term

        if prototype_distances is not None:
            gamma = float(args.hybrid_proto_weight)
            combined_scores = combined_scores + gamma * (-prototype_distances / routing_temp)

        combined_scores = _apply_task_mask(combined_scores)
        weights = _sparse_topk_weights(combined_scores, int(args.routing_topk)).unsqueeze(-1)
        if hard_fallback_thresh > 0:
            weights = _apply_hard_fallback(weights, hard_fallback_thresh)
        return torch.sum(weights * outputs_uncs, dim=0)

    if routing_mode == 'conf_proto':
        # V4 NEW: route using classification head confidence + prototype distance
        # Avoids unreliable fc_route entropy entirely
        confidence_term = torch.max(outputs_uncs, dim=-1).values / routing_temp
        beta = float(args.hybrid_conf_weight)
        combined_scores = beta * confidence_term

        if prototype_distances is not None:
            gamma = float(args.hybrid_proto_weight)
            combined_scores = combined_scores + gamma * (-prototype_distances / routing_temp)

        combined_scores = _apply_task_mask(combined_scores)
        weights = _sparse_topk_weights(combined_scores, int(args.routing_topk)).unsqueeze(-1)
        if hard_fallback_thresh > 0:
            weights = _apply_hard_fallback(weights, hard_fallback_thresh)
        return torch.sum(weights * outputs_uncs, dim=0)

    return None


def _apply_hard_fallback(weights, threshold):
    max_weights, _ = torch.max(weights, dim=0, keepdim=True)
    mask = (weights >= max_weights * 0.999)
    use_hard = (max_weights >= threshold).expand_as(weights)
    hard_weights = torch.zeros_like(weights)
    hard_weights[mask] = 1.0
    return torch.where(use_hard, hard_weights, weights)


def _compute_prototype_regularization_loss(model, audio, current_task, args):
    if current_task <= 0:
        return audio.new_zeros(()), audio.new_zeros(())

    embedding = model.extract_embedding(audio, current_task)
    compact_loss = audio.new_zeros(())
    separation_loss = audio.new_zeros(())

    if float(args.prototype_compact_weight) > 0.0:
        center = embedding.mean(dim=0, keepdim=True)
        compact_loss = torch.mean(torch.sum((embedding - center) ** 2, dim=-1))

    if float(args.prototype_separation_weight) > 0.0:
        prototype_bank = getattr(model, 'prototype_bank', {})
        previous_ids = [task_id for task_id in range(current_task) if task_id in prototype_bank]
        if previous_ids:
            margin = float(args.prototype_margin)
            separation_terms = []
            for task_id in previous_ids:
                prototype = prototype_bank[task_id].to(embedding.device)
                distance = torch.sum((embedding - prototype.unsqueeze(0)) ** 2, dim=-1)
                separation_terms.append(torch.relu(margin - distance))
            separation_loss = torch.stack(separation_terms, dim=0).mean()

    return compact_loss, separation_loss


def _compute_training_aux_losses(model, audio, current_task, current_logits, args):
    """Compute auxiliary losses.

    V4 FIX: Previous tasks' BN layers are already in eval mode (set by
    set_task_bn_train_only at the start of each epoch), so forward passes
    through old task paths will NOT corrupt their running_mean/running_var.
    """
    if current_task <= 0:
        zero = current_logits.new_zeros(())
        return zero, zero, zero

    probs_per_task = []
    for task_id in range(current_task + 1):
        if task_id == current_task:
            logits = current_logits
        else:
            with torch.no_grad():
                logits = model(audio, task_id)
        probs_per_task.append(torch.softmax(logits, dim=1))

    probs_per_task = torch.stack(probs_per_task, dim=0)
    epsilon = sys.float_info.min
    entropies = -torch.sum(probs_per_task * torch.log(probs_per_task + epsilon), dim=-1)

    current_entropy = entropies[current_task].unsqueeze(0)
    previous_entropies = entropies[:current_task]
    routing_margin = float(args.routing_margin)
    routing_loss = torch.relu(current_entropy - previous_entropies + routing_margin).mean()

    routing_temp = max(float(args.routing_temp), 1e-6)
    routing_weights = torch.softmax(-entropies / routing_temp, dim=0).unsqueeze(-1)
    ensemble_probs = torch.sum(routing_weights * probs_per_task, dim=0).detach()
    consistency_loss = F.kl_div(
        F.log_softmax(current_logits, dim=1),
        ensemble_probs,
        reduction='batchmean',
    )

    kd_temperature = max(float(args.pseudo_kd_temperature), 1e-6)
    if args.pseudo_kd_source == 'all_prev':
        teacher_logits = []
        for task_id in range(current_task):
            with torch.no_grad():
                teacher_logits.append(model(audio, task_id) / kd_temperature)
        teacher_logits = torch.stack(teacher_logits, dim=0).mean(dim=0)
    else:
        with torch.no_grad():
            teacher_logits = model(audio, 0) / kd_temperature

    kd_loss = F.kl_div(
        F.log_softmax(current_logits / kd_temperature, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction='batchmean',
    ) * (kd_temperature ** 2)

    return routing_loss, consistency_loss, kd_loss


def _compute_accuracy(model, loader, task, device):

    correct, total = 0, 0
    model.eval()
    correct, total = 0, 0
    output_dict = {}

    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, task)
        outputs = torch.softmax(outputs, dim=1)
        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
        append_to_dict(output_dict, 'clipwise_output',
                       outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def _compute_uncertainity(model, loader, seen_domains, device, args):
    model.eval()
    nb_tasks = len(seen_domains) + 1 # +1 is D1
    routing_start_task = int(getattr(args, 'routing_start_task', 0))
    correct, total = 0, 0
    output_dict = {}
    output_path = config.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    routing_mode = args.routing_mode
    routing_temp = max(float(args.routing_temp), 1e-6)
    use_blockwise = args.blockwise_routing
    use_ttbn = args.ttbn_adapt
    prototype_bank = getattr(model, 'prototype_bank', {})
    tta_shift_ratios = _parse_tta_shift_ratios(args.tta_shifts)

    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float()
        inputs = inputs.to(device)
        tta_outputs = []
        for shift_ratio in tta_shift_ratios:
            routed_inputs = _shift_waveform_batch(inputs, shift_ratio)

            # --- ROUTING stage ---
            routing_uncs = _compute_task_routing_scores(model, routed_inputs, nb_tasks)

            epsilon = 1e-8
            routing_uncs_safe = routing_uncs.clamp(min=epsilon, max=1.0 - epsilon)
            entropy = -torch.sum(routing_uncs_safe * torch.log(routing_uncs_safe), dim=-1)
            max_entropy = float(np.log(routing_uncs.shape[-1]))
            entropy = torch.where(torch.isfinite(entropy), entropy,
                                  torch.full_like(entropy, max_entropy))

            # V4: skip degenerate early tasks from routing
            if routing_start_task > 0:
                entropy[:routing_start_task] = max_entropy

            hard_task_id = torch.argmin(entropy, dim=0)

            if use_ttbn:
                model.eval()
                model.apply(_enable_bn_adapt_only)
                with torch.no_grad():
                    for bi in range(routed_inputs.shape[0]):
                        _ = model.forward_route(routed_inputs[bi:bi + 1], int(hard_task_id[bi].item()))
                model.eval()
                routing_uncs = _compute_task_routing_scores(model, routed_inputs, nb_tasks)
                routing_uncs_safe = routing_uncs.clamp(min=epsilon, max=1.0 - epsilon)
                entropy = -torch.sum(routing_uncs_safe * torch.log(routing_uncs_safe), dim=-1)
                entropy = torch.where(torch.isfinite(entropy), entropy,
                                      torch.full_like(entropy, max_entropy))
                if routing_start_task > 0:
                    entropy[:routing_start_task] = max_entropy
                hard_task_id = torch.argmin(entropy, dim=0)

            prototype_distances = None
            if routing_mode in ['prototype', 'hybrid']:
                task_embeddings = _compute_task_embeddings(model, routed_inputs, nb_tasks)
                prototype_distances = _compute_prototype_distances(task_embeddings, prototype_bank, nb_tasks)

            # --- CLASSIFICATION stage: use per-task FC heads ---
            outputs_uncs = _compute_task_predictions(model, routed_inputs, nb_tasks)

            routed_outputs = _compute_routing_probabilities(
                outputs_uncs,
                entropy,
                prototype_distances,
                args,
            )
            if routed_outputs is not None:
                outputs = routed_outputs
            elif use_blockwise:
                max_prob = torch.max(routing_uncs, dim=-1).values
                late_task_id = torch.argmax(max_prob, dim=0)
                outputs_list = []
                for bi in range(routed_inputs.shape[0]):
                    with torch.no_grad():
                        logits = model.forward_split(
                            routed_inputs[bi:bi + 1],
                            early_task=int(hard_task_id[bi].item()),
                            late_task=int(late_task_id[bi].item()),
                            split_after=4,
                        )
                    outputs_list.append(torch.softmax(logits, dim=1))
                outputs = torch.cat(outputs_list, dim=0)
            else:
                with torch.no_grad():
                    outputs_list = []
                    for bi in range(routed_inputs.shape[0]):
                        logits = model(routed_inputs[bi:bi + 1], int(hard_task_id[bi].item()))
                        outputs_list.append(torch.softmax(logits, dim=1))
                    outputs = torch.cat(outputs_list, dim=0)

            tta_outputs.append(outputs)

        outputs = torch.stack(tta_outputs, dim=0).mean(dim=0)

        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
        append_to_dict(output_dict, 'clipwise_output',
                       outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())
        with open(os.path.join(output_path  + 'output_' + timestr + '.txt'), 'a') as f:
            inv_labels = {v: k for k, v in config.dict_class_labels.items()}
            for bi in range(len(audio_file)):
                pred_idx = int(predicts[bi].cpu().item())
                class_label = inv_labels.get(pred_idx, str(pred_idx))
                f.write(audio_file[bi] + '\t' + class_label + '\n')
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class Learner():
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num, num_tasks):
        super(Learner, self).__init__()

        Model = MCnn14
        self.model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                           classes_num, num_tasks)

        self.classes_seen = 0
        self.known_classes = 0
        self.cur_task = -1
        self.class_increments = []
        self.prototype_bank = {}
        self.prototype_counts = {}

    def _prototype_state_path(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, 'prototype_bank.pt')

    def _maybe_load_prototype_state(self, checkpoint_dir, device):
        prototype_path = self._prototype_state_path(checkpoint_dir)
        if not os.path.exists(prototype_path):
            return

        state = torch.load(prototype_path, map_location=torch.device(device))
        self.prototype_bank = {
            int(task_id): tensor.to(device)
            for task_id, tensor in state.get('prototype_bank', {}).items()
        }
        self.prototype_counts = {
            int(task_id): int(count)
            for task_id, count in state.get('prototype_counts', {}).items()
        }
        self.model.prototype_bank = self.prototype_bank

    def _save_prototype_state(self, checkpoint_dir):
        state = {
            'prototype_bank': {task_id: tensor.detach().cpu() for task_id, tensor in self.prototype_bank.items()},
            'prototype_counts': self.prototype_counts,
        }
        torch.save(state, self._prototype_state_path(checkpoint_dir))

    def _update_task_prototype(self, prototype, task_id, sample_count):
        if task_id in self.prototype_bank:
            prev_count = self.prototype_counts.get(task_id, 0)
            total_count = prev_count + sample_count
            mixed = (self.prototype_bank[task_id] * prev_count + prototype * sample_count) / max(total_count, 1)
            self.prototype_bank[task_id] = mixed.detach()
            self.prototype_counts[task_id] = total_count
        else:
            self.prototype_bank[task_id] = prototype.detach()
            self.prototype_counts[task_id] = sample_count
        self.model.prototype_bank = self.prototype_bank

    def refresh_prototypes(self, train_df, batch_size, num_workers, device):
        dataset_proto = DILDataset(train_df, config.audio_folder_DIL)
        proto_loader = torch.utils.data.DataLoader(
            dataset=dataset_proto,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_prototype_collate_fn,
        )

        def compute_proto(task_id):
            embedding_sum = None
            sample_count = 0
            # V4 FIX: ensure eval mode during prototype computation
            self.model.eval()
            with torch.no_grad():
                for audio, _, _ in proto_loader:
                    audio = audio.to(device)
                    embedding = self.model.extract_embedding(audio, task_id)
                    if embedding_sum is None:
                        embedding_sum = embedding.sum(dim=0)
                    else:
                        embedding_sum += embedding.sum(dim=0)
                    sample_count += embedding.shape[0]
            if embedding_sum is None or sample_count == 0:
                return None, 0
            return embedding_sum / sample_count, sample_count

        current_proto, current_count = compute_proto(self.cur_task)
        if current_proto is not None:
            self._update_task_prototype(current_proto, self.cur_task, current_count)

        if 0 not in self.prototype_bank:
            d1_proto, d1_count = compute_proto(0)
            if d1_proto is not None:
                self._update_task_prototype(d1_proto, 0, d1_count)

        self.model.prototype_bank = self.prototype_bank

    def incremental_train(self, train_loader, val_loader, device, args):

        step = 0
        total = 0
        correct = 0
        check_point = 50
        self.model.to(device)

        self.model.freeze_weight()
        if args.bn_clone_init and self.cur_task > 0:
            source_task = self.cur_task - 1 if args.bn_clone_source == 'prev' else 0
            _clone_task_bn_parameters(self.model, source_task, self.cur_task)

        if self.cur_task == 0:
            for name, param in self.model.named_parameters():
                if 'conv1' in name or 'conv2' in name:
                    param.requires_grad = True
                if 'bn' in name or 'domain_' in name:
                    if '.{}.weight'.format(self.cur_task) in name or '.{}.bias'.format(self.cur_task) in name:
                        param.requires_grad = True
                if 'fc_heads.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
                if 'se.se_layers.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
                if 'adapter.adapters.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate
        elif self.cur_task > 0:
            for name, param in self.model.named_parameters():
                if 'bn' in name or 'domain_' in name:
                    if '.{}.weight'.format(self.cur_task) in name or '.{}.bias'.format(self.cur_task) in name:
                        param.requires_grad = True
                if 'fc_heads.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
                if 'se.se_layers.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
                if 'adapter.adapters.{}.'.format(self.cur_task) in name:
                    param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate / 10

        print(f"params to be adapted")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        criteria = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing)

        optimizer = torch.optim.Adam(non_frozen_parameters, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0., amsgrad=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.001)

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()
            # V4 CRITICAL FIX: only current task's BN in train mode.
            # All other tasks' BN layers stay in eval mode to prevent
            # running_mean/running_var corruption during aux loss forward passes.
            self.model.set_task_bn_train_only(self.cur_task)

            sum_loss = 0
            sum_routing_loss = 0
            sum_consistency_loss = 0
            sum_class_loss = 0
            sum_kd_loss = 0
            sum_proto_compact = 0
            sum_proto_sep = 0
            for batch_idx, (audio, target, _) in enumerate(train_loader):
                optimizer.zero_grad()
                audio = audio.float()
                target = target.float()
                audio = audio.to(device)
                target = target.to(device)
                target_indices = torch.argmax(target, dim=-1)

                logits = self.model(audio, self.cur_task)
                class_loss = criteria(logits, target_indices)
                routing_loss, consistency_loss, kd_loss = _compute_training_aux_losses(
                    self.model,
                    audio,
                    self.cur_task,
                    logits,
                    args,
                )
                proto_compact_loss, proto_separation_loss = _compute_prototype_regularization_loss(
                    self.model,
                    audio,
                    self.cur_task,
                    args,
                )
                loss = class_loss
                loss = loss + args.routing_loss_weight * routing_loss
                loss = loss + args.consistency_loss_weight * consistency_loss
                loss = loss + args.pseudo_kd_weight * kd_loss
                loss = loss + args.prototype_compact_weight * proto_compact_loss
                loss = loss + args.prototype_separation_weight * proto_separation_loss

                sum_loss += loss.item()
                sum_class_loss += class_loss.item()
                sum_routing_loss += routing_loss.item()
                sum_consistency_loss += consistency_loss.item()
                sum_kd_loss += kd_loss.item()
                sum_proto_compact += proto_compact_loss.item()
                sum_proto_sep += proto_separation_loss.item()
                loss.backward()
                optimizer.step()
                step += 1

                if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                    print('==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, class: {:.3f}, routing: {:.3f}, consistency: {:.3f}, kd: {:.3f}, proto_c: {:.3f}, proto_s: {:.3f}'.
                          format(
                              epoch_idx,
                              batch_idx + 1,
                              step,
                              sum_loss / (batch_idx + 1),
                              sum_class_loss / (batch_idx + 1),
                              sum_routing_loss / (batch_idx + 1),
                              sum_consistency_loss / (batch_idx + 1),
                              sum_kd_loss / (batch_idx + 1),
                              sum_proto_compact / (batch_idx + 1),
                              sum_proto_sep / (batch_idx + 1),
                          ))

            scheduler.step()

        # V4: snapshot BN stats after training for safety
        self.model.save_bn_snapshot(self.cur_task)
        for prev_task in range(self.cur_task):
            self.model.save_bn_snapshot(prev_task)
        print(f'[V4] BN stats snapshotted for tasks 0..{self.cur_task}')

        if args.save:
            save_path = _resolve_save_checkpoint_dir(args)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(),
                       os.path.join(save_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth'))
            self._save_prototype_state(save_path)

    def load_checkpoint(self, device, args):
        resume_dir = _resolve_resume_checkpoint_dir(args)
        resume_path = os.path.join(resume_dir, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth')
        if self.cur_task == 0 and not os.path.exists(resume_path):
            resume_dir = config.save_resume_path
            resume_path = os.path.join(resume_dir, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth')
        state_dict = torch.load(resume_path, map_location=torch.device(device))
        if 'fc.weight' in state_dict:
            print('[load_checkpoint] Detected v1 checkpoint, using compatible loader')
            self.model.load_pretrained_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self._maybe_load_prototype_state(resume_dir, device)
        print('model trained on Task D{} is loaded from {}'.format(self.cur_task + 1, resume_dir))

    def incremental_setup(self, train_df, valid_df, seen_domains, batch_size, num_workers, device, args):

        self.cur_task += 1

        if self.cur_task == 0:
            self.load_checkpoint(device, args)
            self.cur_task += 1  # Skip the domain D1

        print("Starting DIL Task D{}".format(self.cur_task + 1))

        dataset_train = DILDataset(train_df, config.audio_folder_DIL)
        dataset_val = DILDataset(valid_df, config.audio_folder_DIL)

        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True,
                                                   collate_fn=_train_collate_fn)

        validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                      num_workers=num_workers, pin_memory=True)

        if _should_resume_current_task(args, self.cur_task):
            self.load_checkpoint(device, args)
        else:
            self.incremental_train(train_loader, validate_loader, device, args)

        if args.routing_mode in ['prototype', 'hybrid'] or args.save:
            self.model.to(device)
            self.refresh_prototypes(train_df, batch_size, num_workers, device)

        if args.save:
            save_path = _resolve_save_checkpoint_dir(args)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self._save_prototype_state(save_path)

    def acc_prev(self, seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device, args):
        self.model.to(device)
        self.model.eval()
        num_domains = len(seen_domains)
        domain_dict = {}
        accuracy_previous = []
        for domain in range(num_domains):
            train_df = df_dev_train[df_dev_train['domain'].isin(seen_domains[domain])]
            valid_df = df_dev_test[df_dev_test['domain'].isin(seen_domains[domain])]
            dataset_val = DILDataset(valid_df, config.audio_folder_DIL)
            validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                          num_workers=num_workers, pin_memory=True)

            accuracy = _compute_uncertainity(self.model, validate_loader, seen_domains, device, args)

            print('seen domain: {} and its accuracy: {}'.format(seen_domains[domain], accuracy))

            accuracy_previous.append(accuracy)

        average_accuracy = np.mean(accuracy_previous).item()
        return average_accuracy

def train(args):
    classes_num = config.classes_num_DIL
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    df_dev_train = config.df_DIL_dev_train
    df_dev_test = config.df_DIL_dev_test

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = args.num_workers

    dil_tasks_0 = ['D1']

    dil_task_1 = ['D2']

    dil_task_2 = ['D3']

    dil_tasks = [dil_task_1, dil_task_2]
    print('Tasks:', dil_tasks)

    np.random.seed(1193)

    num_tasks = len(dil_tasks) + 1
    seen_domains = []
    model = Learner(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                    classes_num, num_tasks)

    for task in range(len(dil_tasks)):
        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]

        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        seen_accuracy = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device, args)
        print('Average Accuracy: ', seen_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')

    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--num_workers', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--resume', action='store_true', default=False)
    parser_train.add_argument('--save', action='store_true', default=False)
    parser_train.add_argument('--label_smoothing', type=float, default=0.0)
    parser_train.add_argument('--routing_loss_weight', type=float, default=0.0)
    parser_train.add_argument('--routing_margin', type=float, default=0.1)
    parser_train.add_argument('--consistency_loss_weight', type=float, default=0.0)
    parser_train.add_argument('--checkpoint_dir', type=str, default='')
    parser_train.add_argument('--resume_checkpoint_dir', type=str, default='')
    parser_train.add_argument('--experiment_name', type=str, default='routeaware_' + timestr)
    parser_train.add_argument('--routing_temp', type=float, default=1.0)
    parser_train.add_argument('--blockwise_routing', action='store_true', default=False)
    parser_train.add_argument('--ttbn_adapt', action='store_true', default=False)
    parser_train.add_argument('--pseudo_kd_weight', type=float, default=0.0)
    parser_train.add_argument('--pseudo_kd_temperature', type=float, default=2.0)
    parser_train.add_argument('--pseudo_kd_source', type=str, choices=['d1', 'all_prev'], default='d1')
    parser_train.add_argument('--bn_clone_init', action='store_true', default=False)
    parser_train.add_argument('--bn_clone_source', type=str, choices=['prev', 'd1'], default='prev')
    parser_train.add_argument('--routing_mode', type=str, choices=['hard', 'soft', 'prototype', 'hybrid', 'conf_proto'], default='hard')
    parser_train.add_argument('--routing_topk', type=int, default=0)
    parser_train.add_argument('--hybrid_entropy_weight', type=float, default=1.0)
    parser_train.add_argument('--hybrid_conf_weight', type=float, default=0.35)
    parser_train.add_argument('--hybrid_proto_weight', type=float, default=0.65)
    parser_train.add_argument('--hard_fallback_thresh', type=float, default=0.0,
                              help='When top routing weight >= threshold, use hard routing for that sample (0=disabled)')
    parser_train.add_argument('--tta_shifts', type=str, default='0.0')
    parser_train.add_argument('--prototype_compact_weight', type=float, default=0.0)
    parser_train.add_argument('--prototype_separation_weight', type=float, default=0.0)
    parser_train.add_argument('--prototype_margin', type=float, default=8.0)
    parser_train.add_argument('--routing_start_task', type=int, default=0,
                              help='Skip tasks < this index from routing (0=use all, 1=skip D1)')
    parser_train.add_argument('--resume_mode', type=str, choices=['all', 'd1_only', 'd1_d2'], default='all')

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')
