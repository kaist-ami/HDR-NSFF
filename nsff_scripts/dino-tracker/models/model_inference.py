"""
model_inference.py

Extended ModelInference class for DINO-tracker with tensor-based flow computation.

Changes vs. upstream model_inference.py:
  - Added compute_framewise_flow_tensor()        (foreground-masked, vectorized)
  - Added compute_framewise_flow_tensor_bwd()    (foreground-masked, vectorized)
  - Added compute_framewise_flow_tensor_all_pixels()      (all pixels)
  - Added compute_framewise_flow_tensor_all_pixels_bwd()  (all pixels)
  - Added compute_framewise_flow() / compute_framewise_flow_bwd()  (legacy, point-by-point)
"""
from typing import Dict, List
import torch
from tqdm import tqdm
from data.dataset import RangeNormalizer
from models.tracker import Tracker


# ---------------------------------------------------------------------------
# Trajectory generation helpers
# ---------------------------------------------------------------------------

def generate_trajectory_input(query_point, video, start_t=None, end_t=None):
    start_t = 0 if start_t is None else start_t
    end_t = video.shape[0] if end_t is None else end_t
    video_subset = video[start_t:end_t]
    rest = video_subset.shape[0]
    device = video.device

    source_points = query_point.unsqueeze(0).repeat(rest, 1)
    frames_set_t = torch.arange(start_t, end_t, dtype=torch.long, device=device)
    frames_set_t = torch.cat([torch.tensor([query_point[2]], device=device), frames_set_t]).int()
    source_frame_indices = torch.tensor([0], device=device).repeat(end_t - start_t)
    target_frame_indices = torch.arange(rest, dtype=torch.long, device=device) + 1

    return source_points, source_frame_indices, target_frame_indices, frames_set_t


@torch.no_grad()
def generate_trajectory(query_point, video, model, range_normalizer, dst_range=(-1, 1),
                        use_raw_features=False, batch_size=None):
    batch_size = video.shape[0] if batch_size is None else batch_size
    trajectory_pred = []
    for start_t in range(0, video.shape[0], batch_size):
        end_t = min(start_t + batch_size, video.shape[0])
        trajectory_input = generate_trajectory_input(query_point, video, start_t=start_t, end_t=end_t)
        trajectory_coordinate_preds_normalized = model(trajectory_input, use_raw_features=use_raw_features)
        trajectory_coordinate_preds = range_normalizer.unnormalize(
            trajectory_coordinate_preds_normalized, dims=[0, 1], src=dst_range)
        trajectory_timesteps = trajectory_input[-1][1:].to(dtype=torch.float32)
        trajectory_pred_cur = torch.cat(
            [trajectory_coordinate_preds, trajectory_timesteps.unsqueeze(dim=1)], dim=1)
        trajectory_pred.append(trajectory_pred_cur)
    return torch.cat(trajectory_pred, dim=0)


@torch.no_grad()
def generate_trajectories(query_points, video, model, range_normalizer, dst_range=(-1, 1),
                          use_raw_features=False, batch_size=None):
    trajectories_list = []
    query_points = query_points.to(dtype=torch.float32)
    for query_point in query_points:
        trajectory_pred = generate_trajectory(
            query_point=query_point, video=video, model=model,
            range_normalizer=range_normalizer, dst_range=dst_range,
            use_raw_features=use_raw_features, batch_size=batch_size)
        trajectories_list.append(trajectory_pred)
    return torch.stack(trajectories_list)


@torch.no_grad()
def generate_flow_trajectory(query_point, video, model, range_normalizer,
                             start_t=0, end_t=None, dst_range=(-1, 1),
                             use_raw_features=False, batch_size=None):
    if end_t is None:
        end_t = start_t + 2
    end_t = min(end_t, video.shape[0])
    if batch_size is None:
        batch_size = end_t - start_t

    trajectory_pred = []
    for st in range(start_t, end_t, batch_size):
        e_t = min(st + batch_size, end_t)
        trajectory_input = generate_trajectory_input(query_point, video, start_t=st, end_t=e_t)
        trajectory_coordinate_preds_normalized = model(trajectory_input, use_raw_features=use_raw_features)
        trajectory_coordinate_preds = range_normalizer.unnormalize(
            trajectory_coordinate_preds_normalized, dims=[0, 1], src=dst_range)
        trajectory_timesteps = trajectory_input[-1][1:].to(dtype=torch.float32)
        trajectory_pred_cur = torch.cat(
            [trajectory_coordinate_preds, trajectory_timesteps.unsqueeze(dim=1)], dim=1)
        trajectory_pred.append(trajectory_pred_cur)
    return torch.cat(trajectory_pred, dim=0)


# ---------------------------------------------------------------------------
# ModelInference
# ---------------------------------------------------------------------------

class ModelInference(torch.nn.Module):
    def __init__(self, model: Tracker, range_normalizer: RangeNormalizer,
                 anchor_cosine_similarity_threshold: float = 0.5,
                 cosine_similarity_threshold: float = 0.5) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.cache_refined_embeddings()
        self.range_normalizer = range_normalizer
        self.anchor_cosine_similarity_threshold = anchor_cosine_similarity_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold

    # ---- Trajectory --------------------------------------------------------

    def compute_trajectories(self, query_points, batch_size=None):
        return generate_trajectories(
            query_points=query_points, model=self.model, video=self.model.video,
            range_normalizer=self.range_normalizer, dst_range=(-1, 1),
            use_raw_features=False, batch_size=batch_size)

    # ---- Cosine Similarity -------------------------------------------------

    def compute_trajectory_cos_sims(self, trajectories, query_points):
        N, T = trajectories.shape[:2]
        trajectories_normalized = self.model.normalize_points_for_sampling(trajectories)
        refined_features_at_trajectories = self.model.sample_embeddings(
            self.model.refined_features, trajectories_normalized.view(-1, 3))
        refined_features_at_trajectories = refined_features_at_trajectories.view(N, T, -1)
        query_frames = query_points[:, 2].long()
        refined_features_at_query_frames = refined_features_at_trajectories[
            torch.arange(N).to(self.model.device), query_frames]
        return torch.nn.functional.cosine_similarity(
            refined_features_at_query_frames.unsqueeze(1), refined_features_at_trajectories, dim=-1)

    # ---- Anchor Trajectories -----------------------------------------------

    def _get_model_preds_at_anchors(self, model, range_normalizer, preds, anchor_indices, batch_size=None):
        batch_size = batch_size if batch_size is not None else preds.shape[0]
        cycle_coords = []
        for vis_frame in anchor_indices:
            coords = []
            for i in range(0, preds.shape[0], batch_size):
                end_idx = min(i + batch_size, preds.shape[0])
                frames_set_t = torch.arange(i, end_idx, device=model.device)
                frames_set_t = torch.cat(
                    [torch.tensor([vis_frame], device=model.device), frames_set_t]).int()
                source_frame_indices = torch.arange(1, frames_set_t.shape[0], device=model.device)
                target_frame_indices = torch.tensor(
                    [0] * (frames_set_t.shape[0] - 1), device=model.device)
                inp = preds[i:end_idx], source_frame_indices, target_frame_indices, frames_set_t
                batch_coords = model(inp)
                batch_coords = range_normalizer.unnormalize(batch_coords, src=(-1, 1), dims=[0, 1])
                coords.append(batch_coords)
            coords = torch.cat(coords)
            cycle_coords.append(coords[:, :2])
        return torch.stack(cycle_coords)

    def compute_anchor_trajectories(self, trajectories, cos_sims, batch_size=None):
        N, T = trajectories.shape[:2]
        eql_anchor_cyc_predictions = {}
        for qp_idx in tqdm(range(N), desc="Iterating over query points"):
            preds = trajectories[qp_idx]
            anchor_frames = torch.arange(T).to(self.model.device)[
                cos_sims[qp_idx] >= self.anchor_cosine_similarity_threshold]
            eql_anchor_cyc_predictions[qp_idx] = self._get_model_preds_at_anchors(
                self.model, self.range_normalizer, preds=preds,
                anchor_indices=anchor_frames, batch_size=batch_size)
        return eql_anchor_cyc_predictions

    # ---- Occlusion ---------------------------------------------------------

    def compute_occ_pred_for_qp(self, green_trajectories_qp, source_trajectories_qp,
                                traj_cos_sim_qp, anch_sim_th, cos_sim_th):
        visible_at_st_frame_qp = traj_cos_sim_qp >= anch_sim_th
        dists_from_source = torch.norm(
            green_trajectories_qp - source_trajectories_qp[visible_at_st_frame_qp, :].unsqueeze(1), dim=-1)
        anchor_median_errors = torch.median(
            dists_from_source[:, visible_at_st_frame_qp], dim=0).values
        median_anchor_dist_th = anchor_median_errors.max()
        median_dists = torch.median(dists_from_source, dim=0).values
        return (median_dists > median_anchor_dist_th) | (traj_cos_sim_qp < cos_sim_th)

    def compute_occlusion(self, trajectories, trajs_cos_sims, anchor_trajectories):
        N = trajectories.shape[0]
        occ_preds = []
        for qp_idx in range(N):
            occ_preds.append(self.compute_occ_pred_for_qp(
                anchor_trajectories[qp_idx],
                trajectories[qp_idx, :, :2],
                trajs_cos_sims[qp_idx],
                self.anchor_cosine_similarity_threshold,
                self.cosine_similarity_threshold))
        return torch.stack(occ_preds)

    # ---- Full inference ----------------------------------------------------

    @torch.no_grad()
    def infer(self, query_points, batch_size=None):
        trajs = self.compute_trajectories(query_points, batch_size)
        cos_sims = self.compute_trajectory_cos_sims(trajs, query_points)
        anchor_trajs = self.compute_anchor_trajectories(trajs, cos_sims, batch_size)
        occ = self.compute_occlusion(trajs, cos_sims, anchor_trajs)
        return trajs[..., :2], occ

    @torch.no_grad()
    def infer_flow(self, query_points, batch_size=None):
        return self.compute_trajectories(query_points, batch_size)

    # ---- Vectorized tensor flow (foreground mask) --------------------------

    @torch.no_grad()
    def compute_framewise_flow_tensor(self, masks, use_raw_features=False,
                                     batch_size=None, interval=1) -> list:
        """Forward flow for foreground pixels. Returns list of [H, W, 2] tensors."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []

        for t in range(T - interval):
            print(f'Computing fwd semantic flow ... frame {t:05d}/{T - interval - 1}')
            current_mask = masks[t]
            query_positions = torch.nonzero(current_mask, as_tuple=False)  # [N, 2] (y, x)

            if query_positions.size(0) == 0:
                flows.append(torch.zeros(H, W, 2, device=video.device, dtype=torch.int32))
                continue

            N = query_positions.size(0)
            source_points = torch.stack([
                query_positions[:, 1].float(),
                query_positions[:, 0].float(),
                torch.full((N,), float(t), device=video.device),
            ], dim=1)  # [N, 3]

            frames_set_t = torch.tensor([t, t + interval], device=video.device, dtype=torch.int)
            src_idx = torch.zeros(N, dtype=torch.long, device=video.device)
            tgt_idx = torch.ones(N, dtype=torch.long, device=video.device)

            bs = N if batch_size is None else batch_size
            pred_list = []
            for s in range(0, N, bs):
                e = min(s + bs, N)
                pred_norm = self.model(
                    (source_points[s:e], src_idx[s:e], tgt_idx[s:e], frames_set_t),
                    use_raw_features=use_raw_features)
                pred_list.append(self.range_normalizer.unnormalize(pred_norm, dims=[0, 1], src=(-1, 1)))
            pred_coords = torch.cat(pred_list, dim=0)  # [N, 2]

            dx = pred_coords[:, 0] - source_points[:, 0]
            dy = pred_coords[:, 1] - source_points[:, 1]
            flow_xy = torch.stack([dx, dy], dim=1)

            flow_map = torch.zeros(H, W, 2, device=video.device, dtype=flow_xy.dtype)
            flow_map[query_positions[:, 0].long(), query_positions[:, 1].long()] = flow_xy
            flows.append(flow_map.round().to(torch.int32))

        return flows

    @torch.no_grad()
    def compute_framewise_flow_tensor_bwd(self, masks, use_raw_features=False,
                                         batch_size=None, interval=1) -> list:
        """Backward flow for foreground pixels. Returns list of [H, W, 2] tensors."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []

        for t in range(interval, T):
            print(f'Computing bwd semantic flow ... frame {t:05d}/{T - 1}')
            current_mask = masks[t]
            query_positions = torch.nonzero(current_mask, as_tuple=False)  # [N, 2] (y, x)

            if query_positions.size(0) == 0:
                flows.append(torch.zeros(H, W, 2, device=video.device, dtype=torch.int32))
                continue

            N = query_positions.size(0)
            source_points = torch.stack([
                query_positions[:, 1].float(),
                query_positions[:, 0].float(),
                torch.full((N,), float(t), device=video.device),
            ], dim=1)

            frames_set_t = torch.tensor([t, t - interval], device=video.device, dtype=torch.int)
            src_idx = torch.zeros(N, dtype=torch.long, device=video.device)
            tgt_idx = torch.ones(N, dtype=torch.long, device=video.device)

            bs = N if batch_size is None else batch_size
            pred_list = []
            for s in range(0, N, bs):
                e = min(s + bs, N)
                pred_norm = self.model(
                    (source_points[s:e], src_idx[s:e], tgt_idx[s:e], frames_set_t),
                    use_raw_features=use_raw_features)
                pred_list.append(self.range_normalizer.unnormalize(pred_norm, dims=[0, 1], src=(-1, 1)))
            pred_coords = torch.cat(pred_list, dim=0)

            dx = pred_coords[:, 0] - source_points[:, 0]
            dy = pred_coords[:, 1] - source_points[:, 1]
            flow_xy = torch.stack([dx, dy], dim=1)

            flow_map = torch.zeros(H, W, 2, device=video.device, dtype=flow_xy.dtype)
            flow_map[query_positions[:, 0].long(), query_positions[:, 1].long()] = flow_xy
            flows.append(flow_map.round().to(torch.int32))

        return flows

    # ---- All-pixel tensor flow ---------------------------------------------

    @torch.no_grad()
    def compute_framewise_flow_tensor_all_pixels(self, use_raw_features=False,
                                                 batch_size=None, interval=1) -> list:
        """Forward flow for every pixel. Returns list of [H, W, 2] tensors."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=video.device),
            torch.arange(W, device=video.device), indexing='ij')
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        num_pixels = x_flat.shape[0]

        for t in range(T - interval):
            print(f'Computing fwd all-pixel flow ... frame {t:05d}/{T - interval - 1}')
            source_points = torch.stack([
                x_flat.float(), y_flat.float(),
                torch.full_like(x_flat, float(t))], dim=1)

            frames_set_t = torch.tensor([t, t + interval], device=video.device, dtype=torch.int)
            src_idx = torch.zeros(num_pixels, dtype=torch.long, device=video.device)
            tgt_idx = torch.ones(num_pixels, dtype=torch.long, device=video.device)

            bs = num_pixels if batch_size is None else batch_size
            pred_list = []
            for s in range(0, num_pixels, bs):
                e = min(s + bs, num_pixels)
                pred_norm = self.model(
                    (source_points[s:e], src_idx[s:e], tgt_idx[s:e], frames_set_t),
                    use_raw_features=use_raw_features)
                pred_list.append(self.range_normalizer.unnormalize(pred_norm, dims=[0, 1], src=(-1, 1)))
            pred_coords = torch.cat(pred_list, dim=0)

            dx = pred_coords[:, 0] - x_flat.float()
            dy = pred_coords[:, 1] - y_flat.float()
            flow_map = torch.stack([dx, dy], dim=1).view(H, W, 2)
            flows.append(flow_map.round().to(torch.int32))

        return flows

    @torch.no_grad()
    def compute_framewise_flow_tensor_all_pixels_bwd(self, use_raw_features=False,
                                                     batch_size=None, interval=1) -> list:
        """Backward flow for every pixel. Returns list of [H, W, 2] tensors."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=video.device),
            torch.arange(W, device=video.device), indexing='ij')
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        num_pixels = x_flat.shape[0]

        for t in range(interval, T):
            print(f'Computing bwd all-pixel flow ... frame {t:05d}/{T - 1}')
            source_points = torch.stack([
                x_flat.float(), y_flat.float(),
                torch.full_like(x_flat, float(t))], dim=1)

            frames_set_t = torch.tensor([t, t - interval], device=video.device, dtype=torch.int)
            src_idx = torch.zeros(num_pixels, dtype=torch.long, device=video.device)
            tgt_idx = torch.ones(num_pixels, dtype=torch.long, device=video.device)

            bs = num_pixels if batch_size is None else batch_size
            pred_list = []
            for s in range(0, num_pixels, bs):
                e = min(s + bs, num_pixels)
                pred_norm = self.model(
                    (source_points[s:e], src_idx[s:e], tgt_idx[s:e], frames_set_t),
                    use_raw_features=use_raw_features)
                pred_list.append(self.range_normalizer.unnormalize(pred_norm, dims=[0, 1], src=(-1, 1)))
            pred_coords = torch.cat(pred_list, dim=0)

            dx = pred_coords[:, 0] - x_flat.float()
            dy = pred_coords[:, 1] - y_flat.float()
            flow_map = torch.stack([dx, dy], dim=1).view(H, W, 2)
            flows.append(flow_map.round().to(torch.int32))

        return flows

    # ---- Legacy point-by-point flow (slow) ---------------------------------

    @torch.no_grad()
    def compute_framewise_flow(self, masks, use_raw_features=False, batch_size=None) -> list:
        """Forward flow, point-by-point (slow). Prefer compute_framewise_flow_tensor."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []
        for t in range(T - 1):
            print(f'Computing fwd flow (legacy) ... frame {t:05d}')
            current_mask = masks[t]
            query_positions = torch.nonzero(current_mask, as_tuple=False)
            if query_positions.size(0) == 0:
                flows.append(torch.zeros(H, W, 2, device=video.device, dtype=torch.int32))
                continue

            query_points_t = torch.stack([
                query_positions[:, 1].float(),
                query_positions[:, 0].float(),
                torch.full((query_positions.size(0),), float(t), device=video.device),
            ], dim=1)

            flow_trajectories = []
            for qp in query_points_t:
                traj = generate_flow_trajectory(
                    query_point=qp, video=video, model=self.model,
                    range_normalizer=self.range_normalizer,
                    start_t=t, end_t=t + 2, batch_size=batch_size)
                flow_trajectories.append(traj)

            flow_trajectories = torch.stack(flow_trajectories, dim=0)  # [N, 2, 3]
            flow_xy = flow_trajectories[:, 1, :2] - flow_trajectories[:, 0, :2]
            flow_map = torch.zeros(H, W, 2, device=video.device, dtype=flow_xy.dtype)
            flow_map[query_positions[:, 0].long(), query_positions[:, 1].long()] = flow_xy
            flows.append(flow_map.round().to(torch.int32))
        return flows

    @torch.no_grad()
    def compute_framewise_flow_bwd(self, masks, use_raw_features=False, batch_size=None) -> list:
        """Backward flow, point-by-point (slow). Prefer compute_framewise_flow_tensor_bwd."""
        video = self.model.video
        T, C, H, W = video.shape
        flows = []
        for t in range(1, T):
            print(f'Computing bwd flow (legacy) ... frame {t:05d}')
            current_mask = masks[t]
            query_positions = torch.nonzero(current_mask, as_tuple=False)
            if query_positions.size(0) == 0:
                flows.append(torch.zeros(H, W, 2, device=video.device, dtype=torch.int32))
                continue

            query_points_t = torch.stack([
                query_positions[:, 1].float(),
                query_positions[:, 0].float(),
                torch.full((query_positions.size(0),), float(t), device=video.device),
            ], dim=1)

            flow_trajectories = []
            for qp in query_points_t:
                traj = generate_flow_trajectory(
                    query_point=qp, video=video, model=self.model,
                    range_normalizer=self.range_normalizer,
                    start_t=t - 1, end_t=t + 1, batch_size=batch_size)
                flow_trajectories.append(traj)

            flow_trajectories = torch.stack(flow_trajectories, dim=0)
            flow_xy = flow_trajectories[:, 0, :2] - flow_trajectories[:, 1, :2]
            flow_map = torch.zeros(H, W, 2, device=video.device, dtype=flow_xy.dtype)
            flow_map[query_positions[:, 0].long(), query_positions[:, 1].long()] = flow_xy
            flows.append(flow_map.round().to(torch.int32))
        return flows
