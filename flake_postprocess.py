from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from maskterial.measurements import attach_measurements_to_flake
from maskterial.structures.FlakeClass import Flake


@dataclass(slots=True)
class PostprocessParams:
    # 合并重叠较大的预测：
    # 两个 mask 的 IoU 大于该值时，认为是重复/重叠预测并做 union 合并。
    overlap_iou_threshold: float = 0.5
    # 小 mask 被大 mask 覆盖的比例大于该值时，也认为是重复预测并合并。
    overlap_containment_threshold: float = 0.8

    # 对于大块被切成小块的情况把小块拼回成大块：
    # 是否启用 Lab 颜色约束的近邻 bridge merge；默认关闭，避免相邻 flake 被误粘连。
    enable_bridge_merge: bool = False
    # 颜色约束外扩的半径，单位是像素；值越大，可检测到的断裂间隙越宽。
    grow_radius_px: int = 3
    # 两个 mask 边界距离不超过该值时，才会进入 bridge merge 候选。
    max_boundary_distance_px: int = 3
    # Lab 颜色距离中 L 亮度通道的权重；越小越不受明暗变化影响。
    lab_l_weight: float = 0.5
    # 像素是否像 flake 的软阈值中心；D(Lab(pixel), mask_median) 等于该值时 membership 为 0.5。
    tau_grow: float = 12.0
    # 两个 mask 自身 Lab median 的最大允许距离；超过该值即使距离近也不合并。
    tau_pair: float = 12.0
    # sigmoid 过渡宽度；越大越平缓，越小越接近硬阈值。
    grow_sigma: float = 1.5
    # gap 区域内 soft bridge weight 的最小总量；防止一两个像素触发合并。
    min_bridge_area_px: float = 15.0
    # soft bridge weight 总量 / gap 面积的最小比例；衡量 gap 中有多少像 flake。
    min_bridge_ratio: float = 0.3
    # 所有合并完成后，再删除面积小于该值的 flake，单位 um^2。
    final_min_area_um2: float = 100.0
    # 所有合并完成后，再删除最终 score 小于该值的 flake。
    final_min_score: float = 0.015
    # 所有合并完成后，再删除形状复杂度超过该值的 flake；值越大表示越细长/破碎。
    final_max_shape_complexity: float = 5.0
    # bridge merge 的最大迭代轮数；每轮合并后会重新计算 mask 和 Lab median。
    max_bridge_passes: int = 5


@dataclass(slots=True)
class PostprocessResult:
    # 模型直接输出的原始 flake。
    raw_flakes: list[Flake]
    # 高重叠 union merge 之后的 flake。
    overlap_merged_flakes: list[Flake]
    # Lab 颜色约束 bridge merge 之后的 flake。
    bridge_merged_flakes: list[Flake]
    # 最后做 area 过滤之后的最终 flake。
    final_flakes: list[Flake]


@dataclass(slots=True)
class _Candidate:
    mask: np.ndarray
    flake: Flake
    score: float
    thickness: str
    mean_contrast: np.ndarray
    entropy: float
    lab_median: np.ndarray


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root

    def groups(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = {}
        for index in range(len(self.parent)):
            grouped.setdefault(self.find(index), []).append(index)
        return list(grouped.values())


class GrapheneFlakePostprocessor:
    def __init__(self, params: PostprocessParams | None = None):
        self.params = params or PostprocessParams()
        self._lab_image: np.ndarray | None = None

    def run(
        self,
        image_bgr: np.ndarray,
        raw_flakes: list[Flake],
        debug_vis_dir: str | None = None,
        image_stem: str | None = None,
    ) -> PostprocessResult:
        self._lab_image = self._to_lab(image_bgr)
        raw_candidates = self._to_candidates(raw_flakes)

        overlap_candidates = self._merge_overlaps(raw_candidates)
        if self.params.enable_bridge_merge:
            bridge_candidates = self._merge_bridges(overlap_candidates)
        else:
            bridge_candidates = list(overlap_candidates)
        final_candidates = [
            candidate
            for candidate in bridge_candidates
            if candidate.score >= self.params.final_min_score
            and candidate.flake.measurements.area_um2 >= self.params.final_min_area_um2
            and self._shape_complexity(candidate.mask)
            <= self.params.final_max_shape_complexity
        ]

        result = PostprocessResult(
            raw_flakes=list(raw_flakes),
            overlap_merged_flakes=[candidate.flake for candidate in overlap_candidates],
            bridge_merged_flakes=[candidate.flake for candidate in bridge_candidates],
            final_flakes=[candidate.flake for candidate in final_candidates],
        )

        if debug_vis_dir is not None:
            self.write_debug_visualizations(image_bgr, result, debug_vis_dir, image_stem)

        return result

    def write_debug_visualizations(
        self,
        image_bgr: np.ndarray,
        result: PostprocessResult,
        debug_vis_dir: str,
        image_stem: str | None = None,
    ):
        os.makedirs(debug_vis_dir, exist_ok=True)
        stem = image_stem or "image"
        stages = [
            ("raw_pred", result.raw_flakes),
            ("overlap_merged", result.overlap_merged_flakes),
            ("bridge_merged", result.bridge_merged_flakes),
            ("final", result.final_flakes),
        ]
        for stage_name, flakes in stages:
            vis = self.draw_flakes(image_bgr, flakes)
            cv2.imwrite(os.path.join(debug_vis_dir, f"{stem}_{stage_name}.jpg"), vis)

    def draw_flakes(self, image_bgr: np.ndarray, flakes: list[Flake]) -> np.ndarray:
        output = image_bgr.copy()
        overlay = image_bgr.copy()
        for index, flake in enumerate(flakes, start=1):
            mask = self._as_mask(flake.mask)
            color = self._color_for_index(index)
            overlay[mask.astype(bool)] = color
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
            self._draw_label(output, flake, index, color)
        return cv2.addWeighted(overlay, 0.25, output, 0.75, 0.0)

    def _to_lab(self, image_bgr: np.ndarray) -> np.ndarray:
        image_float = image_bgr.astype(np.float32) / 255.0
        return cv2.cvtColor(image_float, cv2.COLOR_BGR2LAB)

    def _to_candidates(self, flakes: list[Flake]) -> list[_Candidate]:
        candidates = []
        for flake in flakes:
            candidate = self._candidate_from_flake(flake)
            if candidate is not None:
                candidates.append(candidate)
        return self._sort_candidates(candidates)

    def _candidate_from_flake(self, flake: Flake) -> _Candidate | None:
        score = self._flake_score(flake)
        template = _Candidate(
            mask=self._as_mask(flake.mask),
            flake=flake,
            score=score,
            thickness=str(flake.thickness),
            mean_contrast=np.asarray(flake.mean_contrast),
            entropy=float(flake.entropy),
            lab_median=np.zeros(3, dtype=np.float32),
        )
        return self._build_candidate(template.mask, template)

    def _build_candidate(
        self,
        mask: np.ndarray,
        template: _Candidate,
    ) -> _Candidate | None:
        clean_mask = self._as_mask(mask)
        if cv2.countNonZero(clean_mask) == 0:
            return None

        flake = self._build_flake(clean_mask, template)
        lab_median = self._mask_lab_median(clean_mask)
        return _Candidate(
            mask=clean_mask,
            flake=flake,
            score=template.score,
            thickness=template.thickness,
            mean_contrast=np.asarray(template.mean_contrast),
            entropy=template.entropy,
            lab_median=lab_median,
        )

    def _build_flake(self, mask: np.ndarray, template: _Candidate) -> Flake:
        mask = self._as_mask(mask)
        ys, xs = np.nonzero(mask)
        area_px = int(len(xs))

        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] > 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))

        points = np.column_stack((xs, ys)).astype(np.float32)
        if len(points) >= 2:
            rect = cv2.minAreaRect(points)
            max_sidelength = float(max(rect[1]))
            min_sidelength = float(min(rect[1]))
        else:
            max_sidelength = 1.0
            min_sidelength = 1.0

        flake = Flake(
            mask=mask,
            thickness=str(template.thickness),
            size=area_px,
            mean_contrast=np.asarray(template.mean_contrast),
            center=(center_x, center_y),
            max_sidelength=max_sidelength,
            min_sidelength=min_sidelength,
            false_positive_probability=1.0 - float(template.score),
            entropy=float(template.entropy),
        )
        return attach_measurements_to_flake(flake)

    def _merge_overlaps(self, candidates: list[_Candidate]) -> list[_Candidate]:
        current = list(candidates)
        while True:
            merged, changed = self._merge_overlap_once(current)
            current = merged
            if not changed:
                return self._sort_candidates(current)

    def _merge_overlap_once(
        self,
        candidates: list[_Candidate],
    ) -> tuple[list[_Candidate], bool]:
        if len(candidates) <= 1:
            return candidates, False

        dsu = _DisjointSet(len(candidates))
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                if self._should_overlap_merge(candidates[first], candidates[second]):
                    dsu.union(first, second)

        groups = dsu.groups()
        if len(groups) == len(candidates):
            return candidates, False

        return self._merge_groups(candidates, groups), True

    def _should_overlap_merge(
        self,
        first: _Candidate,
        second: _Candidate,
    ) -> bool:
        intersection = cv2.countNonZero((first.mask & second.mask).astype(np.uint8))
        if intersection == 0:
            return False

        first_area = cv2.countNonZero(first.mask)
        second_area = cv2.countNonZero(second.mask)
        union = first_area + second_area - intersection
        iou = intersection / max(union, 1)
        containment = intersection / max(min(first_area, second_area), 1)
        return (
            iou >= self.params.overlap_iou_threshold
            or containment >= self.params.overlap_containment_threshold
        )

    def _merge_bridges(self, candidates: list[_Candidate]) -> list[_Candidate]:
        current = list(candidates)
        for _ in range(self.params.max_bridge_passes):
            merged, changed = self._merge_bridge_once(current)
            current = merged
            if not changed:
                break
        return self._sort_candidates(current)

    def _merge_bridge_once(
        self,
        candidates: list[_Candidate],
    ) -> tuple[list[_Candidate], bool]:
        if len(candidates) <= 1:
            return candidates, False

        dsu = _DisjointSet(len(candidates))
        bridge_masks: dict[tuple[int, int], np.ndarray] = {}
        for first in range(len(candidates)):
            for second in range(first + 1, len(candidates)):
                should_merge, bridge_mask = self._should_bridge_merge(
                    candidates[first],
                    candidates[second],
                )
                if should_merge:
                    dsu.union(first, second)
                    bridge_masks[(first, second)] = bridge_mask

        groups = dsu.groups()
        if len(groups) == len(candidates):
            return candidates, False

        return self._merge_groups(candidates, groups, bridge_masks), True

    def _should_bridge_merge(
        self,
        first: _Candidate,
        second: _Candidate,
    ) -> tuple[bool, np.ndarray]:
        empty_bridge = np.zeros_like(first.mask, dtype=np.uint8)
        if not self._is_near(first.mask, second.mask, self.params.max_boundary_distance_px):
            return False, empty_bridge

        pair_color_dist = self._lab_distance(first.lab_median, second.lab_median)
        if pair_color_dist > self.params.tau_pair:
            return False, empty_bridge

        first_dilated = self._dilate(first.mask, self.params.grow_radius_px)
        second_dilated = self._dilate(second.mask, self.params.grow_radius_px)
        gap = (first_dilated.astype(bool) & second_dilated.astype(bool)) & ~(
            first.mask.astype(bool) | second.mask.astype(bool)
        )
        gap = gap.astype(np.uint8)
        gap_area = cv2.countNonZero(gap)
        if gap_area == 0:
            return False, empty_bridge

        gap_pixels = self._lab_image[gap.astype(bool)]
        first_dist = self._lab_distance(gap_pixels, first.lab_median)
        second_dist = self._lab_distance(gap_pixels, second.lab_median)
        grow_sigma = max(self.params.grow_sigma, 1e-6)
        first_membership = self._sigmoid(
            (self.params.tau_grow - first_dist) / grow_sigma
        )
        second_membership = self._sigmoid(
            (self.params.tau_grow - second_dist) / grow_sigma
        )
        bridge_weights = np.minimum(first_membership, second_membership)
        soft_bridge_area = float(np.sum(bridge_weights))
        soft_bridge_ratio = soft_bridge_area / gap_area

        bridge_mask = np.zeros_like(gap, dtype=np.uint8)
        bridge_mask[gap.astype(bool)] = (bridge_weights >= 0.5).astype(np.uint8)
        hard_bridge_area = cv2.countNonZero(bridge_mask)

        should_merge = (
            soft_bridge_area >= self.params.min_bridge_area_px
            and soft_bridge_ratio >= self.params.min_bridge_ratio
            and hard_bridge_area > 0
        )
        return should_merge, bridge_mask

    def _merge_groups(
        self,
        candidates: list[_Candidate],
        groups: list[list[int]],
        bridge_masks: dict[tuple[int, int], np.ndarray] | None = None,
    ) -> list[_Candidate]:
        merged_candidates = []
        bridge_masks = bridge_masks or {}
        for group in groups:
            if len(group) == 1:
                merged_candidates.append(candidates[group[0]])
                continue

            template = max((candidates[index] for index in group), key=lambda item: item.score)
            merged_mask = np.zeros_like(candidates[group[0]].mask, dtype=np.uint8)
            for index in group:
                merged_mask |= candidates[index].mask

            group_set = set(group)
            for (first, second), bridge_mask in bridge_masks.items():
                if first in group_set and second in group_set:
                    merged_mask |= bridge_mask

            merged_candidate = self._build_candidate(merged_mask, template)
            if merged_candidate is not None:
                merged_candidates.append(merged_candidate)

        return self._sort_candidates(merged_candidates)

    def _mask_lab_median(self, mask: np.ndarray) -> np.ndarray:
        pixels = self._lab_image[mask.astype(bool)]
        if len(pixels) == 0:
            return np.zeros(3, dtype=np.float32)
        return np.median(pixels, axis=0).astype(np.float32)

    def _lab_distance(self, first: np.ndarray, second: np.ndarray) -> np.ndarray | float:
        diff = np.asarray(first, dtype=np.float32) - np.asarray(second, dtype=np.float32)
        diff = diff.copy()
        diff[..., 0] *= self.params.lab_l_weight
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        if np.ndim(distances) == 0:
            return float(distances)
        return distances

    def _is_near(self, first_mask: np.ndarray, second_mask: np.ndarray, radius: int) -> bool:
        return bool(np.any(self._dilate(first_mask, radius) & second_mask))

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return self._as_mask(mask)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        return cv2.dilate(self._as_mask(mask), kernel, iterations=1).astype(np.uint8)

    def _as_mask(self, mask: np.ndarray) -> np.ndarray:
        return (np.asarray(mask) > 0).astype(np.uint8)

    def _sort_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _flake_score(self, flake: Flake) -> float:
        return max(0.0, min(1.0, 1.0 - float(flake.false_positive_probability)))

    def _shape_complexity(self, mask: np.ndarray) -> float:
        mask = self._as_mask(mask)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter_px = sum(cv2.arcLength(contour, True) for contour in contours)
        area_px = max(cv2.countNonZero(mask), 1)
        return float((perimeter_px * perimeter_px) / (4.0 * np.pi * area_px))

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-values))

    def _color_for_index(self, index: int) -> tuple[int, int, int]:
        palette = [
            (0, 255, 255),
            (0, 128, 255),
            (0, 255, 0),
            (255, 128, 0),
            (255, 0, 255),
            (255, 255, 0),
            (0, 0, 255),
            (128, 255, 0),
            (255, 0, 128),
            (128, 0, 255),
        ]
        return palette[(index - 1) % len(palette)]

    def _draw_label(
        self,
        image: np.ndarray,
        flake: Flake,
        index: int,
        color: tuple[int, int, int],
    ):
        area_um2 = 0.0 if flake.measurements is None else flake.measurements.area_um2
        score = self._flake_score(flake)
        text = f"#{index} s={score:.2f} a={area_um2:.1f}"
        center_x = int(flake.center[0])
        center_y = int(flake.center[1])
        origin = (center_x + 8, center_y - 8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size
        image_h, image_w = image.shape[:2]
        x = min(max(origin[0], 0), max(0, image_w - text_w - 8))
        y = min(max(origin[1], text_h + 8), max(text_h + 8, image_h - baseline - 4))

        cv2.rectangle(
            image,
            (x - 3, y - text_h - 6),
            (x + text_w + 6, y + baseline + 3),
            (0, 0, 0),
            -1,
        )
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def postprocess_flakes(
    image_bgr: np.ndarray,
    raw_flakes: list[Flake],
    params: PostprocessParams | None = None,
    debug_vis_dir: str | None = None,
    image_stem: str | None = None,
) -> PostprocessResult:
    return GrapheneFlakePostprocessor(params).run(
        image_bgr=image_bgr,
        raw_flakes=raw_flakes,
        debug_vis_dir=debug_vis_dir,
        image_stem=image_stem,
    )
