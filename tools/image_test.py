import mmcv
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import Visualizer
import PIL
import numpy as np


def get_colormap(num_colors: int):
    return (np.resize(colormap(), (num_colors, 3))).tolist()


def get_result(model, img, result, is_one_stage, num_rel):
    img = mmcv.imread(img)
    img = img.copy()  # (H, W, 3)
    img_h, img_w = img.shape[:-1]

    # Decrease contrast
    img = PIL.Image.fromarray(img)
    converter = PIL.ImageEnhance.Color(img)
    img = converter.enhance(0.01)

    # Draw masks
    pan_results = result.pan_results

    ids = np.unique(pan_results)[::-1]
    num_classes = 133
    legal_indices = (ids != num_classes)  # for VOID label
    ids = ids[legal_indices]

    # Get predicted labels
    labels = np.array([id % INSTANCE_OFFSET for id in ids], dtype=np.int64)
    labels = [model.CLASSES[l] for l in labels]

    # For psgtr
    rel_obj_labels = result.labels
    rel_obj_labels = [model.CLASSES[l - 1] for l in rel_obj_labels]

    # (N_m, H, W)
    segms = pan_results[None] == ids[:, None, None]
    # Resize predicted masks
    segms = [
        mmcv.image.imresize(m.astype(float), (img_w, img_h)) for m in segms
    ]
    # One stage segmentation
    masks = result.masks

    # Choose colors for each instance in coco
    colormap_coco = get_colormap(
        len(masks)) if is_one_stage else get_colormap(len(segms))
    colormap_coco = (np.array(colormap_coco) / 255).tolist()

    # Viualize masks
    viz = Visualizer(img)
    viz.overlay_instances(
        labels=rel_obj_labels if is_one_stage else labels,
        masks=masks if is_one_stage else segms,
        assigned_colors=colormap_coco,
    )
    viz_img = viz.get_output().get_image()

    # Exclude background class
    rel_dists = result.rel_dists[:, 1:]
    # rel_dists = result.rel_dists
    # Filter out relations
    if len(rel_dists) < num_rel:
        n_rel_topk = len(rel_dists)
    else:
        n_rel_topk = num_rel
    rel_scores = rel_dists.max(1)
    # rel_scores = result.triplet_scores
    # Extract relations with top scores
    rel_topk_idx = np.argpartition(rel_scores, -n_rel_topk)[-n_rel_topk:]
    rel_labels_topk = rel_dists[rel_topk_idx].argmax(1)
    rel_pair_idxes_topk = result.rel_pair_idxes[rel_topk_idx]
    relations = np.concatenate(
        [rel_pair_idxes_topk, rel_labels_topk[..., None]], axis=1)
    return viz_img, rel_obj_labels, relations
