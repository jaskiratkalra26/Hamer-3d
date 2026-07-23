import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Auto-resolve repository root so hamer and vitpose_model modules are always found
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

class HandTracker3D:
    """
    Modular 3D Hand Reconstruction Pipeline using HaMeR, ViTPose, and RegNetY/ViTDet.
    Designed for easy integration into any third-party Python application, ROS node, or API.
    """
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        body_detector: str = 'regnety',
        rescale_factor: float = 2.0,
        det_stride: int = 5,
        scale_inference: float = 0.5,
        device: Optional[str] = None
    ):
        self.rescale_factor = rescale_factor
        self.det_stride = det_stride
        self.scale_inference = scale_inference
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        # 1. Load HaMeR model
        download_models(CACHE_DIR_HAMER)
        self.model, self.model_cfg = load_hamer(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. Load Body Detector
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # 3. Keypoint detector & Renderer
        self.cpm = ViTPoseModel(self.device)
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

        # Tracking state
        self.frame_count = 0
        self.last_boxes = None
        self.last_right = None

    def reset(self):
        """Reset internal tracking state."""
        self.frame_count = 0
        self.last_boxes = None
        self.last_right = None

    def process_frame(self, img_cv2: np.ndarray) -> Dict[str, Any]:
        """
        Process a single BGR image frame and return 3D hand predictions in memory.

        Returns:
            dict containing:
                - 'hands': list of dicts, each with:
                    - 'verts_3d': (778, 3) numpy array of 3D vertices in meters
                    - 'cam_t': (3,) camera translation [tx, ty, tz]
                    - 'is_right': 1 for right hand, 0 for left hand
                    - 'box': bounding box [x1, y1, x2, y2]
                - 'scaled_focal_length': float
        """
        height, width = img_cv2.shape[:2]
        img = img_cv2.copy()[:, :, ::-1]

        # 1. Run Detector & ViTPose keypoints on stride frames or when cache empty
        if self.frame_count % self.det_stride == 0 or self.last_boxes is None:
            if self.scale_inference != 1.0:
                det_h = int(height * self.scale_inference)
                det_w = int(width * self.scale_inference)
                img_det = cv2.resize(img_cv2, (det_w, det_h))
            else:
                img_det = img_cv2

            det_out = self.detector(img_det)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()

            if self.scale_inference != 1.0 and len(pred_bboxes) > 0:
                pred_bboxes = pred_bboxes / self.scale_inference

            if len(pred_bboxes) > 0:
                vitposes_out = self.cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

                bboxes = []
                is_right = []

                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                    right_hand_keyp = vitposes['keypoints'][-21:]

                    keyp = left_hand_keyp
                    valid = keyp[:, 2] > 0.5
                    if sum(valid) > 3:
                        bboxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()])
                        is_right.append(0)

                    keyp = right_hand_keyp
                    valid = keyp[:, 2] > 0.5
                    if sum(valid) > 3:
                        bboxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()])
                        is_right.append(1)

                if len(bboxes) > 0:
                    self.last_boxes = np.stack(bboxes)
                    self.last_right = np.stack(is_right)
                else:
                    self.last_boxes = None
                    self.last_right = None
            else:
                self.last_boxes = None
                self.last_right = None

        self.frame_count += 1

        if self.last_boxes is None or len(self.last_boxes) == 0:
            return {'hands': [], 'scaled_focal_length': 5000.0}

        boxes = self.last_boxes
        right = self.last_right

        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        hands = []
        scaled_focal_length_val = 5000.0

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    out_hamer = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out_hamer['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            scaled_focal_length_val = scaled_focal_length.item() if isinstance(scaled_focal_length, torch.Tensor) else float(scaled_focal_length)

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out_hamer['pred_vertices'][n].detach().cpu().numpy()
                is_right_n = int(batch['right'][n].cpu().numpy())
                verts[:, 0] = (2 * is_right_n - 1) * verts[:, 0]
                
                hands.append({
                    'verts_3d': verts,
                    'cam_t': pred_cam_t_full[n],
                    'is_right': is_right_n,
                    'box': boxes[n] if n < len(boxes) else None
                })

        return {
            'hands': hands,
            'scaled_focal_length': scaled_focal_length_val
        }

    def render_overlay(self, img_cv2: np.ndarray, predictions: Dict[str, Any]) -> np.ndarray:
        """Render 3D hand mesh overlay onto the input BGR image."""
        hands = predictions.get('hands', [])
        if not hands:
            return img_cv2

        height, width = img_cv2.shape[:2]
        all_verts = [h['verts_3d'] for h in hands]
        all_cam_t = [h['cam_t'] for h in hands]
        all_right = [h['is_right'] for h in hands]
        scaled_focal_length_val = predictions.get('scaled_focal_length', 5000.0)

        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length_val,
        )
        cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=[width, height], is_right=all_right, **misc_args)

        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        final_frame = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
        return final_frame
