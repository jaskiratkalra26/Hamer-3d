import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HaMeR video processing code')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--out_video', type=str, default='output.mp4', help='Path to output video')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'])
    parser.add_argument('--max_seconds', type=int, default=0, help='Max seconds of video to process')
    parser.add_argument('--det_stride', type=int, default=3, help='Run body detector every N frames for speed')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load HaMeR model
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load Body Detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening video: {args.video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    max_frames = total_frames
    if args.max_seconds > 0:
        max_frames = min(total_frames, int(fps * args.max_seconds))
    
    print(f"Processing {max_frames} frames with detector stride={args.det_stride}...")
    
    last_pred_bboxes = None
    last_pred_scores = None

    for frame_idx in tqdm(range(max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        img_cv2 = frame
        img = img_cv2.copy()[:, :, ::-1]

        # Run heavy ViTDet body detector every N frames for maximum speed
        if frame_idx % args.det_stride == 0 or last_pred_bboxes is None:
            det_out = detector(img_cv2)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
            if len(pred_bboxes) > 0:
                last_pred_bboxes = pred_bboxes
                last_pred_scores = pred_scores
        else:
            pred_bboxes = last_pred_bboxes
            pred_scores = last_pred_scores

        if pred_bboxes is None or len(pred_bboxes) == 0:
            out.write(frame)
            continue

        vitposes_out = cpm.predict_pose(img, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

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

        if len(bboxes) == 0:
            out.write(frame)
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        img_size_batch = None
        scaled_focal_length_val = None
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    out_hamer = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out_hamer['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            img_size_batch = img_size[0].detach().cpu().numpy().astype(int)
            scaled_focal_length_val = scaled_focal_length.item() if isinstance(scaled_focal_length, torch.Tensor) else float(scaled_focal_length)
            
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out_hamer['pred_vertices'][n].detach().cpu().numpy()
                is_right_n = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right_n-1)*verts[:,0]
                all_verts.append(verts)
                all_cam_t.append(pred_cam_t_full[n])
                all_right.append(is_right_n)

        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length_val,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=[width, height], is_right=all_right, **misc_args)

            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            final_frame = (255*input_img_overlay[:, :, ::-1]).astype(np.uint8)
            out.write(final_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {args.out_video}")

if __name__ == '__main__':
    main()
