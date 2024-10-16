import copy
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics import ConfusionMatrix, AveragePrecision
from sklearn.metrics import average_precision_score


# Function to derive data paths (train/val/test + fake/real) for dolos dataset
def derive_datapaths(opt):
    root_path = opt.data_root_path.rstrip('/') 

    opt.train_path = f"{root_path}/fake/{opt.train_dataset}/images/train"
    opt.valid_path = f"{root_path}/fake/{opt.train_dataset}/images/valid"
    opt.train_masks_ground_truth_path = f"{root_path}/fake/{opt.train_dataset}/masks/train"
    opt.valid_masks_ground_truth_path = f"{root_path}/fake/{opt.train_dataset}/masks/valid"
    opt.train_real_list_path = f"{root_path}/real/train"
    opt.valid_real_list_path = f"{root_path}/real/valid"

    return opt

# Localisation
# IOU
def compute_iou(pred_mask, gt_mask, threshold=0.5):
    if torch.is_tensor(pred_mask) and torch.is_tensor(gt_mask):
        pred_mask = (pred_mask > threshold).to(torch.uint8)
        gt_mask = (gt_mask > threshold).to(torch.uint8)

        intersection = torch.logical_and(pred_mask, gt_mask).float().sum()
        union = torch.logical_or(pred_mask, gt_mask).float().sum()

        iou_score = 100.0 * ((intersection + 1e-8) / (union + 1e-8)).item()
    else:
        pred_mask = (pred_mask > threshold).astype(np.uint8)
        gt_mask = (gt_mask > threshold).astype(np.uint8)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        iou_score = 100.0 * ((intersection + 1e-8) / (union + 1e-8))

    return iou_score

def compute_batch_iou(predictions, ground_truths, threshold=0.5):
    assert len(predictions) == len(ground_truths), "Both lists must have the same length"
    
    iou_scores = []
    for pred_mask, gt_mask in zip(predictions, ground_truths):
        iou_score = compute_iou(pred_mask, gt_mask, threshold)
        iou_scores.append(iou_score)
    
    return iou_scores

# F1 - inspired from Guillaro, Fabrizio et al. TruFor paper https://github.com/grip-unina/TruFor/blob/main/test_docker/metrics.py
# adapted for GPU use
def min_filter(tensor, size):
    padding = (size - 1) // 2
    tensor = F.pad(tensor, (padding, padding, padding, padding), value=float("inf"))
    tensor = tensor.unsqueeze(0) * -1
    result = F.max_pool2d(tensor, kernel_size=size, stride=1)
    return result.squeeze() * -1

def max_filter(tensor, size):
    padding = (size - 1) // 2
    tensor = F.pad(tensor, (padding, padding, padding, padding))
    tensor = tensor.unsqueeze(0)
    result = F.max_pool2d(tensor, kernel_size=size, stride=1)
    return result.squeeze()

def extract_ground_truths(gt, erode_size=15, dilate_size=11):
    gt_eroded = min_filter(gt, erode_size)
    gt_dilated = torch.logical_not(max_filter(gt, dilate_size))
    return gt_dilated, gt_eroded

def compute_f1(fp, tp, fn):
    return (2 * tp + 1e-32) / torch.maximum(2 * tp + fn + fp, torch.tensor(1e-32))

def dynamic_threshold_metrics(preds, gt_dilated, gt_eroded):
    preds, gt_dilated, gt_eroded = preds.flatten(), gt_dilated.flatten(), gt_eroded.flatten()
    inds = torch.argsort(preds)
    inds = inds[(gt_dilated[inds] + gt_eroded[inds]) > 0]
    thresholds = preds[inds]
    gt_dilated, gt_eroded = gt_dilated[inds], gt_eroded[inds]
    tn = torch.cumsum(gt_dilated, dim=0)
    fn = torch.cumsum(gt_eroded, dim=0)
    fp, tp = torch.sum(gt_dilated) - tn, torch.sum(gt_eroded) - fn
    mask = F.pad(thresholds[1:] > thresholds[:-1], (0, 1), mode="constant")
    return fp[mask], tp[mask], fn[mask], tn[mask]

def fixed_threshold_metrics(preds, gt, gt_dilated, gt_eroded, threshold):
    preds = (preds > threshold).flatten().int()
    gt, gt_dilated, gt_eroded = gt.flatten().int(), gt_dilated.flatten().int(), gt_eroded.flatten().int()
    gt, preds = gt[(gt_dilated + gt_eroded) > 0], preds[(gt_dilated + gt_eroded) > 0]
    cm = ConfusionMatrix(task="binary", num_classes=2).to(gt.device)(gt, preds)
    return cm[1, 0], cm[1, 1], cm[0, 1], cm[0, 0]

def localization_f1(pred, gt):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt, dtype=torch.float32)
    pred, gt = pred.float(), gt.float()
    gt_dilated, gt_eroded = extract_ground_truths(gt)

    # Best threshold F1
    try:
        fp, tp, fn, tn = dynamic_threshold_metrics(pred, gt_dilated, gt_eroded)
        f1_dynamic = compute_f1(fp, tp, fn)
        best_f1 = torch.max(f1_dynamic)
    except Exception as e:
        print(e)
        best_f1 = torch.tensor(np.nan)

    # Fixed threshold F1
    try:
        fp, tp, fn, tn = fixed_threshold_metrics(pred, gt, gt_dilated, gt_eroded, 0.5)
        f1_fixed = compute_f1(fp, tp, fn)
    except Exception as e:
        print(e)
        f1_fixed = torch.tensor(np.nan)

    return max(best_f1, f1_fixed), f1_fixed

def compute_batch_localization_f1(preds_list, gts_list):
    assert len(preds_list) == len(gts_list), "Both lists must have the same length"
    
    batch_f1_scores_best = []
    batch_f1_scores_fixed = []
    for preds, gt in zip(preds_list, gts_list):
        best_f1, fixed_f1 = localization_f1(preds, gt)
        batch_f1_scores_best.append(best_f1.item())
        batch_f1_scores_fixed.append(fixed_f1.item())
    return batch_f1_scores_best, batch_f1_scores_fixed

# Average Precision
def compute_ap(pred_mask, gt_mask):
    ap_cls = AveragePrecision(task="binary").to(pred_mask.device)
    pred_mask = pred_mask.flatten()
    # grounf truth has to be of type int
    gt_mask = gt_mask.flatten().int()

    ap_score = ap_cls(pred_mask, gt_mask)
    return ap_score.item()

def compute_batch_ap(predictions, ground_truths):
    assert len(predictions) == len(ground_truths), "Both lists must have the same length"

    ap_scores = []
    for pred_mask, gt_mask in zip(predictions, ground_truths):
        ap_score = compute_ap(pred_mask, gt_mask)
        ap_scores.append(ap_score)
    
    return ap_scores

# Detection
def compute_accuracy_detection(logits, labels, threshold=0.5):
    predicted_classes = (logits >= threshold).float()
    correct_predictions = (predicted_classes == labels).sum().item()
    accuracy = correct_predictions / labels.size()[0]
    return accuracy * 100

def compute_average_precision_detection(logits, labels):
    probabilities_np = logits.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    ap = average_precision_score(labels_np, probabilities_np)
    return ap

# Find best thresholf for detection classification
def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = len(y_true)

    if torch.max(y_pred[0:N//2]) <= torch.min(y_pred[N//2:N]): # perfectly separable case
        return (torch.max(y_pred[0:N//2]) + torch.min(y_pred[N//2:N])) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = copy.deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres.item()

# Outputs saving
def generate_outputs(output_save_path, pred, img_names=[]):
    for i in range(len(pred)):
        if isinstance(pred[i], torch.Tensor):
            pred_i = pred[i].detach().cpu().numpy()
        else:
            pred_i = pred[i]

        pred_mask_array = pred_i * 255
        pred_mask = Image.fromarray(pred_mask_array).convert(mode="L")
        pred_mask.save(output_save_path + "/" + img_names[i].split("/")[-1])