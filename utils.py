import datasets.cifar10
import numpy as np
import torch
from clip import clip
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import json
import pandas as pd

from dassl.config import get_cfg_default
from dassl.data.datasets.build import build_dataset

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers

CUSTOM_TEMPLATES = {}

def load_results(backbone):
    filename = "results_zs_all_RN50.pkl" if backbone == "RN50" else "results_zs_all_ViT16.pkl"
    with open(f"zs_results/{filename}", "rb") as f:
        return pickle.load(f)
    

def get_configs(args):
    
    # one class configurations
    onecls_configs = {
            'RN50':{
            'StanfordCars': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.4, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
            },
            'ViT-B/16': {
                'StanfordCars': {'lamb_preserve': 0.25, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
                'StanfordDogs': {'lamb_preserve': 0.3, 'lamb_forget': 1.3, 'lora_r': 5, 'lamb_weight': 1.},
                'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
                'OxfordFlowers': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.}
            }

    }
    
    # Multiclass configurations
    multiclass_configs = {
        'RN50': {
            'StanfordCars': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
        },
        'ViT-B/16': {
            'StanfordCars': {'lamb_preserve': 0.35, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'StanfordDogs': {'lamb_preserve': 0.35, 'lamb_forget': 1.0, 'lora_r': 5, 'lamb_weight': 1.},
            'Caltech101': {'lamb_preserve': 0.3, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
            'OxfordFlowers': {'lamb_preserve': 0.25, 'lamb_forget': 1.1, 'lora_r': 5, 'lamb_weight': 1.},
        }
    }

    # Determine configurations based on arguments
    if args.multiclass_forget:
        print(f"SETTING {args.backbone_arch} MULTICLASS")
        configs = multiclass_configs[args.backbone_arch]
    else:
        print(f"SETTING {args.backbone_arch} ONE CLASS")
        configs = onecls_configs[args.backbone_arch]

    return configs


def get_model(arch="RN50", device='cpu', load_path="", lr=5e-5):
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    print("Loading model...")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict()).float().to(device).eval()
    
    if load_path:
        print(f"LOADING FROM {load_path}")
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        model = model.float().to(device).eval()
    
    return model


def eval_all_ds(model, datasets_cls, forget_ds, forget_lbl, all_loaders, train_loader=None, eval_forgetonly=False, debug=False, device='cpu', ignore_labels_main=[]):
        
    results = {ds: {} for ds in all_loaders}
    for ds in all_loaders:
        model.eval()
        test_loader = all_loaders[ds]
        
        classnames = datasets_cls[ds].classnames
        # clip_weights = clip_classifier(classnames, [CUSTOM_TEMPLATES[ds]], model).to(device)
        clip_weights = clip_classifier(classnames, CUSTOM_TEMPLATES[ds], model).to(device)

        
        if ds == forget_ds:
            cls_acc_test = None
            no_cls_acc = None
            if debug:
                acc, (labels, clip_logits_test) = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=True)
                # print("acc", acc)
                if ignore_labels_main:
                    ignore_labels = []
                    for tlbl in ignore_labels_main:
                        ignore_labels.append(classnames.index(tlbl))
                    ignore_labels = np.array(ignore_labels)
                    
                    mask_labels = (torch.tensor((~np.isin(labels, ignore_labels)), dtype=labels.dtype)).bool()
                elif forget_lbl not in classnames:
                    id_lbl = 0
                    mask_labels = labels != -9999
                else:
                    id_lbl = classnames.index(forget_lbl)
                    mask_labels = labels != id_lbl
                
                    cls_acc_test = confusion_matrix(labels, clip_logits_test.argmax(1))[id_lbl]
                    cls_acc_test = cls_acc_test[id_lbl] / cls_acc_test.sum()
                    
                no_cls_acc = confusion_matrix(labels[mask_labels], clip_logits_test.argmax(1)[mask_labels])
                no_cls_acc = np.diag(no_cls_acc).sum() / no_cls_acc.sum()
                
                if ignore_labels_main:
                    out_acc_all = {}
                    for c in ignore_labels_main:
                        c_id = classnames.index(c)
                        cls_acc_test = confusion_matrix(labels, clip_logits_test.argmax(1))[c_id]
                        cls_acc_test = cls_acc_test[c_id] / cls_acc_test.sum()
                        out_acc_all[c] = cls_acc_test
            
            # include accuracy of the train data if not None
            if train_loader is not None:
                acc_train = evaluate_clip_zs(model, train_loader, clip_weights, device=device, out_conf=False)
                acc_train = acc_train 
            else:
                acc_train = None
                
            if ignore_labels_main:
                results[ds]['|'.join(ignore_labels_main)] = {'cls_acc_test' : out_acc_all, 
                                          'no_cls_acc' : no_cls_acc, 
                                          'acc_train' : acc_train}
                print(f"{10*'+++'} Train dataset: {ds} - {results[ds]['|'.join(ignore_labels_main)]} {10*'+++'}")
            else:
                results[ds][forget_lbl] = {'cls_acc_test' : cls_acc_test, 
                                          'no_cls_acc' : no_cls_acc, 
                                          'acc_train' : acc_train}
                
                print(f"{10*'+++'} Train dataset: {ds} - {results[ds][forget_lbl]} {10*'+++'}")
                
        else:
            # continue
            if eval_forgetonly or (not debug): continue
            acc = evaluate_clip_zs(model, test_loader, clip_weights, device=device,out_conf=False)
            results[ds]['all'] = {'all_ds' : acc}     
            print(f"{10*'+++'} {ds} - {acc} {10*'+++'}")
                
    return results

        
def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(clip_model.visual.conv1.weight.device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1)#.cuda()
    return clip_weights

def cls_acc(output, target, topk=1):
    # Get the topk predictions
    # pred = np.argsort(output, axis=1)[:, -topk:][:, ::-1].T
    pred = np.argmax(output, axis=1)
    
    # Check if predictions match the target
    correct = pred == target.reshape(1, -1)
    
    # Calculate accuracy
    acc = correct[:topk].reshape(-1).sum(0)
    acc = 100 * acc / target.shape[0]
    
    return acc

def evaluate_clip_zs(model, loader, clip_weights, device=None, out_conf=False, output_probs=False):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):    
            images = batch['img']
            target = batch['label']

            images, target = images.to(device), target.to(device)
            # image_features, image_features_projected = mymodel.encode_image(images)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.cpu())
            labels.append(target.cpu()) 
            
    labels = torch.cat(labels)
    features = torch.cat(features)
    
    clip_logits_test = 100. * features @ clip_weights.detach().cpu().numpy()
    acc = cls_acc(clip_logits_test.detach().cpu().numpy(), labels.detach().cpu().numpy())
    acc = acc / 100.
    
    if output_probs:
        probs = torch.nn.functional.softmax(clip_logits_test, dim=-1)
    
    if out_conf:
        if output_probs:
            return acc, (labels, clip_logits_test), probs
            
        return acc, (labels, clip_logits_test)
    
    if output_probs:
        return acc, probs

    return acc

# @torch.no_grad()
# def eval_zeroshot_metrics(model, loader, P_txt, device, forget_ids, tau=0.07):
#     model.eval()
#     total, correct = 0, 0
#     correct_f, cnt_f = 0, 0
#     correct_r, cnt_r = 0, 0
#     top1, top5 = 0, 0
#     C = P_txt.size(0)
#     # for images, labels in tqdm(loader, desc="Eval (zero-shot)"):
#     for batch in tqdm(loader, desc="Eval (zero-shot)"):
#     # Handle dataset formats flexibly
#         if isinstance(batch, dict):
#             images = batch.get("img", batch.get("image"))
#             labels = batch.get("label", batch.get("target"))
#         elif isinstance(batch, (list, tuple)):
#             images, labels = batch[0], batch[1]
#         else:
#             raise TypeError(f"Unexpected batch type: {type(batch)}")

#         images = images.to(device, non_blocking=True)
#         labels = labels.to(device, non_blocking=True)
#         z = model.encode_image(images)
#         z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
#         # Ensure dimensions are compatible
#         if P_txt.shape[1] != z.shape[1]:
#             P_txt = P_txt.T  # Fix orientation if text embedding shape is transposed

#         logits = 100. * z @ P_txt.T  # (batch, embed) @ (embed, classes) → (batch, classes)

#         # logits = 100. * z @ P_txt.T  # Similar to clip_logits
#         preds = logits.argmax(dim=-1)
#         total += labels.numel()
#         correct += (preds == labels).sum().item()
#         # top-k
#         _, topk = logits.topk(k=min(5, C), dim=-1)
#         top1 += (topk[:, :1] == labels.unsqueeze(1)).any(dim=1).float().sum().item()
#         top5 += (topk == labels.unsqueeze(1)).any(dim=1).float().sum().item()

#         # Separate forget vs retain
#         if len(forget_ids) > 0:
#             mask_f = torch.isin(labels, torch.tensor(forget_ids, device=labels.device))
#             mask_r = ~mask_f
#             if mask_f.any():
#                 correct_f += (preds[mask_f] == labels[mask_f]).sum().item()
#                 cnt_f += mask_f.sum().item()
#             if mask_r.any():
#                 correct_r += (preds[mask_r] == labels[mask_r]).sum().item()
#                 cnt_r += mask_r.sum().item()

#     overall = correct / max(1, total)
#     forget_acc = (correct_f / max(1, cnt_f)) if cnt_f > 0 else float("nan")
#     retain_acc = (correct_r / max(1, cnt_r)) if cnt_r > 0 else float("nan")
#     r1 = top1 / max(1, total)
#     r5 = top5 / max(1, total)
#     return {
#         "overall": overall,
#         "retain_acc": retain_acc,
#         "forget_acc": forget_acc,
#         "r1": r1,
#         "r5": r5
#     }
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

@torch.no_grad()
def eval_zeroshot_metrics(model, loader, P_txt, device, forget_ids, tau=0.07):
    """
    Returns dict with:
     - overall, retain_acc, forget_acc, r1, r5 (as before)
     - conf_mat: full confusion matrix (numpy)
     - per_class: dict with precision, recall (TPR), f1, support (arrays)
     - macro_precision, macro_recall, macro_f1
    """
    model.eval()
    preds_all = []
    labels_all = []

    C = P_txt.size(0)

    for batch in tqdm(loader, desc="Eval (zero-shot)"):
        # unpack robustly
        if isinstance(batch, dict):
            images = batch.get("img", batch.get("image"))
            labels = batch.get("label", batch.get("target"))
        else:
            images, labels = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        z = model.encode_image(images)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        # Ensure P_txt orientation
        if P_txt.shape[1] == z.shape[1]:
            P_mat = P_txt.T
        elif P_txt.shape[0] == z.shape[1]:
            P_mat = P_txt.T
        else:
            P_mat = P_txt.T

        # logits = 100. * z @ P_mat  # (B, C)
        # Ensure orientation (want (embed_dim, num_classes))
        if P_mat.shape[0] == z.shape[1]:
            logits = 100. * z @ P_mat
        elif P_mat.shape[1] == z.shape[1]:
            logits = 100. * z @ P_mat.T
        else:
            raise RuntimeError(f"Incompatible shapes: z={z.shape}, P_mat={P_mat.shape}")
        preds = logits.argmax(dim=-1)

        preds_all.append(preds.cpu())
        labels_all.append(labels.cpu())

    labels_all = torch.cat(labels_all).numpy()
    preds_all = torch.cat(preds_all).numpy()

    overall = (preds_all == labels_all).mean()

    # separate forget vs retain
    mask_f = np.isin(labels_all, np.array(forget_ids))
    if mask_f.any():
        forget_acc = (preds_all[mask_f] == labels_all[mask_f]).mean()
    else:
        forget_acc = float("nan")
    mask_r = ~mask_f
    if mask_r.any():
        retain_acc = (preds_all[mask_r] == labels_all[mask_r]).mean()
    else:
        retain_acc = float("nan")

    # top-1 and top-5 (recompute properly)
    # We need to recompute logits to get top-k; do it quick in batches:
    # (for simplicity reuse evaluate_clip_zs if exists; otherwise you can compute top1/top5 above)
    # Here compute top1/top5 approximated by overall and top-5 by checking model outputs again:
    # For now return r1 as overall, r5 as NaN (or compute separately if needed)
    r1 = overall
    r5 = float("nan")

    # confusion matrix + per-class metrics
    conf = confusion_matrix(labels_all, preds_all)
    precision, recall, f1, support = precision_recall_fscore_support(labels_all, preds_all, zero_division=0)

    per_class = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),   # recall == TPR per class
        "f1": f1.tolist(),
        "support": support.tolist()
    }
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    return {
        "overall": float(overall),
        "retain_acc": float(retain_acc) if not np.isnan(retain_acc) else float("nan"),
        "forget_acc": float(forget_acc) if not np.isnan(forget_acc) else float("nan"),
        "r1": float(r1),
        "r5": float(r5),
        "conf_mat": conf.tolist(),
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

# @torch.no_grad()
# def compute_embedding_drift(teacher, student, loader, device,P_txt=None):
#     teacher.eval()
#     student.eval()
#     dists = []
#     for images, _ in tqdm(loader, desc="Drift pass"):
#         images = images.to(device, non_blocking=True)
#         z_t = teacher.encode_image(images)
#         z_s = student.encode_image(images)
#         z_t = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-8)
#         z_s = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-8)
#         d = (z_t - z_s).pow(2).sum(dim=-1).sqrt()  # ℓ2 distance
#         dists.append(d.cpu())
#     if len(dists) == 0:
#         return float('nan')
#     return float(torch.cat(dists, dim=0).mean().item())
# @torch.no_grad()
# def compute_embedding_drift(teacher, student, loader, device, P_txt=None):
#     teacher.eval()
#     student.eval()
#     dists = []

#     for batch in tqdm(loader, desc="Drift pass"):
#         # Handle both dict and tuple outputs
#         if isinstance(batch, dict):
#             images = batch["img"].to(device, non_blocking=True)
#         else:
#             images, _ = batch
#             images = images.to(device, non_blocking=True)

#         # Encode
#         z_t = teacher.encode_image(images)
#         z_s = student.encode_image(images)

#         # Normalize
#         z_t = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-8)
#         z_s = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-8)

#         # Optional: project into text space
#         if P_txt is not None:
#             z_t = z_t @ P_txt
#             z_s = z_s @ P_txt

#         # ℓ2 distance
#         d = (z_t - z_s).pow(2).sum(dim=-1).sqrt()
#         dists.append(d.cpu())

#     if len(dists) == 0:
#         return float("nan")

#     return float(torch.cat(dists, dim=0).mean().item())
@torch.no_grad()
def compute_embedding_drift(
    teacher, student, loader, device,
    P_txt_teacher: torch.Tensor = None,
    P_txt_student: torch.Tensor = None,
    return_per_sample: bool = False
):
    """
    Returns a dict with:
      - image_drift: mean L2 distance between teacher and student image embeddings
      - textsim_drift_teacherproj: mean L2 between (z_t @ P_txt_teacher) and (z_s @ P_txt_teacher)
          (useful to detect image-encoder changes when using same classifier)
      - textsim_drift_studentproj: mean L2 between (z_t @ P_txt_teacher) and (z_s @ P_txt_student)
          (captures changes due to text projection differences)
    P_txt_teacher and P_txt_student should be shape (num_classes, embed_dim) or (embed_dim, num_classes).
    """
    teacher.eval()
    student.eval()

    image_dists = []
    ts_teacher_dists = []
    ts_student_dists = []

    for batch in tqdm(loader, desc="Drift pass"):
        # robust batch unpacking
        if isinstance(batch, dict):
            images = batch.get("img", batch.get("image"))
        else:
            images = batch[0]

        images = images.to(device, non_blocking=True)

        z_t = teacher.encode_image(images)
        z_s = student.encode_image(images)

        z_t = z_t / (z_t.norm(dim=-1, keepdim=True) + 1e-8)
        z_s = z_s / (z_s.norm(dim=-1, keepdim=True) + 1e-8)

        # image drift (original)
        d_img = (z_t - z_s).pow(2).sum(dim=-1).sqrt().cpu()
        image_dists.append(d_img)

        # text-sim drifts — only if P_txt provided
        if P_txt_teacher is not None:
            P_t = P_txt_teacher
            # ensure orientation: want shape (embed_dim, num_classes) to multiply z @ P
            if P_t.shape[0] == z_t.shape[1]:
                P_t_mat = P_t
            elif P_t.shape[1] == z_t.shape[1]:
                P_t_mat = P_t.T
            else:
                raise RuntimeError(f"P_txt_teacher shape {P_t.shape} incompatible with embed dim {z_t.shape[1]}")

            logits_t = z_t @ P_t_mat  # (B, C)
            logits_s = z_s @ P_t_mat
            d_ts = (logits_t - logits_s).pow(2).sum(dim=-1).sqrt().cpu()
            ts_teacher_dists.append(d_ts)

        if (P_txt_teacher is not None) and (P_txt_student is not None):
            # ensure student P_txt orientation
            P_s = P_txt_student
            if P_s.shape[0] == z_s.shape[1]:
                P_s_mat = P_s
            elif P_s.shape[1] == z_s.shape[1]:
                P_s_mat = P_s.T
            else:
                raise RuntimeError(f"P_txt_student shape {P_s.shape} incompatible with embed dim {z_s.shape[1]}")

            logits_t = z_t @ P_t_mat  # teacher proj
            logits_s_student = z_s @ P_s_mat
            d_ts_student = (logits_t - logits_s_student).pow(2).sum(dim=-1).sqrt().cpu()
            ts_student_dists.append(d_ts_student)

    # aggregate
    if len(image_dists) == 0:
        return {
            "image_drift": float("nan"),
            "textsim_drift_teacherproj": float("nan"),
            "textsim_drift_studentproj": float("nan")
        }

    image_d = float(torch.cat(image_dists).mean().item())
    ts_teacher_d = float(torch.cat(ts_teacher_dists).mean().item()) if len(ts_teacher_dists) else float("nan")
    ts_student_d = float(torch.cat(ts_student_dists).mean().item()) if len(ts_student_dists) else float("nan")

    out = {
        "image_drift": image_d,
        "textsim_drift_teacherproj": ts_teacher_d,
        "textsim_drift_studentproj": ts_student_d
    }

    if return_per_sample:
        out["per_sample_image"] = torch.cat(image_dists).numpy()
        if len(ts_teacher_dists): out["per_sample_text_teacherproj"] = torch.cat(ts_teacher_dists).numpy()
        if len(ts_student_dists): out["per_sample_text_studentproj"] = torch.cat(ts_student_dists).numpy()

    return out

def acc_certain_cls(acc_all, all_lbls, ids_lbl):
    acc_selected = acc_all[ids_lbl]
    labels_selected = np.unique(all_lbls, return_counts=True)[1][ids_lbl]
    
    return np.average(acc_selected, weights=labels_selected)


def create_results(res_folder, return_logs=False, rn=True, log_name='logs.json', multiclass=False):
    
    if res_folder != "":
        with open(res_folder + f"/{log_name}", "r") as f:
            all_logs = json.load(f)

        with open(res_folder + "/args.txt", "r") as f:
            args = f.read()
    else:
        all_logs = log_name
        args = ""
        
    if rn:
        results_zs = load_results("RN50")
    else:
        results_zs = load_results("ViT16")
        
    full_df = []
    final_results = {}
    add_cols = []
    for jj, file in enumerate(all_logs):

        cols = ['cls_forget', 'full_forget', 'acc_train']
        
        single_df = pd.DataFrame(columns=cols)

        if file == 'settings': continue
        final_results[file] = {}
        
        if multiclass:
            all_cls_sameds = []
            all_cls_sameds_ids = []
            cfg = get_cfg_default()
            cfg.DATASET.NAME = file
            cfg.DATASET.SUBSAMPLE_CLASSES = "all"
            cfg.DATASET.ROOT = "/app/datasets/"
            cfg.DATASET.NUM_SHOTS = -1
            dataset = build_dataset(cfg)
            all_lbls = torch.tensor([d.label for d in dataset.test])
            all_accuracies = torch.tensor([results_zs[file][key]['cls_acc_test'] for key in results_zs[file]])

        for ii, k in enumerate(all_logs[file]):
            if not all_logs[file][k]: continue
            if 'kwargs' in all_logs[file][k]: continue
            
            final_results[file][k] = all_logs[file][k]['final_results'][file][k]
            
            single_df.loc[ii, cols] = pd.DataFrame(final_results[file][k].items())[1].values
            single_df.loc[ii, 'name'] = k
            single_df.loc[ii, 'ds'] = file
            
            if multiclass:
                key_splitted = k.split("|")
                all_cls_sameds_ids = [dataset.classnames.index(key) for key in key_splitted]
                remaining_ids = torch.tensor([dataset.classnames.index(cln) for cln in dataset.classnames if cln not in key_splitted])
                
                single_df.loc[ii, 'full_Noforget'] = acc_certain_cls(all_accuracies, all_lbls, remaining_ids)
                single_df.loc[ii, 'cls_Noforget'] = acc_certain_cls(all_accuracies, all_lbls, all_cls_sameds_ids)
            else:
                single_df.loc[ii, 'full_Noforget'] = results_zs[file][k]['no_cls_acc']
                single_df.loc[ii, 'cls_Noforget'] = results_zs[file][k]['cls_acc_test']
                            
            for k1 in list(all_logs[file][k]['final_results'].keys()):
                if 'all' not in all_logs[file][k]['final_results'][k1]: continue
                single_df.loc[ii, f'res_{k1}'] = all_logs[file][k]['final_results'][k1]['all']['all_ds']
                single_df.loc[ii, f'full_{k1}'] = results_zs[k1][list(results_zs[k1].keys())[0]]['full_acc']
                if f'res_{k1}' not in add_cols:
                    add_cols.append(f'res_{k1}')
                if f'full_{k1}' not in add_cols:
                    add_cols.append(f'full_{k1}')
                
        full_df.append(single_df)
    
    full_df = pd.concat(full_df)[['ds', 'name', 'full_Noforget', 'cls_Noforget'] + ['cls_forget', 'full_forget'] + add_cols]
        
    if multiclass:
        full_df = full_df.reset_index(drop=True)
        full_df['cls_forget_all'] = full_df['cls_forget']
        for indx, row in enumerate(full_df.iterrows()):
            full_df.loc[indx, 'cls_forget'] = np.mean([val for key, val in row[1]['cls_forget'].items()])
        
        return full_df, args
    return full_df, args 
    
    
            
def compute_avg_gain(df):
    forget_perc = 1.- (df['cls_Noforget'] - df['cls_forget'])/df['cls_Noforget']
    list_main_perc = ((df['full_Noforget'] - df['full_forget'])/df['full_Noforget']).clip(0)
    forget_perc_scars = ((df['full_StanfordCars'] - df['res_StanfordCars'])/df['full_StanfordCars']).clip(0).fillna(0)
    forget_perc_caltech = ((df['full_Caltech101'] - df['res_Caltech101'])/df['full_Caltech101']).clip(0).fillna(0)
    forget_perc_oxflow = ((df['full_OxfordFlowers'] - df['res_OxfordFlowers'])/df['full_OxfordFlowers']).clip(0).fillna(0)
    forget_perc_sdogs = ((df['full_StanfordDogs'] - df['res_StanfordDogs'])/df['full_StanfordDogs']).clip(0).fillna(0)
    # divide by 5 as we have 4 datasets + forget_perc (we have 6 elements below but in each row one element is 0 as it's NA)
    scores = (forget_perc + list_main_perc + forget_perc_scars + forget_perc_caltech + forget_perc_oxflow + forget_perc_sdogs)/5
    return scores.astype(float)
