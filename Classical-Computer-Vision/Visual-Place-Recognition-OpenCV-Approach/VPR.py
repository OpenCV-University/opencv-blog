import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


# Utilities
def list_images(images_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(exts)
    ]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")
    return paths


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def imread_bgr(path: str, max_width: Optional[int] = None) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if max_width is not None and img.shape[1] > max_width:
        scale = max_width / float(img.shape[1])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray,
                      left_title: str = "Query", right_title: str = "Match") -> np.ndarray:
    h = max(left_bgr.shape[0], right_bgr.shape[0])

    def pad_to_h(img):
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    left = pad_to_h(left_bgr)
    right = pad_to_h(right_bgr)

    def put_title(img, title):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(out, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    left = put_title(left, left_title)
    right = put_title(right, right_title)
    return np.concatenate([left, right], axis=1)



# Global Descriptor Model (ResNet50 + GeM)
class GeM(torch.nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        return x


class GlobalDescriptorNet(torch.nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = torch.nn.Sequential(*list(backbone.children())[:-2])  # [B, 2048, H/32, W/32]
        self.pool = GeM()
        self.proj = torch.nn.Linear(2048, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [B, 2048]
        x = self.proj(x)                          # [B, D]
        x = F.normalize(x, p=2, dim=1)
        return x



# Geometric verification (optional)
def geom_verify_orb(img1_bgr: np.ndarray, img2_bgr: np.ndarray,
                    min_inliers: int = 30, reproj_thresh: float = 3.0) -> Tuple[bool, int]:
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return False, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_inliers:
        return False, 0

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=reproj_thresh)
    if mask is None:
        return False, 0

    inliers = int(mask.sum())
    return inliers >= min_inliers, inliers



# Data structures
@dataclass
class MatchResult:
    query_idx: int
    match_idx: int
    score: float
    geom_verified: bool
    inliers: int
    query_path: str
    match_path: str


def cosine_similarity_matrix(q: np.ndarray, db: np.ndarray) -> np.ndarray:
    # q: [D], db: [N, D] (both L2-normalized)
    return db @ q



# Descriptor extraction helpers
def build_transform(img_size: int):
    w = models.ResNet50_Weights.DEFAULT
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=w.transforms().mean, std=w.transforms().std),
    ])
    return tfm


@torch.no_grad()
def extract_desc(model, tfm, device, img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = tfm(img_rgb).unsqueeze(0).to(device)
    desc = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return desc



# Main
def main():
    parser = argparse.ArgumentParser(description="VPR Retrieval + Loop/Match Saving (GeM + optional ORB verification)")

    # Mode selection:
    # - Online: --images_dir (single ordered sequence; DB is past frames)
    # - Offline: --db_dir + --query_dir (explicit split)
    parser.add_argument("--images_dir", type=str, default=None, help="Online mode: single ordered image sequence folder")
    parser.add_argument("--db_dir", type=str, default=None, help="Offline mode: database/reference images folder")
    parser.add_argument("--query_dir", type=str, default=None, help="Offline mode: query images folder")

    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save results")

    # Model / extraction
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Retrieval / decision
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sim_threshold", type=float, default=0.75)
    parser.add_argument("--exclude_recent", type=int, default=30,
                        help="Online mode only: exclude last N frames from matching")
    parser.add_argument("--min_loop_gap", type=int, default=200,
                        help="Online mode only: minimum frame gap i-j to accept loop")

    # Geometric verification
    parser.add_argument("--verify_geom", action="store_true")
    parser.add_argument("--min_inliers", type=int, default=20)
    parser.add_argument("--reproj_thresh", type=float, default=3.0)

    # Saving visuals
    parser.add_argument("--max_visuals", type=int, default=80)
    parser.add_argument("--preview_max_width", type=int, default=900)

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    matches_dir = os.path.join(args.output_dir, "matches")
    ensure_dir(matches_dir)

    # Decide mode
    offline = (args.db_dir is not None and args.query_dir is not None)
    online = (args.images_dir is not None)

    if offline and online:
        raise ValueError("Use either --images_dir (online) OR --db_dir + --query_dir (offline), not both.")
    if not offline and not online:
        raise ValueError("Provide --images_dir (online) OR --db_dir + --query_dir (offline).")

    device = torch.device(args.device)
    model = GlobalDescriptorNet(embed_dim=args.embed_dim).to(device).eval()
    tfm = build_transform(args.img_size)

    results: List[MatchResult] = []
    best_scores = []

    saved_visuals = 0

    if offline:
        
        # OFFLINE: query -> database
        db_paths = list_images(args.db_dir)
        q_paths = list_images(args.query_dir)

        print(f"[INFO] Offline mode")
        print(f"[INFO] DB images: {len(db_paths)} | Query images: {len(q_paths)}")

        # Build DB descriptors once
        db_desc = []
        for p in tqdm(db_paths, desc="Encoding DB"):
            img = imread_bgr(p)
            db_desc.append(extract_desc(model, tfm, device, img))
        db_desc = np.stack(db_desc, axis=0)  # [Ndb, D]

        # Process queries
        for qi, qp in enumerate(tqdm(q_paths, desc="Querying")):
            q_img = imread_bgr(qp)
            q_desc = extract_desc(model, tfm, device, q_img)

            sims = cosine_similarity_matrix(q_desc, db_desc)
            topk = min(args.top_k, sims.shape[0])
            top_idx = np.argpartition(-sims, kth=topk-1)[:topk]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            top_scores = sims[top_idx]

            best = float(top_scores[0]) if len(top_scores) else 0.0
            best_scores.append(best)

            if len(top_scores) and best >= args.sim_threshold:
                cand_j = int(top_idx[0])
                score = float(top_scores[0])

                geom_ok = True
                inliers = -1
                if args.verify_geom:
                    cand_img = imread_bgr(db_paths[cand_j])
                    geom_ok, inliers = geom_verify_orb(
                        img1_bgr=q_img,
                        img2_bgr=cand_img,
                        min_inliers=args.min_inliers,
                        reproj_thresh=args.reproj_thresh,
                    )

                if (not args.verify_geom) or geom_ok:
                    results.append(MatchResult(
                        query_idx=qi,
                        match_idx=cand_j,
                        score=score,
                        geom_verified=bool(args.verify_geom and geom_ok),
                        inliers=int(inliers) if args.verify_geom else -1,
                        query_path=str(qp),
                        match_path=str(db_paths[cand_j]),
                    ))

                    if saved_visuals < args.max_visuals:
                        cand_small = imread_bgr(db_paths[cand_j], max_width=args.preview_max_width)
                        q_small = imread_bgr(qp, max_width=args.preview_max_width)
                        title_q = f"Query #{qi} (sim={score:.3f})"
                        title_m = f"Dataset #{cand_j}"
                        if args.verify_geom:
                            title_m += f" (inliers={inliers})"
                        viz = make_side_by_side(q_small, cand_small, title_q, title_m)
                        out_path = os.path.join(matches_dir, f"match_q{qi:06d}_to_db{cand_j:06d}.jpg")
                        cv2.imwrite(out_path, viz)
                        saved_visuals += 1

    else:
        
        # ONLINE: query against past frames
        image_paths = list_images(args.images_dir)
        n = len(image_paths)
        print(f"[INFO] Online mode: Found {n} images.")

        db_desc_list = []
        best_scores = np.zeros(n, dtype=np.float32)

        for i in tqdm(range(n), desc="Processing frames"):
            img_bgr = imread_bgr(image_paths[i])
            desc = extract_desc(model, tfm, device, img_bgr)

            if len(db_desc_list) > args.exclude_recent:
                db = np.stack(db_desc_list, axis=0)  # [i, D]
                sims = cosine_similarity_matrix(desc, db)

                if args.exclude_recent > 0:
                    sims[-args.exclude_recent:] = -1.0

                k = min(args.top_k, sims.shape[0])
                top_idx = np.argpartition(-sims, kth=k-1)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                top_scores = sims[top_idx]

                best_scores[i] = float(top_scores[0]) if len(top_scores) else 0.0

                if len(top_scores) and float(top_scores[0]) >= args.sim_threshold:
                    cand_j = int(top_idx[0])
                    score = float(top_scores[0])

                    if (i - cand_j) < args.min_loop_gap:
                        db_desc_list.append(desc)
                        continue

                    geom_ok = True
                    inliers = -1
                    if args.verify_geom:
                        cand_bgr = imread_bgr(image_paths[cand_j])
                        geom_ok, inliers = geom_verify_orb(
                            img1_bgr=img_bgr,
                            img2_bgr=cand_bgr,
                            min_inliers=args.min_inliers,
                            reproj_thresh=args.reproj_thresh,
                        )

                    if (not args.verify_geom) or geom_ok:
                        results.append(MatchResult(
                            query_idx=i,
                            match_idx=cand_j,
                            score=score,
                            geom_verified=bool(args.verify_geom and geom_ok),
                            inliers=int(inliers) if args.verify_geom else -1,
                            query_path=str(image_paths[i]),
                            match_path=str(image_paths[cand_j]),
                        ))

                        if saved_visuals < args.max_visuals:
                            cand_small = imread_bgr(image_paths[cand_j], max_width=args.preview_max_width)
                            q_small = imread_bgr(image_paths[i], max_width=args.preview_max_width)
                            title_q = f"Query #{i} (sim={score:.3f})"
                            title_m = f"Match #{cand_j}"
                            if args.verify_geom:
                                title_m += f" (inliers={inliers})"
                            viz = make_side_by_side(q_small, cand_small, title_q, title_m)
                            out_path = os.path.join(matches_dir, f"loop_{i:06d}_to_{cand_j:06d}.jpg")
                            cv2.imwrite(out_path, viz)
                            saved_visuals += 1

            db_desc_list.append(desc)

    # Save JSON results
    json_path = os.path.join(args.output_dir, "loop_closures.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"[INFO] Saved results: {json_path}")
    print(f"[INFO] Total accepted matches/loops: {len(results)}")
    print(f"[INFO] Saved match previews in: {matches_dir}")

    # Plot similarity timeline (best score per query)
    plt.figure(figsize=(12, 4))
    plt.plot(best_scores, linewidth=1.0)
    plt.axhline(args.sim_threshold, linestyle="--", linewidth=1.0)
    plt.title("Best Similarity to Database (Signal)")
    plt.xlabel("Query index")
    plt.ylabel("Cosine Similarity (higher = more similar)")
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "similarity_plot.png")
    plt.savefig(plot_path, dpi=160)
    print(f"[INFO] Saved similarity plot: {plot_path}")


if __name__ == "__main__":
    main()
