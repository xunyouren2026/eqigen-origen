import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import clip
import lpips

# 可选依赖，用于 FVD
try:
    import scipy.linalg
except ImportError:
    scipy = None


def compute_fvd(gen_videos, real_videos, device='cuda'):
    """
    计算 Fréchet Video Distance (FVD)
    gen_videos: list of numpy arrays, each shape (T, H, W, C), values in [0,255]
    real_videos: same
    """
    try:
        import torch
        import torch.nn.functional as F
        from pytorch_i3d import InceptionI3d
    except ImportError:
        print("pytorch_i3d not installed, FVD returns 0.0")
        return 0.0

    if scipy is None:
        print("scipy not installed, FVD returns 0.0")
        return 0.0

    # 加载 I3D 模型（预训练）
    try:
        i3d = InceptionI3d(400, in_channels=3).to(device)
        state_dict = torch.hub.load_state_dict_url(
            'https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt?raw=True',
            map_location=device
        )
        i3d.load_state_dict(state_dict)
        i3d.eval()
    except Exception as e:
        print(f"Failed to load I3D model: {e}, FVD returns 0.0")
        return 0.0

    def extract_features(videos):
        features = []
        for vid in videos:
            # vid: (T, H, W, C), uint8 -> (1, C, T, H, W) float32 normalized to [-1,1]
            vid_t = torch.from_numpy(vid).permute(
                3, 0, 1, 2).unsqueeze(0).float().to(device)
            vid_t = (vid_t / 127.5) - 1.0
            with torch.no_grad():
                feat = i3d.extract_features(vid_t)
                feat = F.adaptive_avg_pool3d(
                    feat, (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)
                features.append(feat.cpu().numpy())
        return np.array(features)

    gen_feats = extract_features(gen_videos)
    real_feats = extract_features(real_videos)

    mu_gen = np.mean(gen_feats, axis=0)
    mu_real = np.mean(real_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    sigma_real = np.cov(real_feats, rowvar=False)

    diff = mu_gen - mu_real
    covmean, _ = scipy.linalg.sqrtm(sigma_gen.dot(sigma_real), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2 * covmean)
    return fvd


def compute_psnr(gen, real):
    return np.mean([peak_signal_noise_ratio(g, r, data_range=255) for g, r in zip(gen, real)])


def compute_ssim(gen, real):
    return np.mean([structural_similarity(g, r, multichannel=True, channel_axis=-1) for g, r in zip(gen, real)])


def compute_clip_score(gen_videos, texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    scores = []
    for video, text in zip(gen_videos, texts):
        frame = video[len(video)//2]
        image = preprocess(frame).unsqueeze(0).to(device)
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)
            score = (image_features @ text_features.T).item()
        scores.append(score)
    return np.mean(scores)


def compute_lpips(gen_videos, real_videos):
    loss_fn = lpips.LPIPS(net='alex')
    scores = []
    for g, r in zip(gen_videos, real_videos):
        g_t = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        r_t = torch.from_numpy(r).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        score = loss_fn(g_t, r_t).item()
        scores.append(score)
    return np.mean(scores)
