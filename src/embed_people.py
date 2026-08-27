"""Re-embed ONLY the people set with the frozen CLIP RN50 and merge into bundle.json.

Shapes and flags are left untouched (their image assets are not in the repo), so
this is text-only: no cairosvg, no flag PNGs. Run after gen_people.py.
Reuses the same PCA/quantise pipeline as embed.py so people stays in the same
kind of space as before.
"""
import base64, json, os
import numpy as np
import torch
import open_clip

CKPT = "weights/rn50-cc12m.pt"
DEV = "cpu"


@torch.no_grad()
def embed_texts(model, tok, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs]).to(DEV)
        f = model.encode_text(t).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def pca_reduce(X, d):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = min(d, Vt.shape[0])
    Z = Xc @ Vt[:d].T
    Z = Z / (S[:d] / np.sqrt(len(X)) + 1e-8) ** 0.5
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    ev = (S ** 2 / (S ** 2).sum())[:d].sum()
    return Z.astype(np.float32), float(ev)


def quantise(Z):
    scale = float(np.abs(Z).max())
    q = np.clip(np.round(Z / scale * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), scale, q.shape


def probe(Z, items, key):
    vals = sorted({str(i[key]) for i in items})
    if len(vals) < 2 or len(vals) > 14:
        return None
    idx = {v: [j for j, i in enumerate(items) if str(i[key]) == v] for v in vals}
    cents = {v: Z[ix].mean(0) for v, ix in idx.items() if len(ix) > 1}
    within = []
    for v, ix in idx.items():
        if len(ix) < 2:
            continue
        c = cents[v]
        within.append(np.mean([np.dot(Z[j], c) / (np.linalg.norm(c) + 1e-9) for j in ix]))
    cs = list(cents.values())
    between = []
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            between.append(np.dot(cs[a], cs[b]) /
                           ((np.linalg.norm(cs[a]) * np.linalg.norm(cs[b])) + 1e-9))
    return float(np.mean(within)), float(np.mean(between))


def main():
    model, _, _ = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained=CKPT, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer("RN50-quickgelu")
    print("model loaded", flush=True)

    people = json.load(open("assets/people.json"))
    Ep = embed_texts(model, tok, [p["prompt"] for p in people])
    Zp, evp = pca_reduce(Ep, 24)
    b64, sc, shp = quantise(Zp)
    print(f"people: {shp} explained={evp:.3f} ({len(people)} leaders)", flush=True)

    bundle = json.load(open("assets/bundle.json"))
    bundle["people"] = {"dim": shp[1], "scale": sc, "vecs": b64, "explained": evp, "items": people}
    with open("assets/bundle.json", "w") as f:
        json.dump(bundle, f)
    print("bundle", round(os.path.getsize("assets/bundle.json") / 1024), "KB", flush=True)

    print("\nseparability (within-group cos vs between-group cos; bigger gap = easier):")
    for key in ["industry", "region", "polarizing"]:
        r = probe(Zp, people, key)
        if r:
            print(f"  people/{key:10s} within={r[0]:.3f} between={r[1]:.3f}  gap={r[0]-r[1]:.3f}")


if __name__ == "__main__":
    main()
