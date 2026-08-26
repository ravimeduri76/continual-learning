"""Embed all three item sets with a frozen CLIP RN50, then PCA-reduce and quantise."""
import base64, io, json, os
import numpy as np
import torch, cairosvg
from PIL import Image
import open_clip

CKPT = "weights/rn50-cc12m.pt"
DEV = "cpu"


def load():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN50-quickgelu", pretrained=CKPT, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer("RN50-quickgelu")
    return model, preprocess, tok


@torch.no_grad()
def embed_images(model, preprocess, pils, bs=32):
    out = []
    for i in range(0, len(pils), bs):
        batch = torch.stack([preprocess(p) for p in pils[i:i + bs]]).to(DEV)
        f = model.encode_image(batch).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        print(f"  img {min(i+bs, len(pils))}/{len(pils)}", flush=True)
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(model, tok, texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs]).to(DEV)
        f = model.encode_text(t).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out)


def pca_reduce(X, d):
    """Centre, PCA to d dims, whiten mildly, renormalise to the unit sphere."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = min(d, Vt.shape[0])
    Z = Xc @ Vt[:d].T
    # mild whitening: equalise the variance ladder so later components still matter
    Z = Z / (S[:d] / np.sqrt(len(X)) + 1e-8) ** 0.5
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    ev = (S ** 2 / (S ** 2).sum())[:d].sum()
    return Z.astype(np.float32), float(ev)


def quantise(Z):
    """int8 with one global scale; ~0.008 max error on unit vectors."""
    scale = float(np.abs(Z).max())
    q = np.clip(np.round(Z / scale * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), scale, q.shape


def main():
    model, preprocess, tok = load()
    print("model loaded", flush=True)
    bundle = {}

    # ---------------- shapes: pure vision ----------------
    shapes = json.load(open("assets/shapes.json"))
    pils = []
    for s in shapes:
        png = cairosvg.svg2png(bytestring=s["svg"].encode(), output_width=224, output_height=224)
        pils.append(Image.open(io.BytesIO(png)).convert("RGB"))
    print("rasterised shapes", len(pils), flush=True)
    Es = embed_images(model, preprocess, pils)
    Zs, evs = pca_reduce(Es, 24)
    b64, sc, shp = quantise(Zs)
    bundle["shapes"] = {
        "dim": shp[1], "scale": sc, "vecs": b64, "explained": evs,
        "items": [{k: v for k, v in s.items()} for s in shapes],
    }
    print(f"shapes: {shp} explained={evs:.3f}", flush=True)

    # ---------------- flags: vision + text ----------------
    flags = json.load(open("assets/flags.json"))
    fp = [Image.open(f"assets/flags/{f['id']}.png").convert("RGB") for f in flags]
    Ef = embed_images(model, preprocess, fp)
    Tf = embed_texts(model, tok, [f"the national flag of {f['name']}" for f in flags])
    Zi, evi = pca_reduce(Ef, 16)
    Zt, evt = pca_reduce(Tf, 16)
    Zf = np.concatenate([Zi, Zt], axis=1)
    Zf = Zf / (np.linalg.norm(Zf, axis=1, keepdims=True) + 1e-9)
    b64, sc, shp = quantise(Zf.astype(np.float32))
    for f in flags:
        with open(f"assets/flags/{f['id']}.webp", "rb") as fh:
            f["img"] = base64.b64encode(fh.read()).decode()
        f.pop("webp_bytes", None)
    bundle["flags"] = {
        "dim": shp[1], "scale": sc, "vecs": b64, "explained": (evi + evt) / 2,
        "split": 16, "items": flags,
    }
    print(f"flags: {shp} img_ev={evi:.3f} txt_ev={evt:.3f}", flush=True)

    # ---------------- people: pure text ----------------
    people = json.load(open("assets/people.json"))
    Ep = embed_texts(model, tok, [p["prompt"] for p in people])
    Zp, evp = pca_reduce(Ep, 24)
    b64, sc, shp = quantise(Zp)
    bundle["people"] = {"dim": shp[1], "scale": sc, "vecs": b64, "explained": evp, "items": people}
    print(f"people: {shp} explained={evp:.3f}", flush=True)

    with open("assets/bundle.json", "w") as f:
        json.dump(bundle, f)
    print("bundle", round(os.path.getsize("assets/bundle.json") / 1024), "KB")

    # ---------- sanity: does the geometry track the known factors? ----------
    def probe(Z, items, key):
        vals = sorted({str(i[key]) for i in items})
        if len(vals) < 2 or len(vals) > 14:
            return None
        idx = {v: [j for j, i in enumerate(items) if str(i[key]) == v] for v in vals}
        within, between = [], []
        cents = {v: Z[ix].mean(0) for v, ix in idx.items() if len(ix) > 1}
        for v, ix in idx.items():
            if len(ix) < 2: continue
            c = cents[v]
            within.append(np.mean([np.dot(Z[j], c) / (np.linalg.norm(c) + 1e-9) for j in ix]))
        cs = list(cents.values())
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                between.append(np.dot(cs[a], cs[b]) /
                               ((np.linalg.norm(cs[a]) * np.linalg.norm(cs[b])) + 1e-9))
        return np.mean(within), np.mean(between)

    print("\nseparability (within-group cos vs between-group cos; bigger gap = easier):")
    for key in ["family", "colour", "fill", "count"]:
        r = probe(Zs, shapes, key)
        if r: print(f"  shapes/{key:8s} within={r[0]:.3f} between={r[1]:.3f}  gap={r[0]-r[1]:.3f}")
    for key in ["continent", "bands"]:
        r = probe(Zf, flags, key)
        if r: print(f"  flags/{key:9s} within={r[0]:.3f} between={r[1]:.3f}  gap={r[0]-r[1]:.3f}")
    for key in ["field", "era", "region"]:
        r = probe(Zp, people, key)
        if r: print(f"  people/{key:8s} within={r[0]:.3f} between={r[1]:.3f}  gap={r[0]-r[1]:.3f}")


if __name__ == "__main__":
    main()
