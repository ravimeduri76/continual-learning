"""Fetch national flags, rasterise them, and derive interpretable metadata."""
import io, json, os, time, urllib.request
import numpy as np
import cairosvg
from PIL import Image
import pycountry

RAW = "https://raw.githubusercontent.com/lipis/flag-icons/main/flags/4x3/{}.svg"

CONTINENT = {}
try:
    import pycountry_convert as pcc
    for c in pycountry.countries:
        try:
            CONTINENT[c.alpha_2] = pcc.convert_continent_code_to_continent_name(
                pcc.country_alpha2_to_continent_code(c.alpha_2))
        except Exception:
            pass
except ImportError:
    pass

# UN member states + a few widely-recognised others, by ISO alpha-2.
CODES = """af al dz ad ao ag ar am au at az bs bh bd bb by be bz bj bt bo ba bw br bn bg bf bi
cv kh cm ca cf td cl cn co km cg cd cr ci hr cu cy cz dk dj dm do ec eg sv gq er ee sz et fj fi
fr ga gm ge de gh gr gd gt gn gw gy ht hn hu is in id ir iq ie il it jm jp jo kz ke ki kw kg la
lv lb ls lr ly li lt lu mg mw my mv ml mt mh mr mu mx fm md mc mn me ma mz mm na nr np nl nz ni
ne ng kp mk no om pk pw pa pg py pe ph pl pt qa ro ru rw kn lc vc ws sm st sa sn rs sc sl sg sk
si sb so za kr ss es lk sd sr se ch sy tj tz th tl tg to tt tn tr tm tv ug ua ae gb us uy uz vu
va ve vn ye zm zw""".split()


def band_orientation(arr):
    """Compare how much colour varies along rows vs columns."""
    a = arr.astype(np.float32)
    row_means = a.mean(axis=1)          # (H, 3)  colour of each row
    col_means = a.mean(axis=0)          # (W, 3)  colour of each column
    row_var = row_means.var(axis=0).sum()
    col_var = col_means.var(axis=0).sum()
    if row_var > col_var * 1.8:
        return "horizontal"
    if col_var > row_var * 1.8:
        return "vertical"
    return "neither"


def palette(arr, k=3):
    """Coarse dominant-colour names by quantised hue/lightness histogram."""
    a = arr.reshape(-1, 3).astype(np.float32) / 255.0
    mx = a.max(axis=1); mn = a.min(axis=1)
    v = mx; s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    r, g, b = a[:, 0], a[:, 1], a[:, 2]
    h = np.zeros_like(v)
    d = mx - mn + 1e-9
    m = (mx == r); h[m] = ((g - b) / d)[m] % 6
    m = (mx == g); h[m] = ((b - r) / d + 2)[m]
    m = (mx == b); h[m] = ((r - g) / d + 4)[m]
    h = h * 60

    names = np.full(h.shape, "other", dtype=object)
    names[(v < .22)] = "black"
    names[(s < .18) & (v >= .22) & (v < .78)] = "grey"
    names[(s < .18) & (v >= .78)] = "white"
    chrom = s >= .18
    for lo, hi, nm in [(0, 18, "red"), (18, 45, "orange"), (45, 70, "yellow"),
                       (70, 165, "green"), (165, 200, "cyan"), (200, 260, "blue"),
                       (260, 320, "purple"), (320, 360, "red")]:
        names[chrom & (h >= lo) & (h < hi)] = nm
    vals, counts = np.unique(names.astype(str), return_counts=True)
    order = np.argsort(-counts)
    total = counts.sum()
    return [(str(vals[i]), float(counts[i] / total)) for i in order[:k] if counts[i] / total > .06]


def main():
    os.makedirs("assets/flags", exist_ok=True)
    out = []
    misses = []
    for i, code in enumerate(CODES):
        try:
            with urllib.request.urlopen(RAW.format(code), timeout=25) as r:
                svg = r.read()
        except Exception as e:
            misses.append((code, str(e)[:40])); continue
        try:
            png224 = cairosvg.svg2png(bytestring=svg, output_width=224, output_height=224,
                                      background_color="white")
            png_disp = cairosvg.svg2png(bytestring=svg, output_width=180, output_height=135,
                                        background_color="white")
        except Exception as e:
            misses.append((code, "render " + str(e)[:30])); continue

        c = pycountry.countries.get(alpha_2=code.upper())
        name = c.name.split(",")[0] if c else code.upper()

        with open(f"assets/flags/{code}.png", "wb") as f:
            f.write(png224)
        im = Image.open(io.BytesIO(png_disp)).convert("RGB")
        buf = io.BytesIO(); im.save(buf, "WEBP", quality=72, method=5)
        disp = buf.getvalue()

        arr = np.array(Image.open(io.BytesIO(png_disp)).convert("RGB"))
        pal = palette(arr)
        out.append({
            "id": code,
            "name": name,
            "continent": CONTINENT.get(code.upper(), "Other"),
            "bands": band_orientation(arr),
            "colours": [p[0] for p in pal],
            "webp_bytes": len(disp),
        })
        with open(f"assets/flags/{code}.webp", "wb") as f:
            f.write(disp)
        if i % 40 == 0:
            print(i, code, name, flush=True)
        time.sleep(0.02)

    with open("assets/flags.json", "w") as f:
        json.dump(out, f)
    print(f"ok={len(out)} missed={len(misses)}")
    if misses:
        print(misses[:12])
    tot = sum(o["webp_bytes"] for o in out)
    print("total display bytes", round(tot / 1024), "KB")


if __name__ == "__main__":
    main()
