"""Procedurally generate shape specimens as SVG with known generative factors."""
import json, math, random, os

random.seed(20260826)

W = H = 224
CX = CY = 112
BG = "#F2F1EC"

HUES = {
    "red": 0, "orange": 28, "amber": 45, "lime": 82, "green": 132,
    "teal": 168, "cyan": 190, "blue": 215, "indigo": 246, "violet": 272,
    "magenta": 302, "pink": 335,
}
ACHRO = {"black": "#1B1B1E", "grey": "#8A8A90", "white": "#FBFBFB"}

FAMILIES = ["circle", "square", "triangle", "pentagon", "hexagon", "star5",
            "star6", "cross", "diamond", "crescent", "ring", "arrow"]
FILLS = ["solid", "outline", "dots", "stripes"]


def hsl(h, s, l):
    """HSL -> hex. cairosvg does not parse hsl() notation, so resolve it here."""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l / 100.0, s / 100.0)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255 + .5), int(g * 255 + .5), int(b * 255 + .5))


def poly(n, r, cx, cy, rot=0.0, start=-math.pi / 2):
    pts = []
    for i in range(n):
        a = start + rot + 2 * math.pi * i / n
        pts.append(f"{cx + r*math.cos(a):.2f},{cy + r*math.sin(a):.2f}")
    return " ".join(pts)


def star(n, r_out, r_in, cx, cy, rot=0.0):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = -math.pi / 2 + rot + math.pi * i / n
        pts.append(f"{cx + r*math.cos(a):.2f},{cy + r*math.sin(a):.2f}")
    return " ".join(pts)


def shape_path(fam, r, cx, cy, rot):
    """Return an SVG element string for one shape primitive."""
    if fam == "circle":
        return f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" {{STYLE}}/>'
    if fam == "ring":
        return (f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
                f'stroke="{{STROKE}}" stroke-width="{max(4, r*0.30):.2f}"/>')
    if fam == "square":
        return f'<polygon points="{poly(4, r, cx, cy, rot + math.pi/4)}" {{STYLE}}/>'
    if fam == "diamond":
        return f'<polygon points="{poly(4, r, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "triangle":
        return f'<polygon points="{poly(3, r, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "pentagon":
        return f'<polygon points="{poly(5, r, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "hexagon":
        return f'<polygon points="{poly(6, r, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "star5":
        return f'<polygon points="{star(5, r, r*0.42, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "star6":
        return f'<polygon points="{star(6, r, r*0.55, cx, cy, rot)}" {{STYLE}}/>'
    if fam == "cross":
        a = r * 0.34
        b = r * 0.95
        pts = [(-a, -b), (a, -b), (a, -a), (b, -a), (b, a), (a, a),
               (a, b), (-a, b), (-a, a), (-b, a), (-b, -a), (-a, -a)]
        c, s = math.cos(rot), math.sin(rot)
        pp = " ".join(f"{cx + x*c - y*s:.2f},{cy + x*s + y*c:.2f}" for x, y in pts)
        return f'<polygon points="{pp}" {{STYLE}}/>'
    if fam == "arrow":
        pts = [(-r*.35, r*.85), (-r*.35, -r*.15), (-r*.8, -r*.15),
               (0, -r*.9), (r*.8, -r*.15), (r*.35, -r*.15), (r*.35, r*.85)]
        c, s = math.cos(rot), math.sin(rot)
        pp = " ".join(f"{cx + x*c - y*s:.2f},{cy + x*s + y*c:.2f}" for x, y in pts)
        return f'<polygon points="{pp}" {{STYLE}}/>'
    if fam == "crescent":
        off = r * 0.42
        return (f'<path d="M {cx} {cy-r:.2f} A {r:.2f} {r:.2f} 0 1 0 {cx} {cy+r:.2f} '
                f'A {r*1.02:.2f} {r*1.02:.2f} 0 1 1 {cx} {cy-r:.2f} Z" '
                f'transform="rotate({math.degrees(rot):.1f} {cx} {cy})" {{STYLE}}/>')
    raise ValueError(fam)


def build(fam, fill, colour_name, count, rot, scale, idx):
    if colour_name in ACHRO:
        base = ACHRO[colour_name]
        light = base
        chroma = "achromatic"
    else:
        h = HUES[colour_name]
        base = hsl(h, 72, 48)
        light = hsl(h, 72, 48)
        chroma = "chromatic"

    defs = ""
    if fill == "solid":
        style = f'fill="{base}"'
        stroke = base
    elif fill == "outline":
        style = f'fill="none" stroke="{base}" stroke-width="7"'
        stroke = base
    elif fill == "dots":
        pid = f"d{idx}"
        defs = (f'<pattern id="{pid}" width="9" height="9" patternUnits="userSpaceOnUse">'
                f'<circle cx="4.5" cy="4.5" r="2.6" fill="{base}"/></pattern>')
        style = f'fill="url(#{pid})" stroke="{base}" stroke-width="2.5"'
        stroke = base
    else:  # stripes
        pid = f"s{idx}"
        defs = (f'<pattern id="{pid}" width="10" height="10" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)">'
                f'<rect width="5" height="10" fill="{base}"/></pattern>')
        style = f'fill="url(#{pid})" stroke="{base}" stroke-width="2.5"'
        stroke = base

    body = []
    if count == 1:
        r = 74 * scale
        el = shape_path(fam, r, CX, CY, rot)
        body.append(el.replace("{STYLE}", style).replace("{STROKE}", stroke))
    else:
        r = 34 * scale
        for k, (dx, dy) in enumerate([(-52, -30), (52, -30), (0, 48)]):
            el = shape_path(fam, r, CX + dx, CY + dy, rot)
            body.append(el.replace("{STYLE}", style).replace("{STROKE}", stroke))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
           f'<rect width="{W}" height="{H}" fill="{BG}"/>'
           f'<defs>{defs}</defs>{"".join(body)}</svg>')

    return svg, {
        "family": fam, "fill": fill, "colour": colour_name, "chroma": chroma,
        "count": count, "tilted": abs(rot) > 0.05, "big": scale > 0.95,
        "curved": fam in ("circle", "ring", "crescent"),
        "pointy": fam in ("star5", "star6", "triangle", "arrow", "diamond"),
    }


def main():
    os.makedirs("assets/shapes", exist_ok=True)
    colours = list(HUES) + list(ACHRO)
    items = []
    seen = set()
    target = 240
    tries = 0
    while len(items) < target and tries < 20000:
        tries += 1
        fam = random.choice(FAMILIES)
        fill = random.choice(FILLS)
        col = random.choice(colours)
        count = random.choice([1, 1, 1, 3])
        rot = random.choice([0.0, 0.0, random.uniform(0.15, 0.9)])
        scale = random.choice([1.0, 1.0, 0.72])
        key = (fam, fill, col, count, round(rot, 1), scale)
        if key in seen:
            continue
        seen.add(key)
        idx = len(items)
        svg, meta = build(fam, fill, col, count, rot, scale, idx)
        meta["id"] = f"s{idx:03d}"
        meta["svg"] = svg
        items.append(meta)

    with open("assets/shapes.json", "w") as f:
        json.dump(items, f)
    print(f"wrote {len(items)} shapes")
    fams = {}
    for it in items:
        fams[it["family"]] = fams.get(it["family"], 0) + 1
    print(fams)


if __name__ == "__main__":
    main()
