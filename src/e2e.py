"""Play the whole game headlessly, capture errors and screenshots."""
import sys, pathlib
from playwright.sync_api import sync_playwright

body = open("game.html").read()
pathlib.Path("preview.html").write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "</head><body>" + body + "</body></html>")

MODE = sys.argv[1] if len(sys.argv) > 1 else "shapes"
THEME = sys.argv[2] if len(sys.argv) > 2 else "light"

errors, logs = [], []
with sync_playwright() as pw:
    br = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium"
                            if pathlib.Path("/opt/pw-browsers/chromium").exists() else None)
    pg = br.new_page(viewport={"width": 1300, "height": 1000},
                     color_scheme="dark" if THEME == "dark" else "light")
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: (logs.append(m.text) if m.type in ("error", "warning") else None))
    pg.goto("file://" + str(pathlib.Path("preview.html").resolve()))
    pg.wait_for_timeout(900)

    pg.screenshot(path=f"shots/00-menu-{THEME}.png", full_page=True)
    pg.click(f'.mode[data-mode="{MODE}"]')
    pg.wait_for_timeout(350)
    pg.screenshot(path=f"shots/01-q1-{MODE}-{THEME}.png", full_page=True)

    def answer_round(n, pattern):
        for k in range(n):
            btn = '[data-y="1"]' if pattern(k) else '[data-y="0"]'
            pg.wait_for_selector(btn, timeout=4000)
            pg.click(btn)
            pg.wait_for_timeout(90)

    answer_round(9, lambda k: k % 3 != 2)          # act 1
    pg.wait_for_timeout(200)
    pg.screenshot(path=f"shots/02-test1-{MODE}-{THEME}.png", full_page=True)
    answer_round(6, lambda k: k % 2 == 0)          # test 1
    pg.wait_for_timeout(300)
    pg.screenshot(path=f"shots/03-result1-{MODE}-{THEME}.png", full_page=True)

    pg.click("#go"); pg.wait_for_timeout(220)
    pg.screenshot(path=f"shots/04-drift-{MODE}-{THEME}.png", full_page=True)
    pg.click("#go"); pg.wait_for_timeout(220)

    answer_round(8, lambda k: k % 2 == 1)          # act 2, a different rule
    answer_round(6, lambda k: k % 3 == 0)          # test 2
    pg.wait_for_timeout(400)
    pg.screenshot(path=f"shots/05-final-{MODE}-{THEME}.png", full_page=True)

    # exercise the dial controls
    pg.eval_on_selector("#d-gam", "el => { el.value = 0.85; el.dispatchEvent(new Event('input')); }")
    pg.wait_for_timeout(200)
    pg.check("#d-ad")
    pg.wait_for_timeout(300)
    pg.screenshot(path=f"shots/06-dials-{MODE}-{THEME}.png", full_page=True)

    table = pg.eval_on_selector_all(".res tbody tr",
        "rows => rows.map(r => Array.from(r.cells).map(c => c.innerText.trim()))")
    br.close()

print("page errors:", errors or "none")
print("console warn/err:", [l for l in logs if "font" not in l.lower()][:6] or "none")
print("\nfinal table with adapter on:")
for r in table:
    print("   " + " | ".join(r))
sys.exit(1 if errors else 0)
