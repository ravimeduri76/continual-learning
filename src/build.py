"""Inline the engine and the embedding bundle into a single self-contained page."""
import json, os, re

tpl = open("game.template.html").read()
lrn = open("learners.js").read()

# Reuse learners.js's own export list as the browser namespace, so CL can never
# drift out of sync with what the module exports (this is how TwoClock got dropped).
m = re.search(r"module\.exports\s*=\s*(\{[\s\S]*?\})\s*;", lrn)
assert m, "could not find module.exports in learners.js"
exports_obj = m.group(1)
lrn = re.sub(r"if \(typeof module[\s\S]*$", "", lrn).rstrip()
lrn += "\nconst CL = " + exports_obj + ";\n"

bundle = open("assets/bundle.json").read()
# guard against breaking out of the <script type=application/json> block
assert "</script" not in bundle.lower()

# Feedback sink is configured out-of-band (env var / CI secret), never committed.
cfg = "window.FEEDBACK_ENDPOINT = %s;" % json.dumps(os.environ.get("FEEDBACK_ENDPOINT", ""))

out = (tpl.replace("/*__CONFIG__*/", cfg)
          .replace("/*__LEARNERS__*/", lrn)
          .replace("/*__BUNDLE__*/", bundle))
open("game.html", "w").write(out)
print("game.html", round(os.path.getsize("game.html") / 1024), "KB")
