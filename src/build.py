"""Inline the engine and the embedding bundle into a single self-contained page."""
import json, os, re

tpl = open("game.template.html").read()
lrn = open("learners.js").read()

# strip the node export tail; expose the classes on one namespace object
lrn = re.sub(r"if \(typeof module[\s\S]*$", "", lrn).rstrip()
lrn += """
const CL = { SGDLogistic, FTRLProximal, Prototype, ReplayEWC, AdapterHead,
             PageHinkley, adwinEpsCut, makeLearners, sigmoid, dot, mulberry32 };
"""

bundle = open("assets/bundle.json").read()
# guard against breaking out of the <script type=application/json> block
assert "</script" not in bundle.lower()

out = tpl.replace("/*__LEARNERS__*/", lrn).replace("/*__BUNDLE__*/", bundle)
open("game.html", "w").write(out)
print("game.html", round(os.path.getsize("game.html") / 1024), "KB")
