"""Cross-check the JS learners against an independent numpy/scipy implementation."""
import json, subprocess, sys
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
D, N = 12, 25
X = rng.normal(size=(N, D)); X /= np.linalg.norm(X, axis=1, keepdims=True)
w_true = rng.normal(size=D)
Y = (X @ w_true > 0).astype(float)

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

ALPHA, BETA, L1, L2, STEPS = 1.20, 1.0, 0.004, 1.0, 4


def ftrl_numpy(X, Y):
    z = np.zeros(D); n = np.zeros(D); zb = 0.0; nb = 0.0
    w = np.zeros(D); b = 0.0

    def materialise(z, n, zb, nb):
        w = np.where(np.abs(z) <= L1, 0.0,
                     -(z - np.sign(z) * L1) / ((BETA + np.sqrt(n)) / ALPHA + L2))
        b = -zb / ((BETA + np.sqrt(nb)) / ALPHA + L2)
        return w, b

    for x, y in zip(X, Y):
        for _ in range(STEPS):
            w, b = materialise(z, n, zb, nb)
            err = sig(x @ w + b) - y
            g = err * x
            sigma = (np.sqrt(n + g ** 2) - np.sqrt(n)) / ALPHA
            z += g - sigma * w
            n += g ** 2
            sb = (np.sqrt(nb + err ** 2) - np.sqrt(nb)) / ALPHA
            zb += err - sb * b
            nb += err ** 2
    w, b = materialise(z, n, zb, nb)
    return w, b, z, n


def check_argmin(w, z, n):
    """The materialised w must be the exact minimiser of the FTRL objective."""
    eta_inv = (BETA + np.sqrt(n)) / ALPHA

    def obj(v):
        return z @ v + 0.5 * np.sum(eta_inv * v ** 2) + L1 * np.sum(np.abs(v)) + 0.5 * L2 * v @ v

    r = minimize(obj, np.zeros(D), method="Powell",
                 options={"xtol": 1e-12, "ftol": 1e-14, "maxiter": 200000})
    return np.max(np.abs(r.x - w)), obj(r.x), obj(w)


def prototype_numpy(X, Y):
    mp = X[Y == 1].sum(0); mn = X[Y == 0].sum(0)
    cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return np.array([6.0 * (cos(x, mp) - cos(x, mn)) for x in X])


def main():
    w, b, z, n = ftrl_numpy(X, Y)
    gap, o_opt, o_w = check_argmin(w, z, n)
    print(f"FTRL closed form vs numerical argmin:  max|Δw| = {gap:.3e}")
    print(f"  objective at numerical optimum = {o_opt:.9f}")
    print(f"  objective at materialised w    = {o_w:.9f}   (delta {o_w - o_opt:+.2e})")

    proto = prototype_numpy(X, Y)

    payload = {"X": X.tolist(), "Y": Y.tolist()}
    with open("/tmp/xcheck.json", "w") as f:
        json.dump(payload, f)

    js = subprocess.run([ "node", "-e", """
const fs=require('fs'); const L=require('/home/claude/cl/learners.js');
const {X,Y}=JSON.parse(fs.readFileSync('/tmp/xcheck.json'));
const D=X[0].length;
const f=new L.FTRLProximal(D), p=new L.Prototype(D);
for(let i=0;i<X.length;i++){const x=Float64Array.from(X[i]); f.observe(x,Y[i]); p.observe(x,Y[i]);}
f.score(Float64Array.from(X[0]));
console.log(JSON.stringify({w:Array.from(f.w), b:f.b,
  proto: X.map(x=>p.score(Float64Array.from(x)))}));
"""], capture_output=True, text=True)
    if js.returncode:
        print(js.stderr); sys.exit(1)
    out = json.loads(js.stdout)

    dw = np.max(np.abs(np.array(out["w"]) - w))
    db = abs(out["b"] - b)
    dp = np.max(np.abs(np.array(out["proto"]) - proto))
    print(f"\nJS vs numpy, after {N} examples x {STEPS} steps:")
    print(f"  FTRL   max|Δw| = {dw:.3e}   |Δb| = {db:.3e}")
    print(f"  Proto  max|Δscore| = {dp:.3e}")

    ok = gap < 1e-5 and dw < 1e-9 and db < 1e-9 and dp < 1e-9
    print("\n" + ("PASS — implementations agree to floating-point precision" if ok
                  else "FAIL — investigate"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
