"""Contemporary business leaders -> CLIP text prompts with interpretable attributes.

Pruned to widely-recognisable names (plus a few niche-but-genuinely-notable ones,
e.g. Amodei, Liang Wenfeng, Dangote) so a player actually has an opinion to give.
A deliberate mix of admired and divisive figures so like/pass varies.

Descriptors are neutral role facts. `polarizing` is kept here ONLY to balance the
roster during curation — it is NOT written to the item or shown in the UI, so it
cannot prime a player's rating. Only adjudicated facts (a conviction) or documented
public positions are stated; never an unproven allegation.
"""
import json

# (name, short role descriptor, industry, region, polarizing[curation-only, not emitted])
P = [
 # --- tech & AI, Americas ---
 ("Elon Musk", "chief executive of Tesla and SpaceX", "tech", "Americas", True),
 ("Jeff Bezos", "founder of Amazon", "tech", "Americas", True),
 ("Mark Zuckerberg", "co-founder and chief executive of Meta", "tech", "Americas", True),
 ("Bill Gates", "co-founder of Microsoft and philanthropist", "tech", "Americas", True),
 ("Sam Altman", "chief executive of OpenAI, which he steered toward a for-profit structure", "tech", "Americas", True),
 ("Dario Amodei", "chief executive of Anthropic; backs strict AI chip export controls", "tech", "Americas", True),
 ("Satya Nadella", "chief executive of Microsoft", "tech", "Americas", False),
 ("Sundar Pichai", "chief executive of Alphabet and Google", "tech", "Americas", False),
 ("Tim Cook", "chief executive of Apple and successor to Steve Jobs", "tech", "Americas", False),
 ("Jensen Huang", "co-founder and chief executive of Nvidia", "tech", "Americas", True),
 ("Lisa Su", "chief executive of AMD", "tech", "Americas", False),
 ("Andy Jassy", "chief executive of Amazon", "tech", "Americas", False),
 ("Peter Thiel", "venture capitalist, co-founder of PayPal and Palantir", "tech", "Americas", True),
 ("Marc Andreessen", "venture capitalist and Netscape co-founder", "tech", "Americas", True),
 ("Travis Kalanick", "co-founder and former chief executive of Uber", "tech", "Americas", True),
 ("Adam Neumann", "co-founder of WeWork", "tech", "Americas", True),
 ("Larry Ellison", "co-founder and chairman of Oracle", "tech", "Americas", True),
 ("Marc Benioff", "co-founder and chief executive of Salesforce", "tech", "Americas", False),
 ("Jack Dorsey", "co-founder of Twitter and Block", "tech", "Americas", True),
 ("Palmer Luckey", "founder of Oculus and Anduril", "tech", "Americas", True),
 ("Brian Chesky", "co-founder and chief executive of Airbnb", "tech", "Americas", False),
 ("Sheryl Sandberg", "former chief operating officer of Meta", "tech", "Americas", True),
 ("Whitney Wolfe Herd", "founder of Bumble", "tech", "Americas", False),
 ("Reed Hastings", "co-founder of Netflix", "tech", "Americas", False),
 ("Mustafa Suleyman", "chief executive of Microsoft AI, co-founder of DeepMind", "tech", "Americas", True),
 ("Elizabeth Holmes", "founder of Theranos, convicted of fraud", "tech", "Americas", True),

 # --- tech & AI, Europe ---
 ("Demis Hassabis", "co-founder and chief executive of Google DeepMind", "tech", "Europe", False),
 ("Daniel Ek", "co-founder and chief executive of Spotify", "tech", "Europe", True),
 ("Pavel Durov", "founder of Telegram", "tech", "Europe", True),

 # --- tech & AI, Asia ---
 ("Narayana Murthy", "co-founder of Infosys", "tech", "Asia", True),
 ("Nandan Nilekani", "co-founder of Infosys and architect of Aadhaar", "tech", "Asia", False),
 ("Azim Premji", "chairman of Wipro and philanthropist", "tech", "Asia", False),
 ("Byju Raveendran", "founder of Byju's", "tech", "Asia", True),
 ("Jack Ma", "co-founder of Alibaba", "tech", "Asia", True),
 ("Zhang Yiming", "founder of ByteDance", "tech", "Asia", True),
 ("Lei Jun", "founder of Xiaomi", "tech", "Asia", False),
 ("Liang Wenfeng", "founder of DeepSeek", "tech", "Asia", True),

 # --- crypto ---
 ("Sam Bankman-Fried", "founder of FTX, convicted of fraud", "crypto", "Americas", True),
 ("Brian Armstrong", "co-founder and chief executive of Coinbase", "crypto", "Americas", True),
 ("Changpeng Zhao", "founder of Binance", "crypto", "Asia", True),
 ("Vitalik Buterin", "co-founder of Ethereum", "crypto", "Europe", True),
 ("Do Kwon", "founder of Terraform Labs", "crypto", "Asia", True),

 # --- finance ---
 ("Warren Buffett", "chairman of Berkshire Hathaway", "finance", "Americas", False),
 ("Jamie Dimon", "chief executive of JPMorgan Chase", "finance", "Americas", True),
 ("Ray Dalio", "founder of Bridgewater Associates", "finance", "Americas", True),
 ("Cathie Wood", "founder of ARK Invest", "finance", "Americas", True),
 ("Ken Griffin", "founder of Citadel", "finance", "Americas", True),
 ("Larry Fink", "co-founder and chief executive of BlackRock", "finance", "Americas", True),
 ("Masayoshi Son", "founder of SoftBank", "finance", "Asia", True),

 # --- retail / consumer / media ---
 ("Howard Schultz", "former chief executive of Starbucks", "retail", "Americas", True),
 ("Indra Nooyi", "former chief executive of PepsiCo", "retail", "Americas", False),
 ("Phil Knight", "co-founder of Nike", "retail", "Americas", False),
 ("Bob Iger", "chief executive of Disney", "media", "Americas", False),
 ("Rupert Murdoch", "media magnate and founder of News Corp", "media", "Americas", True),
 ("Bernard Arnault", "chairman of LVMH", "retail", "Europe", False),

 # --- auto / conglomerate / Africa ---
 ("Mary Barra", "chief executive of General Motors", "auto", "Americas", False),
 ("Richard Branson", "founder of the Virgin Group", "conglomerate", "Europe", False),
 ("Mukesh Ambani", "chairman of Reliance Industries", "conglomerate", "Asia", True),
 ("Gautam Adani", "chairman of the Adani Group", "conglomerate", "Asia", True),
 ("Aliko Dangote", "founder of the Dangote Group", "conglomerate", "Africa", False),
]


def main():
    out = []
    for i, (name, desc, industry, region, _polarizing) in enumerate(P):
        out.append({
            "id": f"p{i:03d}",
            "name": name,
            "desc": desc,
            "industry": industry,
            "region": region,
            "prompt": f"a portrait photograph of {name}, {desc}, a {industry} industry leader",
        })
    with open("assets/people.json", "w") as f:
        json.dump(out, f)
    from collections import Counter
    print(len(out), "business leaders")
    print("industry  ", Counter(o["industry"] for o in out))
    print("region    ", Counter(o["region"] for o in out))
    print("polarizing(curation-only, not emitted)", Counter(p[4] for p in P))


if __name__ == "__main__":
    main()
