"""Contemporary business leaders -> CLIP text prompts with interpretable attributes.

A deliberate mix of the admired-and-uncontroversial and the admired-but-polarizing,
so a user's like/pass signal actually varies instead of everyone being likable.
Descriptors are neutral role facts; controversy is carried by the `polarizing`
flag, not by editorial text. Only adjudicated public facts are stated.
"""
import json

# (name, short role descriptor, industry, region, polarizing)
P = [
 # --- tech, Americas ---
 ("Elon Musk", "chief executive of Tesla and SpaceX", "tech", "Americas", True),
 ("Jeff Bezos", "founder of Amazon", "tech", "Americas", True),
 ("Mark Zuckerberg", "co-founder and chief executive of Meta", "tech", "Americas", True),
 ("Bill Gates", "co-founder of Microsoft and philanthropist", "tech", "Americas", True),
 ("Sam Altman", "chief executive of OpenAI", "tech", "Americas", True),
 ("Satya Nadella", "chief executive of Microsoft", "tech", "Americas", False),
 ("Sundar Pichai", "chief executive of Alphabet and Google", "tech", "Americas", False),
 ("Tim Cook", "chief executive of Apple", "tech", "Americas", False),
 ("Jensen Huang", "co-founder and chief executive of Nvidia", "tech", "Americas", False),
 ("Lisa Su", "chief executive of AMD", "tech", "Americas", False),
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
 ("Ginni Rometty", "former chief executive of IBM", "tech", "Americas", False),
 ("Elizabeth Holmes", "founder of Theranos, convicted of fraud", "tech", "Americas", True),

 # --- crypto ---
 ("Sam Bankman-Fried", "founder of FTX, convicted of fraud", "crypto", "Americas", True),
 ("Brian Armstrong", "co-founder and chief executive of Coinbase", "crypto", "Americas", True),
 ("Changpeng Zhao", "founder of Binance", "crypto", "Asia", True),

 # --- finance, Americas ---
 ("Warren Buffett", "chairman of Berkshire Hathaway", "finance", "Americas", False),
 ("Jamie Dimon", "chief executive of JPMorgan Chase", "finance", "Americas", True),
 ("Ray Dalio", "founder of Bridgewater Associates", "finance", "Americas", True),
 ("Cathie Wood", "founder of ARK Invest", "finance", "Americas", True),
 ("Ken Griffin", "founder of Citadel", "finance", "Americas", True),

 # --- retail / consumer / media / auto, Americas ---
 ("Howard Schultz", "former chief executive of Starbucks", "retail", "Americas", True),
 ("Indra Nooyi", "former chief executive of PepsiCo", "retail", "Americas", False),
 ("Phil Knight", "co-founder of Nike", "retail", "Americas", False),
 ("Bob Iger", "chief executive of Disney", "media", "Americas", False),
 ("Rupert Murdoch", "media magnate and founder of News Corp", "media", "Americas", True),
 ("Mary Barra", "chief executive of General Motors", "auto", "Americas", False),

 # --- Europe ---
 ("Bernard Arnault", "chairman of LVMH", "retail", "Europe", False),
 ("Richard Branson", "founder of the Virgin Group", "conglomerate", "Europe", False),
 ("Daniel Ek", "co-founder and chief executive of Spotify", "tech", "Europe", True),
 ("Pavel Durov", "founder of Telegram", "tech", "Europe", True),
 ("Ola Kallenius", "chief executive of Mercedes-Benz", "auto", "Europe", False),
 ("Belen Garijo", "chief executive of the Merck Group", "industrial", "Europe", False),
 ("Jose Neves", "founder of Farfetch", "retail", "Europe", True),

 # --- Asia ---
 ("Mukesh Ambani", "chairman of Reliance Industries", "conglomerate", "Asia", True),
 ("Gautam Adani", "chairman of the Adani Group", "conglomerate", "Asia", True),
 ("Narayana Murthy", "co-founder of Infosys", "tech", "Asia", True),
 ("Nandan Nilekani", "co-founder of Infosys and architect of Aadhaar", "tech", "Asia", False),
 ("Azim Premji", "chairman of Wipro and philanthropist", "tech", "Asia", False),
 ("Kiran Mazumdar-Shaw", "founder of Biocon", "industrial", "Asia", False),
 ("Falguni Nayar", "founder of Nykaa", "retail", "Asia", False),
 ("Byju Raveendran", "founder of Byju's", "tech", "Asia", True),
 ("Vijay Mallya", "founder of Kingfisher Airlines", "conglomerate", "Asia", True),
 ("Jack Ma", "co-founder of Alibaba", "tech", "Asia", True),
 ("Zhang Yiming", "founder of ByteDance", "tech", "Asia", True),
 ("Masayoshi Son", "founder of SoftBank", "finance", "Asia", True),
 ("Lei Jun", "founder of Xiaomi", "tech", "Asia", False),
 ("Robin Li", "co-founder of Baidu", "tech", "Asia", True),
 ("Akio Toyoda", "chairman of Toyota", "auto", "Asia", False),
]


def main():
    out = []
    for i, (name, desc, industry, region, polarizing) in enumerate(P):
        out.append({
            "id": f"p{i:03d}",
            "name": name,
            "desc": desc,
            "industry": industry,
            "region": region,
            "polarizing": polarizing,
            "prompt": f"a portrait photograph of {name}, {desc}, a {industry} industry leader",
        })
    with open("assets/people.json", "w") as f:
        json.dump(out, f)
    from collections import Counter
    print(len(out), "business leaders")
    print("industry  ", Counter(o["industry"] for o in out))
    print("region    ", Counter(o["region"] for o in out))
    print("polarizing", Counter(o["polarizing"] for o in out))


if __name__ == "__main__":
    main()
