"""Transfer / Global success rates per K in the layout of MOTIF's Table 1 (arXiv:2602.13764), without the Params
column: a two-level header (K-shot -> Transfer | Global), the paper's five rows, then our two rows
(E-LoRA + geometry tokens, and + perception steering) and the same two with the motion-aware graph (docs/15).
Transfer = 6 few-shot robot-task pairs x 50 episodes (300); Global = all 18 pairs x 50 episodes (900).
Ours: eval_results/graph_hard_elora_geod{,_k<K>}{,_all18,_cp_300,_cp_900,...}/SUMMARY.json, motion-aware graph:
eval_results/abl_tgraph_k<K>{,_cp_300,_all18,_cp_900}; "-" where a run has not finished. Best value per column in bold. Writes docs/table_global.tex and prints markdown. Re-run as results land.

    python eval/make_global_table.py
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KS = [1, 3, 5, 10, 50]
# MOTIF paper, Table 1 (Transfer, Global) in %, K = 1, 3, 5, 10, 50
PAPER = {
    "Diffusion Policy": [(15.67, 27.00), (27.00, 31.33), (29.33, 30.44), (38.00, 36.11), (46.00, 38.00)],
    "HPT$^*$":         [(10.00, 21.11), (15.00, 23.44), (17.33, 26.33), (26.33, 29.67), (44.33, 35.11)],
    "$\\pi_0^*$":       [(33.33, 45.78), (38.33, 48.22), (45.67, 52.67), (56.33, 59.56), (67.67, 62.89)],
    "GR00T N1$^*$":    [(21.67, 43.67), (33.00, 50.67), (35.00, 50.44), (44.67, 50.78), (57.67, 57.44)],
    "MOTIF":           [(36.00, 55.78), (48.33, 55.33), (54.33, 60.44), (60.33, 60.44), (75.00, 66.00)],
}
XVLA = "X-VLA-0.9B (fine-tuned, soft prompts)"   # measured with our evaluator: eval_results/xvla_maniskill_k<K>{,_all18}
OURS = [("Ours (E-LoRA + geometry tokens)", False), ("Ours + perception steering", True)]
TGRAPH = [("Ours, motion-aware graph", False), ("Ours, motion-aware graph + perception steering", True)]
MISSING = "-"


def xvla(k):
    t = REPO / "eval_results" / f"xvla_maniskill_k{k}" / "SUMMARY.json"
    g = REPO / "eval_results" / f"xvla_maniskill_k{k}_all18" / "SUMMARY.json"
    tr = json.load(open(t))["successes"] / 3 if t.exists() else None
    gl = json.load(open(g))["successes"] / 9 if g.exists() and json.load(open(g))["episodes"] == 900 else None
    return tr, gl


def ours(k, steer, tgraph=False):
    base = f"abl_tgraph_k{k}" if tgraph else "graph_hard_elora_geod" if k == 5 else f"graph_hard_elora_geod_k{k}"
    if steer:
        g = REPO / "eval_results" / f"{base}_cp_900" / "SUMMARY.json"
        t = REPO / "eval_results" / (f"{base}_percept_cp_300" if k == 5 and not tgraph else f"{base}_cp_300") / "SUMMARY.json"
    else:
        g = REPO / "eval_results" / f"{base}_all18" / "SUMMARY.json"
        t = REPO / "eval_results" / base / "SUMMARY.json"
    tr = json.load(open(t))["successes"] / 3 if t.exists() else None
    gl = None
    if g.exists():
        d = json.load(open(g))
        if d["episodes"] == 900:
            gl = d["successes"] / 9
    return tr, gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs/table_global.tex")
    args = ap.parse_args()
    rows = [(name, [v for pair in vals for v in pair]) for name, vals in PAPER.items()]
    rows.append((XVLA, [v for k in KS for v in xvla(k)]))
    rows += [(label, [v for k in KS for v in ours(k, steer)]) for label, steer in OURS]
    rows += [(label, [v for k in KS for v in ours(k, steer, tgraph=True)]) for label, steer in TGRAPH]
    ncol = 2 * len(KS)
    best = [max((r[1][c] for r in rows if r[1][c] is not None), default=None) for c in range(ncol)]

    def tex_cell(v, c):
        if v is None:
            return MISSING
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if v == best[c] else s

    def md_cell(v, c):
        if v is None:
            return MISSING
        s = f"{v:.2f}"
        return f"**{s}**" if v == best[c] else s

    md = ["| Method | " + " | ".join(f"{k}-shot T | {k}-shot G" for k in KS) + " |", "|---|" + "---|" * ncol]
    tex = []
    for name, vals in rows:
        label = name.replace('$', '').replace('^*', '*').replace('\\pi_0', 'π0')
        md.append(f"| {label} | " + " | ".join(md_cell(v, c) for c, v in enumerate(vals)) + " |")
        tex.append(f"{name} & " + " & ".join(tex_cell(v, c) for c, v in enumerate(vals)) + " \\\\")
    tex.insert(len(PAPER) + 1 + len(OURS), "\\midrule")
    tex.insert(len(PAPER) + 1, "\\midrule")
    tex.insert(len(PAPER), "\\midrule")
    body = "\n".join(tex)
    head1 = "Method & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{k}-shot}}" for k in KS) + " \\\\"
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(KS)))
    head2 = "& " + " & ".join("Transfer & Global" for _ in KS) + " \\\\"
    doc = f"""% Transfer / Global success rate (%) vs. number of target demonstrations K, layout of MOTIF Table 1 (no Params column).
% Transfer = 6 few-shot robot-task pairs x 50 episodes (300); Global = all 18 pairs x 50 episodes (900).
% Paper rows copied from MOTIF (arXiv:2602.13764) Table 1; X-VLA and ours generated by eval/make_global_table.py from
% eval_results/*/SUMMARY.json. "{MISSING}" = not evaluated yet. Best per column in bold. Requires booktabs.
\\begin{{table*}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{l {"cc " * len(KS)}}}
\\toprule
{head1}
{cmid}
{head2}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Multi-task cross-embodiment transfer results (simulation). Success rate (\\%) as a function of the number $K\\in\\{{1,3,5,10,50\\}}$ of target demonstrations. Transfer is the average over the six few-shot target robot--task pairs, Global the average over all 18 pairs (50 episodes each). The upper block is reported by MOTIF; the X-VLA row (fine-tuned from X-VLA-Pt with its own recipe, absolute end-effector control) and the lower blocks are measured with our evaluator under the same protocol; the last block replaces the static kinematic graph of the tokenizer by the motion-aware graph. $^*$ denotes methods pre-trained on large-scale robot datasets. ``{MISSING}'': not evaluated. Best value per column in bold.}}
\\label{{tab:global}}
\\end{{table*}}
"""
    args.out.write_text(doc)
    print("\n".join(md))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
