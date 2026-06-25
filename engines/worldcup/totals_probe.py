"""Probe the international TOTALS market: does the model beat the book on Over/Under 2.5?

Uses the model's own logged edges (data/edge_snapshots.csv: p_model vs de-vigged p_book)
joined to actual results. The model leans Under; this tests whether that lean is a real
edge or just bias. Re-run as more matches settle:
    python -m engines.worldcup.totals_probe
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def wc_calibration():
    """Book-free totals check on past World Cups (no totals odds needed). Tests whether
    the model's Over/Under 2.5 is biased and whether it beats the base rate per match.
    Works wherever results.csv has the WC fixtures (2018, 2022, 2026-so-far)."""
    sys.path.insert(0, str(ROOT))
    import engines.worldcup.predictor as P
    played, _ = P.load_matches(); _, played = P.compute_elo(played)

    def rows_for(y0, y1, asof):
        df = played[(played.tournament=="FIFA World Cup") & (played.date>=y0) & (played.date<=y1)]
        beta = P.fit_goal_model(played[played.date < asof])           # point-in-time
        out = []
        for r in df.itertuples(index=False):
            adv = 0.0 if r.neutral else P.HOME_ADV
            l1, l2 = P.expected_goals(r.elo_h, r.elo_a, beta, adv)
            M = P.score_matrix(l1, l2)
            tot = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
            out.append((float(M[tot>=3].sum()), int((r.home_score+r.away_score)>=3), r.home_score+r.away_score))
        return out

    def ll(p, a): p = min(max(p,1e-6),1-1e-6); return -(a*math.log(p)+(1-a)*math.log(1-p))
    def show(label, rows):
        n=len(rows)
        if not n: print(f"{label}: no matches"); return
        mp=np.mean([r[0] for r in rows]); ar=np.mean([r[1] for r in rows])
        m_ll=np.mean([ll(p,a) for p,a,_ in rows]); b_ll=np.mean([ll(ar,a) for _,a,_ in rows])
        print(f"  {label:20s} n={n:3d}  model P(over) {mp:.3f}  actual {ar:.3f}  avg goals {np.mean([t for *_,t in rows]):.2f}  "
              f"| log-loss model {m_ll:.4f} vs base {b_ll:.4f}  ({'beats' if m_ll<b_ll else 'LOSES'})")

    print("World Cup totals — model vs reality (book-free calibration):")
    a18=rows_for("2018-06-01","2018-07-31",pd.Timestamp("2018-05-01"))
    a22=rows_for("2022-11-01","2022-12-31",pd.Timestamp("2022-10-01"))
    a26=rows_for("2026-06-01","2026-07-31",pd.Timestamp("2026-05-01"))
    show("WC2018", a18); show("WC2022", a22); show("WC2026 (so far)", a26)
    show("WC2018+2022", a18+a22)
    print("\n  Read: model is well-calibrated on average (not Under-biased) but LOSES to the\n"
          "  base rate per match at WC level — bunched teams => near-constant ~0.47 over, no\n"
          "  discrimination. Can't beat base rate => can't beat a sharp book. No totals edge.")

def main():
    e = pd.read_csv(ROOT/"data/edge_snapshots.csv")
    ov = e[e.side=="over25"][["date","home","away","odds","p_book","p_model"]].rename(
            columns={"odds":"odds_over","p_book":"book_over","p_model":"model_over"})
    un = e[e.side=="under25"][["date","home","away","odds"]].rename(columns={"odds":"odds_under"})
    m = ov.merge(un, on=["date","home","away"], how="left").drop_duplicates(["date","home","away"])

    r = pd.read_csv(ROOT/"data/results.csv", parse_dates=["date"])
    r["date"] = r.date.dt.strftime("%Y-%m-%d")
    r["total"] = pd.to_numeric(r.home_score,errors="coerce")+pd.to_numeric(r.away_score,errors="coerce")
    r = r.dropna(subset=["total"])
    m["k"]=m.date+"|"+m.home+"|"+m.away; r["k"]=r.date+"|"+r.home_team+"|"+r.away_team
    m = m.merge(r[["k","total"]], on="k", how="inner")
    m["over"] = (m.total > 2.5).astype(int)
    n = len(m)
    if n == 0:
        print("No settled matches with logged totals edges yet."); return
    print(f"Matches with logged totals edge AND a result: {n}\n")

    def ll(p,a): p=np.clip(p,1e-6,1-1e-6); return -(a*np.log(p)+(1-a)*np.log(1-p))
    base = m.over.mean()
    print("Over/Under 2.5 log-loss (lower=better):")
    print(f"  model {ll(m.model_over,m.over).mean():.4f} | book {ll(m.book_over,m.over).mean():.4f} "
          f"| base-rate {ll(np.full(n,base),m.over).mean():.4f}   (actual over-rate {base:.3f})")
    print(f"\n  mean model P(over) {m.model_over.mean():.3f} | mean book {m.book_over.mean():.3f} | actual {base:.3f}")
    print("  -> model leans", "UNDER" if m.model_over.mean()<base else "OVER")

    m["ms"]=np.where(m.model_over>0.5,"over","under"); m["bs"]=np.where(m.book_over>0.5,"over","under")
    m["as_"]=np.where(m.over==1,"over","under")
    dis = m[m.ms!=m.bs]
    if len(dis):
        print(f"\n  disagree on side: {len(dis)} matches | model right {(dis.ms==dis.as_).mean():.1%} "
              f"| book right {(dis.bs==dis.as_).mean():.1%}")

    def pnl(row):
        if row.model_over > row.book_over: return (row.odds_over-1) if row.over==1 else -1
        return (row.odds_under-1) if row.over==0 else -1
    m["pnl"]=m.apply(pnl,axis=1); m["edge"]=np.abs(m.model_over-m.book_over)
    print(f"\n  follow model's value side: {n} bets, P/L {m.pnl.sum():+.2f}u, "
          f"ROI {m.pnl.sum()/n:+.1%}, win {(m.pnl>0).mean():.1%}")
    for thr in (0.03,0.05,0.08):
        s=m[m.edge>=thr]
        if len(s): print(f"    edge>={thr:.0%}: {len(s):2d} bets, ROI {s.pnl.sum()/len(s):+.1%}, win {(s.pnl>0).mean():.1%}")
    print("\n  NOTE: small live-tournament sample; directional, not proof. Re-run as matches settle.")

if __name__ == "__main__":
    if "--wc-calibration" in sys.argv:
        wc_calibration()
    else:
        main()
