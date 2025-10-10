
#!/usr/bin/env python3
"""
Generate one PDF report per player from PBS-style CSV exports.

Inputs (CSV files; headers may vary slightly by export/version):
  - bets:              round-by-round betting activity
  - deviations:        instances where player's action differed from PBS, with explanations if available
  - decisions:         per-hand decisions (used for richer context and fallback fields)
  - players_summary:   per-player aggregates (e.g., total bet, net, house edge)
  - session_summary:   overall session aggregates (used to compute/compare house edge)

Output: A folder of per-player PDF reports.

Example:
    python generate_player_reports.py \
        --bets pbs_bets.csv \
        --deviations pbs_deviations.csv \
        --decisions pbs_decisions.csv \
        --players-summary pbs_players_summary.csv \
        --session-summary pbs_session_summary.csv \
        --outdir out_reports

Dependencies:
    pip install pandas reportlab matplotlib numpy

Design notes:
- ReportLab is used for the PDF; Matplotlib renders charts to PNG which are embedded.
- The script is tolerant to column name variations. Use --print-sample-cols to inspect columns.
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib for figures (do NOT set specific colors here; keep defaults)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ReportLab for PDF composition
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ------------------------
# Column detection helpers
# ------------------------

def pick_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    """Return the first column present in df from candidates; else default."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return default

def must_pick_col(df: pd.DataFrame, candidates: List[str], friendly_name: str) -> str:
    col = pick_col(df, candidates)
    if col is None:
        raise KeyError(f"Required column for '{friendly_name}' not found. Tried: {candidates}. Available: {list(df.columns)}")
    return col

def safe_number(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# ------------------------
# Data loading
# ------------------------

def load_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(p)
    return df

# ------------------------
# Feature engineering
# ------------------------

def compute_bet_time_series(bets_df: pd.DataFrame,
                            player_key: str,
                            player_value,
                            col_round: str,
                            col_bet: Optional[str],
                            col_net: Optional[str],
                            col_balance: Optional[str]) -> pd.DataFrame:
    """Return a sorted per-round series for plotting and table display."""
    sub = bets_df[bets_df[player_key] == player_value].copy()
    if col_round:
        sub[col_round] = pd.to_numeric(sub[col_round], errors="coerce")
        sub = sub.sort_values(by=[col_round, sub.columns[0]]).reset_index(drop=True)
    else:
        sub = sub.reset_index(drop=True)
        sub["_round_idx"] = np.arange(1, len(sub) + 1)
        col_round = "_round_idx"

    # Best-effort numeric casting
    for c in [col_bet, col_net, col_balance]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # If no balance provided, compute a running sum of net results
    if (col_balance is None or col_balance not in sub.columns) and (col_net in sub.columns):
        sub["_bankroll"] = sub[col_net].fillna(0).cumsum()
        col_balance = "_bankroll"

    # Return simple set of columns if present
    keep_cols = [c for c in [col_round, col_bet, col_net, col_balance] if c and c in sub.columns]
    if not keep_cols:
        return sub
    return sub[keep_cols]

def infer_house_edge_for_player(player_row: pd.Series) -> Optional[float]:
    """
    Try typical column names:
      - 'house_edge', 'house_edge_pct', 'edge', 'house_edge_percent'
      - or compute as -expected_value / total_bet, falling back to -net/total_bet
    """
    candidates = ["house_edge", "house_edge_pct", "edge", "house_edge_percent"]
    for c in candidates:
        if c in player_row.index and pd.notna(player_row[c]):
            val = safe_number(player_row[c])
            if val is not None and not np.isnan(val):
                # Normalize: if like 0.012 or 1.2, attempt to detect scale
                if abs(val) > 1.5:  # assume percentage given as like 1.2 (%)
                    return float(val) / 100.0
                return float(val)

    total_bet = safe_number(player_row.get("total_bet", np.nan))
    ev = safe_number(player_row.get("expected_value", np.nan))
    net = safe_number(player_row.get("net", np.nan))
    if not np.isnan(total_bet) and total_bet > 0:
        if not np.isnan(ev):
            return -ev / total_bet
        if not np.isnan(net):
            return -net / total_bet
    return None

def summarize_play_style(deviation_rate: float,
                         avg_bet: float,
                         bet_std: float,
                         edge: Optional[float]) -> str:
    """
    Heuristic summary:
      - Aggressive: high avg bet or high variability
      - Conservative: low avg bet and low variability
      - Frequent deviations: deviation_rate >= 0.15
      - Occasional deviations: 0.05..0.15
      - Rare deviations: < 0.05
      - Edge buckets: high expected losses if edge >= 1.5% (0.015)
    """
    style_bits = []

    # Aggression via variability (coefficient of variation) and absolute average bet level
    if avg_bet > 0:
        cv = bet_std / avg_bet if avg_bet > 0 else 0.0
        if avg_bet >= 2.0 * (bet_std + 1e-9):
            style_bits.append("Consistent staking")
        elif cv >= 1.0:
            style_bits.append("Highly variable stakes")
        elif cv >= 0.5:
            style_bits.append("Moderately variable stakes")
        else:
            style_bits.append("Low bet variability")

        if avg_bet >= 50:
            style_bits.append("High average bet")
        elif avg_bet >= 10:
            style_bits.append("Medium average bet")
        else:
            style_bits.append("Low average bet")

    # Deviation rates
    if deviation_rate >= 0.15:
        style_bits.append("Frequent deviations from PBS")
    elif deviation_rate >= 0.05:
        style_bits.append("Occasional deviations from PBS")
    else:
        style_bits.append("Rare deviations from PBS")

    # Edge
    if edge is not None:
        if edge >= 0.02:
            style_bits.append("High expected losses (house edge ≥ 2%)")
        elif edge >= 0.01:
            style_bits.append("Moderate expected losses (house edge ≈ 1–2%)")
        elif edge > 0:
            style_bits.append("Low expected losses (house edge < 1%)")
        else:
            style_bits.append("Non-positive edge (possible advantage)")

    # Compose sentence
    if style_bits:
        # Emphasize first two most telling bits
        primary = ", ".join(style_bits[:2])
        rest = ", ".join(style_bits[2:])
        if rest:
            return f"{primary}; {rest}."
        return f"{primary}."
    return "No sufficient data to summarize play style."

# ------------------------
# Charts
# ------------------------

def plot_bankroll(series_df: pd.DataFrame,
                  col_round: Optional[str],
                  col_balance: Optional[str],
                  out_png: Path) -> Optional[Path]:
    try:
        if col_balance is None or col_balance not in series_df.columns:
            return None
        x = series_df[col_round] if col_round in series_df.columns else np.arange(1, len(series_df) + 1)
        y = series_df[col_balance].astype(float)
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title("Bankroll over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Bankroll")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return out_png
    except Exception:
        return None

def plot_deviation_types(dev_df: pd.DataFrame,
                         col_dev_type: Optional[str],
                         out_png: Path) -> Optional[Path]:
    try:
        if col_dev_type is None or col_dev_type not in dev_df.columns or dev_df.empty:
            return None
        counts = dev_df[col_dev_type].fillna("Unknown").value_counts()
        plt.figure()
        counts.plot(kind="bar")
        plt.title("Deviations by Type")
        plt.xlabel("Deviation Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return out_png
    except Exception:
        return None

# ------------------------
# PDF composition
# ------------------------

def build_pdf_report(out_pdf: Path,
                     player_name: str,
                     player_id: Optional[str],
                     bets_series: pd.DataFrame,
                     dev_df: pd.DataFrame,
                     player_row: Optional[pd.Series],
                     col_round: Optional[str],
                     col_bet: Optional[str],
                     col_net: Optional[str],
                     col_balance: Optional[str],
                     dev_cols: Dict[str, Optional[str]]):
    doc = SimpleDocTemplate(str(out_pdf), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    body = styles["BodyText"]

    story = []

    title = f"Player Report: {player_name}"
    if player_id and str(player_id) != str(player_name):
        title += f" (ID: {player_id})"
    story.append(Paragraph(title, styles["H1"]))

    # House edge and top-line stats
    total_bet = bets_series[col_bet].sum() if col_bet in bets_series.columns else np.nan
    total_net = bets_series[col_net].sum() if col_net in bets_series.columns else np.nan
    avg_bet = bets_series[col_bet].mean() if col_bet in bets_series.columns else np.nan
    bet_std = bets_series[col_bet].std(ddof=0) if col_bet in bets_series.columns else np.nan

    # Deviation rate
    total_hands = max(len(bets_series), 1)
    dev_rate = (len(dev_df) / total_hands) if total_hands > 0 else 0.0

    # Edge
    edge = infer_house_edge_for_player(player_row) if player_row is not None else None

    style_summary = summarize_play_style(
        deviation_rate=dev_rate,
        avg_bet=float(avg_bet) if not np.isnan(avg_bet) else 0.0,
        bet_std=float(bet_std) if not np.isnan(bet_std) else 0.0,
        edge=edge
    )

    # Top summary table
    summary_data = [
        ["Overall Summary", ""],
        ["Total Bet", f"{total_bet:,.2f}" if not np.isnan(total_bet) else "N/A"],
        ["Total Net (player winnings)", f"{total_net:,.2f}" if not np.isnan(total_net) else "N/A"],
        ["Estimated House Edge", f"{edge*100:.2f}% (session)" if edge is not None else "N/A"],
        ["Deviation Rate", f"{dev_rate*100:.1f}%"],
        ["Style Summary", style_summary],
    ]
    t = Table(summary_data, colWidths=[2.5*inch, 3.75*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Betting history section
    story.append(Paragraph("Betting History (Round-by-Round)", styles["H2"]))

    # Render a chart of bankroll if possible
    charts_dir = out_pdf.parent / "_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    bankroll_png = plot_bankroll(bets_series, col_round, col_balance, charts_dir / f"{player_id or player_name}_bankroll.png")
    if bankroll_png and bankroll_png.exists():
        story.append(Image(str(bankroll_png), width=6.5*inch, height=3.0*inch))
        story.append(Spacer(1, 8))

    # Compact table with key columns
    table_cols = []
    headers = []
    # Round column
    if col_round in bets_series.columns:
        headers.append("Round")
        table_cols.append(bets_series[col_round].astype("Int64").astype(str))
    else:
        headers.append("#")
        table_cols.append(pd.Series(np.arange(1, len(bets_series)+1)).astype(str))

    if col_bet in bets_series.columns:
        headers.append("Bet")
        table_cols.append(bets_series[col_bet].map(lambda v: f"{v:,.2f}" if pd.notna(v) else ""))

    if col_net in bets_series.columns:
        headers.append("Net")
        table_cols.append(bets_series[col_net].map(lambda v: f"{v:,.2f}" if pd.notna(v) else ""))

    if col_balance in bets_series.columns:
        headers.append("Bankroll")
        table_cols.append(bets_series[col_balance].map(lambda v: f"{v:,.2f}" if pd.notna(v) else ""))

    # Assemble rows (limit row count per table chunk)
    table_rows = [headers]
    for i in range(len(bets_series)):
        row = [col.iloc[i] if i < len(col) else "" for col in table_cols]
        table_rows.append(row)

    # Split into multiple tables if large
    CHUNK = 40
    for s in range(0, len(table_rows)-1, CHUNK):
        chunk_rows = [table_rows[0]] + table_rows[1+s:1+s+CHUNK]
        tt = Table(chunk_rows, repeatRows=1, hAlign="LEFT")
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ]))
        story.append(tt)
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # Deviations section
    story.append(Paragraph("Deviations from PBS", styles["H2"]))

    # Deviation chart by type if present
    dev_type_col = dev_cols.get("type")
    dev_chart = plot_deviation_types(dev_df, dev_type_col, charts_dir / f"{player_id or player_name}_devtypes.png")
    if dev_chart and dev_chart.exists():
        story.append(Image(str(dev_chart), width=6.5*inch, height=3.0*inch))
        story.append(Spacer(1, 8))

    # Deviation table with explanations
    d_headers = ["Round", "Your Action", "PBS Suggestion", "Explanation"]
    d_rows = [d_headers]
    col_round_d = dev_cols.get("round")
    col_act = dev_cols.get("action")
    col_pbs = dev_cols.get("pbs")
    col_expl = dev_cols.get("explain")

    if not dev_df.empty:
        display_df = dev_df.copy()
        # Keep concise explanation
        for idx, r in display_df.iterrows():
            round_val = r.get(col_round_d, "")
            your_action = str(r.get(col_act, "")) if col_act in display_df.columns else ""
            pbs_suggestion = str(r.get(col_pbs, "")) if col_pbs in display_df.columns else ""
            explanation = str(r.get(col_expl, "")) if col_expl in display_df.columns else ""
            if isinstance(round_val, float) and round_val.is_integer():
                round_val = int(round_val)
            d_rows.append([str(round_val), your_action, pbs_suggestion, explanation])

    if len(d_rows) == 1:
        story.append(Paragraph("No deviations recorded for this player.", body))
    else:
        CHUNK = 25
        for s in range(1, len(d_rows), CHUNK):
            chunk_rows = [d_rows[0]] + d_rows[s:s+CHUNK]
            td = Table(chunk_rows, repeatRows=1, colWidths=[0.8*inch, 1.4*inch, 1.6*inch, 3.0*inch])
            td.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(td)
            story.append(Spacer(1, 8))

    # Build doc
    doc.build(story)

# ------------------------
# Main flow
# ------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate per-player PDF reports from PBS CSVs.")
    ap.add_argument("--bets", required=True, help="Path to pbs_bets.csv")
    ap.add_argument("--deviations", required=True, help="Path to pbs_deviations.csv")
    ap.add_argument("--decisions", required=False, help="Path to pbs_decisions.csv")
    ap.add_argument("--players-summary", required=True, help="Path to pbs_players_summary.csv")
    ap.add_argument("--session-summary", required=False, help="Path to pbs_session_summary.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for per-player PDFs")
    ap.add_argument("--print-sample-cols", action="store_true", help="Print available columns and exit (for debugging).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bets_df = load_csv(args.bets)
    dev_df = load_csv(args.deviations)
    dec_df = load_csv(args.decisions) if args.decisions else None
    players_df = load_csv(args.players_summary)
    session_df = load_csv(args.session_summary) if args.session_summary else None

    if args.print_sample_cols:
        print("Bets columns:", list(bets_df.columns))
        print("Deviations columns:", list(dev_df.columns))
        if dec_df is not None:
            print("Decisions columns:", list(dec_df.columns))
        print("Players summary columns:", list(players_df.columns))
        if session_df is not None:
            print("Session summary columns:", list(session_df.columns))
        return

    # Identify player keys (prefer player_id; fall back to player or name)
    player_key = None
    for key in ["player_id", "player", "player_name", "name", "playerId"]:
        if key in players_df.columns:
            player_key = key
            break
    if player_key is None:
        raise KeyError(f"Could not find a player identifier column in players_summary. Got: {list(players_df.columns)}")

    # Common columns in bets
    col_round_b = pick_col(bets_df, ["round", "round_id", "round_number", "hand_no", "hand", "index"], default=None)
    col_bet = pick_col(bets_df, ["bet", "bet_amount", "wager", "stake"])
    col_net = pick_col(bets_df, ["net", "net_win", "profit", "delta"])
    col_balance = pick_col(bets_df, ["balance", "bankroll", "cum_net", "cumulative_net"])

    # Columns in deviations
    dev_round_col = pick_col(dev_df, ["round", "round_id", "round_number", "hand_no", "hand"])
    dev_action_col = pick_col(dev_df, ["action_taken", "decision_taken", "your_action", "player_action"])
    dev_pbs_col = pick_col(dev_df, ["pbs_action", "pbs_suggestion", "recommended_action", "optimal_action"])
    dev_type_col = pick_col(dev_df, ["deviation_type", "type", "category"])
    dev_expl_col = pick_col(dev_df, ["explanation", "reason", "note"])
    dev_cols = {"round": dev_round_col, "action": dev_action_col, "pbs": dev_pbs_col, "type": dev_type_col, "explain": dev_expl_col}

    # Build player cache for quick lookup
    players_df = players_df.copy()
    # Normalize types
    if player_key in players_df.columns:
        pass

    # Discover the same player_key in other dfs
    bets_player_key = player_key if player_key in bets_df.columns else pick_col(bets_df, [player_key, "player_id", "player", "player_name", "name"])
    dev_player_key = player_key if player_key in dev_df.columns else pick_col(dev_df, [player_key, "player_id", "player", "player_name", "name"])

    # Iterate players
    for _, prow in players_df.iterrows():
        pid = prow.get(player_key)
        # Best-effort human-readable name
        name = None
        for k in ["player_name", "name", "player"]:
            if k in players_df.columns and pd.notna(prow.get(k)):
                name = str(prow.get(k))
                break
        if name is None:
            name = str(pid)

        # Filter data for this player
        bets_subset = bets_df[bets_df.get(bets_player_key, None) == pid] if bets_player_key else pd.DataFrame(columns=bets_df.columns)
        dev_subset = dev_df[dev_df.get(dev_player_key, None) == pid] if dev_player_key else pd.DataFrame(columns=dev_df.columns)

        # Build time series table
        series_df = compute_bet_time_series(
            bets_subset, bets_player_key, pid, col_round_b, col_bet, col_net, col_balance
        )

        # Compose PDF
        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name).strip() or f"player_{pid}"
        out_pdf = outdir / f"{safe_name}.pdf"

        build_pdf_report(
            out_pdf=out_pdf,
            player_name=name,
            player_id=str(pid),
            bets_series=series_df,
            dev_df=dev_subset,
            player_row=prow,
            col_round=col_round_b or "round",
            col_bet=col_bet or "",
            col_net=col_net or "",
            col_balance=(col_balance if (col_balance and col_balance in series_df.columns) else ("_bankroll" if "_bankroll" in series_df.columns else "")),
            dev_cols=dev_cols
        )

    print(f"Done. Wrote {len(players_df)} reports to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
