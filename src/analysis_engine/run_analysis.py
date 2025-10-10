import argparse
import pandas as pd
from .analyze import analyze_csv_against_pbs
from .house_edge import build_house_edge_report

def analyze_with_house_edge(csv_path: str,
                            out_decisions="src/reports/pbs_decisions.csv",
                            out_summary="src/reports/pbs_players_summary.csv",
                            out_bets="src/reports/pbs_bets.csv",
                            out_deviations="src/reports/pbs_deviations.csv",
                            out_session="src/reports/pbs_session_summary.csv",
                            out_report_tex="src/reports/pbs_report.tex"):
    """
    Run PBS comparison + house-edge reporting and produce a LaTeX report.

    Compile with:
        pdflatex src/reports/pbs_report.tex
    """
    import math

    # ---- Run the existing analysis pipeline (unchanged) ----
    decisions_df, players_summary_df = analyze_csv_against_pbs(csv_path)
    bets_df, deviations_df, session_summary_df = build_house_edge_report(csv_path, decisions_df)

    # ---- Write the CSV artifacts (unchanged) ----
    decisions_df.to_csv(out_decisions, index=False)
    players_summary_df.to_csv(out_summary, index=False)
    bets_df.to_csv(out_bets, index=False)
    if not deviations_df.empty:
        deviations_df.to_csv(out_deviations, index=False)
    session_summary_df.to_csv(out_session, index=False)

    # ---- Helpers for LaTeX ----
    def _tex_escape_text(s) -> str:
        """Escape user/text fields; DO NOT escape backslashes so \% etc. remain intact."""
        if s is None:
            return ""
        s = str(s)
        return (s
                # DO NOT .replace('\\', r'\textbackslash{}')  # keep \ intact
                .replace('&', r'\&')
                .replace('%', r'\%')
                .replace('#', r'\#')
                .replace('_', r'\_')
                .replace('{', r'\{')
                .replace('}', r'\}')
                .replace('~', r'\textasciitilde{}')
                .replace('^', r'\textasciicircum{}'))

    def _pct(x):
        """Format numeric as '12.34\\%'. Returns LaTeX-ready string (no further escaping)."""
        if x is None:
            return ""
        try:
            if isinstance(x, str):
                x = float(x)
            if math.isnan(x) or math.isinf(x):
                return ""
        except Exception:
            return ""
        return f"{x:.2f}\\%"

    def _money(x):
        """Return '\$1,234.56'. Returns LaTeX-ready string (no further escaping)."""
        try:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return ""
            return f"\\${float(x):,.2f}"
        except Exception:
            # Fall back to escaped text (e.g., if value is non-numeric label)
            return _tex_escape_text(str(x))

    def _int(x):
        try:
            return f"{int(x)}"
        except Exception:
            return ""

    def _tabular_from_rows(caption: str, headers: list[str], rows: list[list[str]], label: str) -> str:
        """Assumes cell values in `rows` are already LaTeX-safe; only escape headers/caption/label."""
        cols_spec = " | ".join(["l"] + ["r"] * (len(headers) - 1))
        head = " & ".join([_tex_escape_text(h) for h in headers]) + r" \\ \hline"
        body_lines = []
        for r in rows:
            # DO NOT escape again; cells are already formatted/escaped appropriately.
            body_lines.append(" & ".join(r) + r" \\")
        body = "\n".join(body_lines)
        return rf"""
\begin{{table}}[ht]
\centering
\caption{{{_tex_escape_text(caption)}}}
\label{{{_tex_escape_text(label)}}}
\begin{{tabular}}{{{cols_spec}}}
\hline
{head}
{body}
\hline
\end{{tabular}}
\end{{table}}
""".strip()

    def _longtable_from_rows(caption: str, headers: list[str], rows: list[list[str]], label: str) -> str:
        """Assumes cell values in `rows` are already LaTeX-safe; only escape headers/caption/label."""
        cols_spec = " | ".join(["l"] + ["r"] * (len(headers) - 1))
        head = " & ".join([_tex_escape_text(h) for h in headers]) + r" \\ \hline"
        body_lines = []
        for r in rows:
            body_lines.append(" & ".join(r) + r" \\")
        body = "\n".join(body_lines)
        return rf"""
\begin{{longtable}}{{{cols_spec}}}
\caption{{{_tex_escape_text(caption)}}}\label{{{_tex_escape_text(label)}}}\\
\hline
{head}
\endfirsthead
\hline
{head}
\endhead
{body}
\hline
\end{{longtable}}
""".strip()

    # ---- Build LaTeX content ----
    preamble = r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{siunitx}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}
\title{Blackjack PBS Analysis \& House-Edge Report}
\date{}
\begin{document}
\maketitle
"""

    # Session Summary table (per-player)
    sess_headers = [
        "Player", "Rounds", "Total Wager", "Total Result",
        "Realized Edge", "Expected (PBS)", "Expected (Actual)", "Added by Deviations"
    ]
    sess_rows = []
    for _, r in session_summary_df.iterrows():
        sess_rows.append([
            _int(r["player_id"]),
            _int(r["rounds"]),
            _money(r["total_wager"]),                                # \$…
            f"{float(r['total_result']):+,.2f}",                      # plain number; keep as-is
            _pct(r["realized_house_edge_pct"]),                       # …\%
            _pct(r["expected_house_edge_pbs_pct"]),
            _pct(r["expected_house_edge_actual_pct"]),
            _pct(r["expected_penalty_addition_pct"]),
        ])
    session_table = _tabular_from_rows(
        "Session Summary by Player",
        sess_headers,
        sess_rows,
        "tab:session_summary"
    )

    # Players style summary (from PBS decisions)
    style_headers = ["Player", "Decisions", "Deviations", "Conservative", "Aggressive", "Style"]
    style_rows = []
    for _, r in players_summary_df.sort_values("player_id").iterrows():
        style_rows.append([
            _int(r["player_id"]),
            _int(r["decisions"]),
            _int(r["deviations"]),
            _int(r["conservative_deviations"]),
            _int(r["aggressive_deviations"]),
            _tex_escape_text(r["style"]),  # escape text field only here
        ])
    style_table = _tabular_from_rows(
        "Player Style Summary (PBS Compliance)",
        style_headers,
        style_rows,
        "tab:style_summary"
    )

    # Round-by-round betting history per player (longtable)
    hist_sections = []
    if not bets_df.empty:
        for pid, g in bets_df.groupby("player_id"):
            g = g.sort_values("timestamp_utc")
            hdr = ["Time", "Bet", "Result", "Realized Edge", "Expected (PBS)", "Expected (Actual)"]
            rows = []
            for _, rr in g.iterrows():
                rows.append([
                    _tex_escape_text(rr["timestamp_utc"]),
                    _money(rr["base_bet"]),
                    f"{float(rr['result']):+,.2f}",
                    _pct(rr["realized_edge_pct"]),
                    _pct(rr["expected_edge_pbs_pct"]),
                    _pct(rr["expected_edge_actual_pct"]),
                ])
            hist_sections.append(
                rf"\section*{{Player {int(pid)} -- Betting History}}" + "\n" +
                _longtable_from_rows(f"Player {int(pid)} Betting History", hdr, rows, f"tab:history_p{int(pid)}")
            )

    # Deviations table (longtable)
    dev_section = ""
    if not deviations_df.empty:
        d_hdr = ["Player", "Time", "Hand\\#", "Stage", "Cards", "Up", "Action", "PBS", "Penalty", "Note"]
        d_rows = []
        for _, d in deviations_df.sort_values(["player_id","timestamp_utc","hand_index","stage"]).iterrows():
            penalty_pct = d.get("penalty_pct", None)
            d_rows.append([
                _int(d["player_id"]),
                _tex_escape_text(d["timestamp_utc"]),
                _int(d["hand_index"]),
                _tex_escape_text(d["stage"]),
                _tex_escape_text(d["player_cards"]),
                _tex_escape_text(d["dealer_upcard"]),
                _tex_escape_text(d["action_taken"]),
                _tex_escape_text(d["pbs_action"]),
                _pct((penalty_pct or 0.0) * 100.0),
                _tex_escape_text(d.get("explanation", "")),
            ])
        dev_section = (
            r"\section*{Deviations from Perfect Basic Strategy}" + "\n" +
            _longtable_from_rows("All Deviations", d_hdr, d_rows, "tab:deviations")
        )
    else:
        dev_section = r"\section*{Deviations from Perfect Basic Strategy}\par No deviations recorded."

    # Notes
    notes = r"""
\section*{Notes}
\begin{itemize}
  \item \textbf{Expected (PBS)} is the table’s baseline house edge under perfect basic strategy for the given rules.
  \item \textbf{Expected (Actual)} adds estimated penalties for each deviation from PBS.
  \item \textbf{Realized edge} is computed from outcomes in the CSV and can differ from expectation due to variance.
  \item Penalties are approximate and expressed as percentage of the base bet per round; they are meant for directional analysis.
\end{itemize}
"""

    ending = r"\end{document}"

    # Assemble LaTeX
    tex = "\n".join([
        preamble,
        r"\section*{Overview}",
        session_table,
        style_table,
        *hist_sections,
        dev_section,
        notes,
        ending
    ])

    # Write .tex
    with open(out_report_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    # Console output (mirrors prior behavior)
    print(f"Wrote decisions to {out_decisions}")
    print(f"Wrote player summary to {out_summary}")
    print(f"Wrote bets to {out_bets}")
    if not deviations_df.empty:
        print(f"Wrote deviations to {out_deviations}")
    print(f"Wrote session summary to {out_session}")
    print(f"Wrote LaTeX report to {out_report_tex}")


def main():
    p = argparse.ArgumentParser(description="PBS comparison + house-edge reporting")
    p.add_argument("csv", help="Path to your game CSV.")
    p.add_argument("--out-decisions", default="src/reports/pbs_decisions.csv")
    p.add_argument("--out-summary", default="src/reports/pbs_players_summary.csv")
    p.add_argument("--out-bets", default="src/reports/pbs_bets.csv")
    p.add_argument("--out-deviations", default="src/reports/pbs_deviations.csv")
    p.add_argument("--out-session", default="src/reports/pbs_session_summary.csv")
    p.add_argument("--out-report", default="src/reports/pbs_report.tex")
    args = p.parse_args()

    analyze_with_house_edge(
        csv_path=args.csv,
        out_decisions=args.out_decisions,
        out_summary=args.out_summary,
        out_bets=args.out_bets,
        out_deviations=args.out_deviations,
        out_session=args.out_session,
        out_report_tex=args.out_report
    )

if __name__ == "__main__":
    main()
