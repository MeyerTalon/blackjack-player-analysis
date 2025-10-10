import math
import os
import pandas as pd
from datetime import datetime
import re
import subprocess
import shutil
from pathlib import Path

# ---- Helpers for LaTeX ----

def tex_to_pdf_pdflatex(tex_path: str, clean_aux: bool = False, timeout: int = 180) -> str:
    """
    Compile a LaTeX .tex file to PDF in the same directory using pdflatex.

    Args:
        tex_path: Path to the .tex file.
        clean_aux: If True, remove common auxiliary files after a successful build.
        timeout: Seconds allowed for each pdflatex run.

    Returns:
        Absolute path to the generated PDF.

    Raises:
        FileNotFoundError: If the .tex file does not exist.
        ValueError: If the path is not a .tex file.
        RuntimeError: If pdflatex is not found or the compile fails.
    """
    tex_path = Path(tex_path).expanduser().resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"No such file: {tex_path}")
    if tex_path.suffix.lower() != ".tex":
        raise ValueError("Input must be a .tex file")

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex not found on PATH. Install TeX Live/MacTeX/MiKTeX and try again.")

    workdir = tex_path.parent
    pdf_path = workdir / f"{tex_path.stem}.pdf"

    cmd = [
        pdflatex,
        "-halt-on-error",
        "-interaction=nonstopmode",
        tex_path.name,           # run in workdir so outputs land next to .tex
    ]

    try:
        # Run twice to resolve references/toc
        for _ in range(2):
            result = subprocess.run(
                cmd, cwd=workdir, check=True, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
    except subprocess.CalledProcessError as e:
        # Write a .log next to the .tex for easier debugging
        (workdir / f"{tex_path.stem}.build.log").write_text(e.stdout or "", encoding="utf-8")
        raise RuntimeError(f"LaTeX build failed. See {tex_path.stem}.build.log for details.") from e

    if not pdf_path.exists():
        raise RuntimeError(f"Expected PDF not found at {pdf_path}")

    if clean_aux:
        for ext in (".aux", ".log", ".toc", ".out", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz"):
            p = workdir / f"{tex_path.stem}{ext}"
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass  # ignore cleanup errors

    return str(pdf_path)



def _tex_escape_text(s) -> str:
    """Escape user/text fields; DO NOT escape backslashes so \% etc. remain intact."""
    if s is None:
        return ""
    s = str(s)
    return (s
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
        return _tex_escape_text(str(x))

def _int(x):
    try:
        return f"{int(x)}"
    except Exception:
        return ""

def _sanitize_label(s: str) -> str:
    """
    Make a LaTeX-safe label token.
    Keep letters, numbers, colon, dot, underscore, and dash.
    Remove everything else; collapse repeats.
    """
    t = re.sub(r'[^A-Za-z0-9:_.-]+', '-', str(s))
    t = re.sub(r'-{2,}', '-', t).strip('-')
    # labels cannot be empty; fall back
    return t or "label"


def _tabular_from_rows(caption: str, headers: list[str], rows: list[list[str]], label: str) -> str:
    """Assumes cell values in `rows` are already LaTeX-safe; only escape headers/caption/label."""
    cols_spec = " | ".join(["l"] + ["r"] * (len(headers) - 1))
    head = " & ".join([_tex_escape_text(h) for h in headers]) + r" \\ \hline"
    body_lines = []
    for r in rows:
        body_lines.append(" & ".join(r) + r" \\")
    body = "\n".join(body_lines)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{{_tex_escape_text(caption)}}}
\label{{{_sanitize_label(label)}}}
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
        # Custom col spec for deviations table
        if "Explanation" in headers:
            cols_spec = "l | r | r | r | r | r | p{5cm}"
        head = " & ".join([_tex_escape_text(h) for h in headers]) + r" \\ \hline"
        body_lines = []
        for r in rows:
            body_lines.append(" & ".join(r) + r" \\")
        body = "\n".join(body_lines)
        return rf"""
\begin{{longtable}}{{{cols_spec}}}
\caption{{{_tex_escape_text(caption)}}}\label{{{_sanitize_label(label)}}}\\
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

def format_time(time_str: str) -> str:
    """Convert 'YYYY-MM-DDTHH:MM:SSZ' to 'HH:MM:SS'."""
    return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ").strftime("%H:%M:%S") + " UTC"

def generate_latex_reports(
        players_summary_df,
        bets_df,
        deviations_df,
        session_summary_df,
        out_report_tex=f"src/reports/{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}_analysis_report.tex"
):
    # ---- Prepare basename for per-player TEX files ----
    base_dir = os.path.dirname(out_report_tex) or "."
    os.makedirs(base_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(out_report_tex))[0]  # e.g., "pbs_report"
    # We'll write: {base_dir}/{base_name}_player_{pid}.tex

    # ---- Generate one LaTeX report per player ----
    player_ids = sorted(session_summary_df["player_id"].unique().tolist())
    for pid in player_ids:
        # Filter per-player data
        ss_row = session_summary_df[session_summary_df["player_id"] == pid].iloc[0]
        style_row = players_summary_df[players_summary_df["player_id"] == pid].iloc[0]
        bets_g = bets_df[bets_df["player_id"] == pid].sort_values("timestamp_utc")
        devs_g = deviations_df[deviations_df["player_id"] == pid].sort_values(
            ["timestamp_utc", "hand_index", "stage"]
        ) if not deviations_df.empty else pd.DataFrame()

        # ---- Build LaTeX content for this player ----
        preamble = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parskip}}{{6pt}}
\setlength{{\parindent}}{{0pt}}
\title{{Blackjack PBS Analysis \& House-Edge Report \\ {{\large Player {int(pid)}}}}}
\date{{{datetime.now().strftime("%B %d, %Y")}}}
\begin{{document}}
\maketitle""".strip()

        # Session summary (single row) + style (single row)
        sess_headers = [
            "Rounds", "Total Wager", "Total Result",
            "Realized Edge", "Expected (PBS)", "Expected (Actual)"
        ]
        sess_rows = [[
            _int(ss_row["rounds"]),
            _money(ss_row["total_wager"]),
            f"{float(ss_row['total_result']):+,.2f}",
            _pct(ss_row["realized_house_edge_pct"]),
            _pct(ss_row["expected_house_edge_pbs_pct"]),
            _pct(ss_row["expected_house_edge_actual_pct"]),
        ]]
        session_table = _tabular_from_rows(
            "Session Summary",
            sess_headers,
            sess_rows,
            f"tab:session_p{int(pid)}"
        )

        style_headers = ["Decisions", "Total Deviations", "Conservative Deviations", "Aggressive Deviations", "Style"]
        style_rows = [[
            _int(style_row["decisions"]),
            _int(style_row["deviations"]),
            _int(style_row["conservative_deviations"]),
            _int(style_row["aggressive_deviations"]),
            _tex_escape_text(style_row["style"]),
        ]]
        style_table = _tabular_from_rows(
            "PBS Compliance / Style",
            style_headers,
            style_rows,
            f"tab:style_p{int(pid)}"
        )

        # Deviations for the player
        if devs_g is not None and not devs_g.empty:
            d_hdr = ["Time", "Cards", "Up", "Action", "PBS", "Penalty", "Explanation"]
            d_rows = []
            for _, d in devs_g.iterrows():
                penalty_pct = d.get("penalty_pct", None)
                d_rows.append([
                    format_time(_tex_escape_text(d["timestamp_utc"])),
                    _tex_escape_text(d["player_cards"]),
                    _tex_escape_text(d["dealer_upcard"]),
                    _tex_escape_text(d["action_taken"]),
                    _tex_escape_text(d["pbs_action"]),
                    _pct((penalty_pct or 0.0) * 100.0),
                    _tex_escape_text(d.get("explanation", "")),
                ])
            dev_section = (
                    r"\section*{Deviations from Perfect Basic Strategy}" + "\n" +
                    _longtable_from_rows(f"Player {int(pid)} Deviations", d_hdr, d_rows, f"tab:devs_p{int(pid)}")
            )
        else:
            dev_section = r"\section*{Deviations from Perfect Basic Strategy}\par No deviations recorded."

        # Betting history for the player
        hist_section = ""
        if not bets_g.empty:
            hdr = ["Time", "Bet", "Result", "Realized Edge", "Expected (PBS)", "Expected (Actual)"]
            rows = []
            for _, rr in bets_g.iterrows():
                rows.append([
                    format_time(_tex_escape_text(rr["timestamp_utc"])), # 2025-10-10T04:37:31Z
                    _money(rr["base_bet"]),
                    f"{float(rr['result']):+,.2f}",
                    _pct(rr["realized_edge_pct"]),
                    _pct(rr["expected_edge_pbs_pct"]),
                    _pct(rr["expected_edge_actual_pct"]),
                ])
            hist_section = (
                rf"\section*{{Complete Betting History}}" + "\n" +
                _longtable_from_rows(f"Player {int(pid)} Betting History", hdr, rows, f"tab:history_p{int(pid)}")
            )



        # Notes
        notes = r"""
\section*{Notes}
\begin{itemize}
  \item \textbf{Expected (PBS)} is the table’s baseline house edge under perfect basic strategy for the given rules.
  \item \textbf{Expected (Actual)} adds estimated penalties for each deviation from PBS.
  \item \textbf{Realized edge} is computed from outcomes in the CSV and can differ from expectation due to variance.
  \item \textbf{Penalty} refers to the amount of EV the player lost by deviating from PBS. This is equivalent to the increase in house edge.
  \item Penalties are approximate and expressed as percentage of the base bet per round; they are meant for directional analysis.
\end{itemize}
"""

        ending = r"\end{document}"

        tex = "\n".join([
            preamble,
            r"\section*{Overview}",
            session_table,
            style_table,
            dev_section,
            hist_section,
            notes,
            ending
        ])

        # ---- Write this player's .tex file ----
        out_tex_path = os.path.join(base_dir, f"{base_name}_player_{int(pid)}.tex")
        with open(out_tex_path, "w", encoding="utf-8") as f:
            f.write(tex)

        # ---- Compile this player's .tex file to .pdf ----
        pdf_path = tex_to_pdf_pdflatex(out_tex_path)
        print(f"Wrote LaTeX and PDF report for player {pid} to {out_tex_path}")

        # --- Remove extraneous files generated in the compilation process ----
        if pdf_path and os.path.exists(pdf_path):
            root, _ = os.path.splitext(out_tex_path)
            for ext in (".out", ".log", ".aux", ".toc"):
                try:
                    os.remove(root + ext)
                except FileNotFoundError:
                    pass

