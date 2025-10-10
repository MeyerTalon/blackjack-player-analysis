import argparse
from .analyze import analyze_csv_against_pbs
from .house_edge import build_house_edge_report
from .latex_generator import generate_latex_reports

def analyze_with_house_edge(
        csv_path: str,
        out_report_tex="src/reports/pbs_report.tex"
) -> None:
    """
    Run PBS comparison + house-edge reporting and produce one LaTeX report per player.

    Compile with (example for player 0):
        pdflatex src/reports/pbs_report_player_0.tex
    """

    # ---- Run the analysis pipeline ----
    decisions_df, players_summary_df = analyze_csv_against_pbs(csv_path)
    bets_df, deviations_df, session_summary_df = build_house_edge_report(csv_path, decisions_df)
    generate_latex_reports(
        players_summary_df,
        bets_df,
        deviations_df,
        session_summary_df
    )



def main():
    p = argparse.ArgumentParser(description="PBS comparison + house-edge reporting (per-player LaTeX)")
    p.add_argument("csv", help="Path to your game CSV.")
    p.add_argument("--out-report", default="src/reports/pbs_report.tex",
                   help="Basename path used to derive per-player files, e.g., pbs_report_player_<id>.tex")
    args = p.parse_args()

    analyze_with_house_edge(
        csv_path=args.csv,
        out_report_tex=args.out_report
    )

if __name__ == "__main__":
    main()
