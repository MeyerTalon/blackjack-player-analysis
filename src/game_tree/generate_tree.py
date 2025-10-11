"""
Graphviz visualization for blackjack decision trees.

This module renders the decision/chance tree produced by `build_tree` into a
Graphviz diagram (SVG/PNG/PDF). It supports depth/node limits, pruning tiny
chance branches, and an option to keep only the best-EV path at decisions.
"""

from graphviz import Digraph
from itertools import count
from typing import Dict, Any, Optional
from .blackjack_tree import build_tree, Rules
import argparse
from typing import List

def visualize_tree(
    tree: Dict[str, Any],
    out_path: str = "blackjack_tree",
    fmt: str = "svg",
    max_depth: int = 6,
    max_nodes: int = 2000,
    collapse_chance_below_prob: float = 0.001,
    best_path_only: bool = False,
) -> str:
    """
    Render a blackjack tree (from build_tree(...)) using Graphviz.

    Args:
        tree: The root dictionary returned by build_tree(...)
        out_path: Output path *without* extension (Graphviz adds it)
        fmt: 'svg' | 'png' | 'pdf'
        max_depth: stop expanding deeper than this (keeps graphs readable)
        max_nodes: hard cap on total nodes to add
        collapse_chance_below_prob: for chance nodes, hide branches with prob < threshold
        best_path_only: at decision nodes, keep only the child with the highest EV

    Returns:
        The full file path produced by Graphviz (e.g., 'blackjack_tree.svg').
    """
    dot = Digraph("blackjack_tree", format=fmt)
    dot.attr(rankdir="TB", fontsize="10", labelloc="t")
    dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="9")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    nid = count(0)
    nodes_added = 0

    def fmt_ev(x: Optional[float]) -> str:
        """Formats an EV value to four decimals (default 0.0000)."""
        return f"{x:.4f}" if isinstance(x, (int, float)) else "0.0000"

    def node_label(n: Dict[str, Any]) -> str:
        """Builds a multi-line label describing a node."""
        t = n.get("node")
        info = n.get("info", {})
        ev = fmt_ev(n.get("ev"))
        if t == "decision":
            total = info.get("total", "")
            soft = info.get("soft", False)
            hand = info.get("hand", [])
            dealer = info.get("dealer_up", "")
            return f"DECISION\nhand={hand}\ntotal={total}{' (soft)' if soft else ''}\ndealer={dealer}\nEV*={ev}"
        if t == "chance":
            a = info.get("action", "chance")
            return f"CHANCE\n{a}\nEV={ev}"
        # terminal
        return f"TERMINAL\n{info}\nEV={ev}"

    def node_color(n: Dict[str, Any]) -> str:
        """Returns a fill color based on node type."""
        t = n.get("node")
        if t == "decision": return "#e3f2fd"  # light blue
        if t == "chance":   return "#fff3e0"  # light orange
        return "#e8f5e9"                 # light green (terminal)

    def best_child(children) -> str:
        """Selects the child with the highest EV."""
        # Choose the child with max EV (for chance nodes we already have 'ev')
        best = None
        best_val = float("-inf")
        for c in children:
            v = c.get("ev", 0.0)
            if v > best_val:
                best_val = v
                best = c
        return best

    def add_subtree(
            n: Dict[str, Any],
            depth: int,
            parent_id:
            Optional[str],
            edge_label: Optional[str]
    ) -> Optional[str]:
        """
        Recursively adds nodes/edges to the Graphviz graph with pruning.

        Args:
            n: Current tree node to render.
            depth: Depth of this node from the root.
            parent_id: Graphviz node id of the parent, or None for root.
            edge_label: Optional edge label (e.g., action or probability).

        Returns:
            Optional[str]: The Graphviz node id created for `n`, or None if pruned.
        """
        nonlocal nodes_added
        if nodes_added >= max_nodes:
            return None

        this_id = f"n{next(nid)}"
        lbl = node_label(n)
        color = node_color(n)
        dot.node(this_id, label=lbl, fillcolor=color)
        nodes_added += 1

        if parent_id is not None:
            if edge_label:
                dot.edge(parent_id, this_id, label=edge_label)
            else:
                dot.edge(parent_id, this_id)

        # Stop conditions
        if depth >= max_depth or n.get("node") == "terminal":
            return this_id

        kids = n.get("children", []) or []

        # Optionally prune to best path at decision nodes
        if best_path_only and n.get("node") == "decision" and kids:
            kids = [best_child(kids)]

        # For chance nodes, optionally collapse tiny-prob branches
        if n.get("node") == "chance" and collapse_chance_below_prob is not None:
            kids = [c for c in kids if (c.get("prob") or 0.0) >= collapse_chance_below_prob or c.get("node") == "decision"]

        # Add children
        for c in kids:
            prob = c.get("prob")
            elabel = None
            if prob is not None:
                elabel = f"p={prob:.4f}"
            # Add short action label for terminals emitted from a decision (we store action in info sometimes)
            if n.get("node") == "decision":
                a = c.get("info", {}).get("action")
                if a:
                    elabel = f"{a}" + (f" • {elabel}" if elabel else "")
            add_subtree(c, depth + 1, this_id, elabel)

        return this_id

    add_subtree(tree, depth=0, parent_id=None, edge_label=None)
    return dot.render(out_path, cleanup=False)


def _parse_hand(s: str) -> List[str]:
    """
    Parses a string like '9H,7D' or '9h 7d' into ['9H','7D'].

    Accepts ranks {2..10, J, Q, K, A} and suits {H, D, C, S}. Also normalizes
    'TH' to '10H'.

    Args:
        s: Input card list string.

    Returns:
        List[str]: Normalized card strings.

    Raises:
        argparse.ArgumentTypeError: If any card has an invalid rank or suit.
    """
    toks = [t.strip().upper() for t in s.replace(" ", ",").split(",") if t.strip()]
    norm = []
    for t in toks:
        if t.startswith("T") and len(t) == 2:
            t = "10" + t[1]  # normalize 'TH' -> '10H'
        rank, suit = t[:-1], t[-1]
        if rank not in {"A","K","Q","J","10","9","8","7","6","5","4","3","2"}:
            raise argparse.ArgumentTypeError(f"Bad rank in card: {t!r}")
        if suit not in {"H","D","C","S"}:
            raise argparse.ArgumentTypeError(f"Bad suit in card: {t!r}")
        norm.append(rank + suit)
    return norm




def main() -> None:
    p = argparse.ArgumentParser(
        description="Render a simple blackjack decision tree with Graphviz."
    )
    p.add_argument("--hand", default="9H,7D", help="Player cards, e.g. '9H,7D' (default).")
    p.add_argument("--dealer", default="6S", help="Dealer upcard, e.g. '6S' (default).")
    p.add_argument("--fmt", default="svg", choices=["svg", "png", "pdf"], help="Graphviz output format (default: svg).")
    p.add_argument("--max-depth", type=int, default=5, help="Maximum decision depth (default: 5).")
    p.add_argument("--max-nodes", type=int, default=1500,  help="Hard cap on nodes (default: 1500).")
    p.add_argument("--collapse-prob", type=float, default=0.002, help="Hide chance branches with probability < threshold (default: 0.002).")
    p.add_argument("--best-path-only", action="store_true", help="Keep only the child with the highest EV at each decision.")
    p.add_argument("--s17", action="store_true", help="Dealer stands on soft 17 (override default H17).")
    p.add_argument("--no-das", action="store_true", help="Disable double after split.")
    args = p.parse_args()

    hand = _parse_hand(args.hand)
    dealer = _parse_hand(args.dealer)
    if len(dealer) != 1:
        p.error("--dealer must be a single card (e.g. '6S').")

    # Default to H17 with DAS unless overridden
    rules = Rules(
        h17=not args.s17,
        das=not args.no_das,
    )

    root = build_tree(hand, dealer[0], rules)

    out_file = visualize_tree(
        root,
        out_path=f"src/game_tree/tree_svgs/blackjack_{'_'.join([h[:-1] for h in hand])}_vs_{args.dealer[:-1]}",
        fmt=args.fmt,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        collapse_chance_below_prob=args.collapse_prob,
        best_path_only=args.best_path_only,
    )
    print("Wrote to:", out_file)


if __name__ == "__main__":
    main()
