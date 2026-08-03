"""
Game outcomes analysis: 5422 games in game_outcomes table.
Analyzes ROI by stadium, month, day/night proxy, max drawdown, and systematic failures.
"""

import os
import sqlite3
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Relativa a la raíz del repo, no a un $HOME concreto: antes era
# "/home/raulio/data/predictions_history.db", que sólo resuelve en la máquina
# del dueño. `FBQ_DB_PATH` permite apuntar a otra copia (una restaurada de
# backup, por ejemplo) sin editar el archivo.
DB = os.environ.get(
    "FBQ_DB_PATH",
    str(Path(__file__).resolve().parent / "data" / "predictions_history.db"),
)

STADIUM_MAP = {
    "Arizona Diamondbacks": "Chase Field",
    "Athletics": "Sutter Health Park",
    "Atlanta Braves": "Truist Park",
    "Baltimore Orioles": "Camden Yards",
    "Boston Red Sox": "Fenway Park",
    "Chicago Cubs": "Wrigley Field",
    "Chicago White Sox": "Guaranteed Rate Field",
    "Cincinnati Reds": "Great American Ball Park",
    "Cleveland Guardians": "Progressive Field",
    "Colorado Rockies": "Coors Field",
    "Detroit Tigers": "Comerica Park",
    "Houston Astros": "Minute Maid Park",
    "Kansas City Royals": "Kauffman Stadium",
    "Los Angeles Angels": "Angel Stadium",
    "Los Angeles Dodgers": "Dodger Stadium",
    "Miami Marlins": "loanDepot park",
    "Milwaukee Brewers": "American Family Field",
    "Minnesota Twins": "Target Field",
    "New York Mets": "Citi Field",
    "New York Yankees": "Yankee Stadium",
    "Oakland Athletics": "Oakland Coliseum",
    "Philadelphia Phillies": "Citizens Bank Park",
    "Pittsburgh Pirates": "PNC Park",
    "San Diego Padres": "Petco Park",
    "San Francisco Giants": "Oracle Park",
    "Seattle Mariners": "T-Mobile Park",
    "St. Louis Cardinals": "Busch Stadium",
    "Tampa Bay Rays": "Tropicana Field",
    "Texas Rangers": "Globe Life Field",
    "Toronto Blue Jays": "Rogers Centre",
    "Washington Nationals": "Nationals Park",
}

MONTH_NAMES = {3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}

EDGE_THRESHOLD = 0.10  # 10% edge for drawdown curve


def decimal_to_implied(decimal_odds):
    """Convert decimal odds to implied probability (no vig removal)."""
    return 1.0 / decimal_odds if decimal_odds and decimal_odds > 1 else None


def compute_pnl(bet_side, home_won, ml_home_pin, ml_away_pin):
    """Return P&L for a 1-unit flat bet. bet_side: 'home' or 'away'."""
    if bet_side == "home":
        return (ml_home_pin - 1) if home_won == 1 else -1.0
    else:
        return (ml_away_pin - 1) if home_won == 0 else -1.0


def load_data():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT game_date, home_team, away_team,
               p_home, p_away, market_prob_home, market_prob_away,
               ml_home_pin, ml_away_pin,
               home_won
        FROM game_outcomes
        WHERE ml_home_pin IS NOT NULL AND ml_away_pin IS NOT NULL
          AND p_home IS NOT NULL AND p_away IS NOT NULL
          AND home_won IS NOT NULL
          AND ml_home_pin > 1 AND ml_away_pin > 1
    """)
    rows = cur.fetchall()
    conn.close()

    games = []
    for row in rows:
        (game_date, home_team, away_team,
         p_home, p_away, market_prob_home, market_prob_away,
         ml_home_pin, ml_away_pin, home_won) = row

        dt = datetime.strptime(game_date, "%Y-%m-%d")
        month = dt.month
        dow = dt.weekday()  # 0=Mon ... 6=Sun
        # Proxy: weekend day games (Sat/Sun) = "Day"; weekdays + Fri nights = "Night"
        day_night = "Day" if dow in (5, 6) else "Night"

        # Edge vs Pinnacle vig-removed market
        # market_prob_home is already vig-removed
        mp_h = market_prob_home if market_prob_home else (1 / ml_home_pin)
        mp_a = market_prob_away if market_prob_away else (1 / ml_away_pin)

        edge_home = p_home - mp_h
        edge_away = p_away - mp_a

        games.append({
            "game_date": game_date,
            "dt": dt,
            "home_team": home_team,
            "away_team": away_team,
            "stadium": STADIUM_MAP.get(home_team, home_team),
            "month": month,
            "day_night": day_night,
            "p_home": p_home,
            "p_away": p_away,
            "mp_h": mp_h,
            "mp_a": mp_a,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "ml_home_pin": ml_home_pin,
            "ml_away_pin": ml_away_pin,
            "home_won": home_won,
        })

    games.sort(key=lambda x: x["dt"])
    return games


def roi_summary(bets_list):
    """bets_list: list of (pnl,). Returns (n, profit, roi_pct)."""
    if not bets_list:
        return 0, 0.0, 0.0
    n = len(bets_list)
    profit = sum(bets_list)
    roi = 100.0 * profit / n
    return n, profit, roi


def analyze_roi_by_group(games, key_fn):
    """
    For each group (key), bet home when edge_home >= 0 OR edge_away >= 0 (best side).
    Return dict: key -> (n_bets, profit, roi_pct).
    """
    groups = defaultdict(list)
    for g in games:
        # Bet the side with positive edge only
        if g["edge_home"] >= 0:
            pnl = compute_pnl("home", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            groups[key_fn(g)].append(pnl)
        if g["edge_away"] >= 0:
            pnl = compute_pnl("away", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            groups[key_fn(g)].append(pnl)
    return {k: roi_summary(v) for k, v in groups.items()}


def max_drawdown_flat(games, edge_threshold):
    """
    Flat 1-unit betting on best edge side when edge >= threshold.
    Returns (bets, equity_curve, max_dd, max_dd_start, max_dd_end).
    """
    equity = [0.0]
    dates = ["start"]
    peak = 0.0
    max_dd = 0.0
    peak_date = "start"
    dd_start = "start"
    dd_end = "start"

    for g in games:
        best_edge = None
        best_side = None
        if g["edge_home"] >= edge_threshold and (best_edge is None or g["edge_home"] > best_edge):
            best_edge = g["edge_home"]
            best_side = "home"
        if g["edge_away"] >= edge_threshold and (best_edge is None or g["edge_away"] > best_edge):
            best_edge = g["edge_away"]
            best_side = "away"

        if best_side is None:
            continue

        pnl = compute_pnl(best_side, g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
        new_equity = equity[-1] + pnl
        equity.append(new_equity)
        dates.append(g["game_date"])

        if new_equity > peak:
            peak = new_equity
            peak_date = g["game_date"]

        dd = peak - new_equity
        if dd > max_dd:
            max_dd = dd
            dd_start = peak_date
            dd_end = g["game_date"]

    return equity, dates, max_dd, dd_start, dd_end


def analyze_team_matchups(games, edge_threshold=0.0, top_n=20):
    """
    Find matchups where the model bets most confidently but loses most.
    We define 'bet side' as the side with positive edge.
    Failure = model predicted a side but was wrong.
    """
    matchup_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0.0, "total_edge": 0.0})

    for g in games:
        home, away = g["home_team"], g["away_team"]

        if g["edge_home"] >= edge_threshold:
            key = f"{away} @ {home}"  # "away @ home" = this game context
            matchup_stats[key]["bets"] += 1
            matchup_stats[key]["total_edge"] += g["edge_home"]
            pnl = compute_pnl("home", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            matchup_stats[key]["profit"] += pnl
            if g["home_won"] == 1:
                matchup_stats[key]["wins"] += 1

        if g["edge_away"] >= edge_threshold:
            key = f"{home} vs {away}"  # "home vs away" = betting away here
            matchup_stats[key]["bets"] += 1
            matchup_stats[key]["total_edge"] += g["edge_away"]
            pnl = compute_pnl("away", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            matchup_stats[key]["profit"] += pnl
            if g["home_won"] == 0:
                matchup_stats[key]["wins"] += 1

    # Only matchups with >= 3 bets, sorted by worst ROI
    result = []
    for key, s in matchup_stats.items():
        if s["bets"] >= 3:
            roi = 100.0 * s["profit"] / s["bets"]
            win_rate = 100.0 * s["wins"] / s["bets"]
            avg_edge = 100.0 * s["total_edge"] / s["bets"]
            result.append({
                "matchup": key,
                "bets": s["bets"],
                "wins": s["wins"],
                "win_rate": win_rate,
                "profit": s["profit"],
                "roi": roi,
                "avg_edge": avg_edge,
            })

    result.sort(key=lambda x: x["roi"])
    return result[:top_n]


def analyze_team_level_failures(games):
    """Per-team: when model bets FOR that team (as home or away), ROI."""
    team_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "profit": 0.0})

    for g in games:
        if g["edge_home"] >= 0:
            team = g["home_team"]
            team_stats[team]["bets"] += 1
            pnl = compute_pnl("home", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            team_stats[team]["profit"] += pnl
            if g["home_won"] == 1:
                team_stats[team]["wins"] += 1

        if g["edge_away"] >= 0:
            team = g["away_team"]
            team_stats[team]["bets"] += 1
            pnl = compute_pnl("away", g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            team_stats[team]["profit"] += pnl
            if g["home_won"] == 0:
                team_stats[team]["wins"] += 1

    result = []
    for team, s in team_stats.items():
        if s["bets"] >= 10:
            roi = 100.0 * s["profit"] / s["bets"]
            win_rate = 100.0 * s["wins"] / s["bets"]
            result.append({
                "team": team,
                "bets": s["bets"],
                "win_rate": win_rate,
                "profit": s["profit"],
                "roi": roi,
            })
    result.sort(key=lambda x: x["roi"])
    return result


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    games = load_data()
    print(f"\nLoaded {len(games)} games with complete betting data")

    # ─── 1. ROI by Stadium ─────────────────────────────────────────────────
    print_separator("1. ROI BY STADIUM (home team = stadium)")
    stadium_roi = analyze_roi_by_group(games, lambda g: g["stadium"])
    rows = [(k, *v) for k, v in stadium_roi.items()]
    rows.sort(key=lambda x: x[3])  # sort by ROI

    print(f"{'Stadium':<30} {'Bets':>6} {'Profit':>8} {'ROI%':>8}")
    print("-" * 56)
    for stadium, n, profit, roi in rows:
        bar = "▲" if roi > 0 else "▼"
        print(f"{stadium:<30} {n:>6} {profit:>+8.2f} {roi:>+7.2f}% {bar}")

    positive = [(r) for r in rows if r[3] > 0]
    negative = [(r) for r in rows if r[3] <= 0]
    print(f"\n  Profitable venues: {len(positive)}/{len(rows)}")
    if rows:
        best = max(rows, key=lambda x: x[3])
        worst = min(rows, key=lambda x: x[3])
        print(f"  Best:  {best[0]} ROI={best[3]:+.2f}% ({best[1]} bets)")
        print(f"  Worst: {worst[0]} ROI={worst[3]:+.2f}% ({worst[1]} bets)")

    # ─── 2. ROI by Month ───────────────────────────────────────────────────
    print_separator("2. ROI BY MONTH")
    month_roi = analyze_roi_by_group(games, lambda g: g["month"])
    rows_m = [(MONTH_NAMES.get(k, str(k)), *v) for k, v in month_roi.items()]
    rows_m.sort(key=lambda x: list(MONTH_NAMES.values()).index(x[0]) if x[0] in MONTH_NAMES.values() else 99)

    print(f"{'Month':<8} {'Bets':>6} {'Profit':>8} {'ROI%':>8}  {'Bar':}")
    print("-" * 50)
    for month_name, n, profit, roi in rows_m:
        bar_len = int(abs(roi) / 2)
        bar = ("+" if roi > 0 else "-") * min(bar_len, 20)
        print(f"{month_name:<8} {n:>6} {profit:>+8.2f} {roi:>+7.2f}%  {bar}")

    # ─── 3. ROI by Day vs Night (Weekend proxy) ───────────────────────────
    print_separator("3. ROI BY DAY vs NIGHT  (proxy: Sat+Sun = Day, weekdays = Night)")
    print("  Note: DB has no game start times. Sat/Sun coded as Day, Mon-Fri as Night.")
    dn_roi = analyze_roi_by_group(games, lambda g: g["day_night"])
    rows_dn = [(k, *v) for k, v in dn_roi.items()]
    rows_dn.sort(key=lambda x: x[0])

    print(f"\n{'Type':<10} {'Bets':>6} {'Profit':>8} {'ROI%':>8}")
    print("-" * 38)
    for label, n, profit, roi in rows_dn:
        print(f"{label:<10} {n:>6} {profit:>+8.2f} {roi:>+7.2f}%")

    # Day-of-week breakdown for detail
    print(f"\n  Day-of-week detail:")
    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_roi = analyze_roi_by_group(games, lambda g: g["dt"].weekday())
    rows_dow = [(DOW[k], *v) for k, v in dow_roi.items()]
    rows_dow.sort(key=lambda x: DOW.index(x[0]))
    print(f"  {'Day':<6} {'Bets':>6} {'Profit':>8} {'ROI%':>8}")
    print("  " + "-" * 34)
    for day, n, profit, roi in rows_dow:
        bar = "▲" if roi > 0 else "▼"
        print(f"  {day:<6} {n:>6} {profit:>+8.2f} {roi:>+7.2f}% {bar}")

    # ─── 4. Max Drawdown Curve (edge >= 10%) ──────────────────────────────
    print_separator(f"4. MAX DRAWDOWN CURVE  (flat 1-unit, edge >= {EDGE_THRESHOLD*100:.0f}%)")
    equity, dates, max_dd, dd_start, dd_end = max_drawdown_flat(games, EDGE_THRESHOLD)

    n_bets = len(equity) - 1
    final_profit = equity[-1]
    roi_pct = 100.0 * final_profit / n_bets if n_bets > 0 else 0

    peak_equity = max(equity)
    trough_after_peak = min(equity[equity.index(peak_equity):])

    print(f"\n  Bets placed (edge >= {EDGE_THRESHOLD*100:.0f}%): {n_bets}")
    print(f"  Final profit:   {final_profit:+.2f} units")
    print(f"  ROI:            {roi_pct:+.2f}%")
    print(f"  Peak equity:    {peak_equity:+.2f} units")
    print(f"  Max drawdown:   -{max_dd:.2f} units  ({dd_start} → {dd_end})")

    # Drawdown as % of peak
    if peak_equity > 0:
        dd_pct = 100.0 * max_dd / peak_equity
        print(f"  Max DD (% peak): {dd_pct:.1f}%")

    # ASCII equity curve (last 80 points for readability)
    sample_eq = equity
    if len(sample_eq) > 80:
        step = len(sample_eq) // 80
        sample_eq = sample_eq[::step]

    min_eq = min(sample_eq)
    max_eq = max(sample_eq)
    span = max_eq - min_eq if max_eq != min_eq else 1
    height = 12

    print(f"\n  Equity curve (each col ≈ {len(equity)//len(sample_eq)} bets):")
    print(f"  Peak={peak_equity:+.1f}u  Trough={min(equity):+.1f}u  Final={equity[-1]:+.1f}u\n")

    for row in range(height, -1, -1):
        threshold_val = min_eq + (row / height) * span
        label_val = min_eq + (row / height) * span
        line = f"  {label_val:+6.1f} |"
        for eq_val in sample_eq[1:]:
            line += "█" if eq_val >= threshold_val else " "
        print(line)
    print("         " + "─" * len(sample_eq))

    # Drawdown curve
    print(f"\n  Drawdown from peak (rolling):")
    dd_curve = []
    running_peak = 0.0
    for e in equity:
        if e > running_peak:
            running_peak = e
        dd_curve.append(running_peak - e)

    sample_dd = dd_curve
    if len(sample_dd) > 80:
        step = len(sample_dd) // 80
        sample_dd = sample_dd[::step]

    max_dd_val = max(dd_curve) if dd_curve else 0
    for row in range(height, -1, -1):
        threshold_val = (row / height) * max_dd_val
        label_val = (row / height) * max_dd_val
        line = f"  {label_val:+6.1f} |"
        for dd_val in sample_dd[1:]:
            line += "█" if dd_val >= threshold_val else " "
        print(line)
    print("         " + "─" * len(sample_dd))

    # Monthly breakdown of edge>=10% bets
    print(f"\n  Monthly P&L for edge >= {EDGE_THRESHOLD*100:.0f}% bets:")
    monthly_edge = defaultdict(list)
    for g in games:
        best_edge, best_side = None, None
        if g["edge_home"] >= EDGE_THRESHOLD:
            if best_edge is None or g["edge_home"] > best_edge:
                best_edge, best_side = g["edge_home"], "home"
        if g["edge_away"] >= EDGE_THRESHOLD:
            if best_edge is None or g["edge_away"] > best_edge:
                best_edge, best_side = g["edge_away"], "away"
        if best_side:
            pnl = compute_pnl(best_side, g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            key = f"{g['dt'].year}-{MONTH_NAMES.get(g['month'], g['month'])}"
            monthly_edge[key].append(pnl)

    print(f"  {'Period':<12} {'Bets':>5} {'Profit':>8} {'ROI%':>8}")
    print("  " + "-" * 38)
    for period in sorted(monthly_edge.keys()):
        bets = monthly_edge[period]
        n = len(bets)
        p = sum(bets)
        r = 100 * p / n
        bar = "▲" if r > 0 else "▼"
        print(f"  {period:<12} {n:>5} {p:>+8.2f} {r:>+7.2f}% {bar}")

    # ─── 5. Systematic Failures by Team Matchup ──────────────────────────
    print_separator("5. SYSTEMATIC MODEL FAILURES BY TEAM MATCHUP")

    # A. Worst specific matchups (away_team @ home_team where model consistently wrong)
    print("\n  A. Worst specific matchups (min 3 bets on that matchup):")
    bad_matchups = analyze_team_matchups(games, edge_threshold=0.0, top_n=15)
    print(f"  {'Matchup':<45} {'Bets':>5} {'Win%':>6} {'Profit':>8} {'ROI%':>8} {'AvgEdge%':>9}")
    print("  " + "-" * 82)
    for m in bad_matchups:
        print(f"  {m['matchup']:<45} {m['bets']:>5} {m['win_rate']:>5.1f}% "
              f"{m['profit']:>+8.2f} {m['roi']:>+7.2f}% {m['avg_edge']:>+8.2f}%")

    # B. Per-team: worst ROI when model backs that team
    print("\n  B. Teams where model performs worst when backing them (min 10 bets):")
    team_failures = analyze_team_level_failures(games)
    print(f"  {'Team':<30} {'Bets':>5} {'Win%':>6} {'Profit':>8} {'ROI%':>8}")
    print("  " + "-" * 62)
    for t in team_failures[:15]:
        print(f"  {t['team']:<30} {t['bets']:>5} {t['win_rate']:>5.1f}% "
              f"{t['profit']:>+8.2f} {t['roi']:>+7.2f}%")

    # C. Overconfidence analysis: model edge >= 15% but actual win rate
    print("\n  C. Model overconfidence bands (home+away bets combined):")
    bands = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    print(f"  {'Edge Band':<18} {'Bets':>6} {'Win%':>7} {'Exp Win%':>9} {'Profit':>8} {'ROI%':>8}")
    print("  " + "-" * 62)
    for lo, hi in bands:
        band_bets = []
        wins = 0
        exp_wins = 0.0
        for g in games:
            for side in ("home", "away"):
                edge = g["edge_home"] if side == "home" else g["edge_away"]
                mp = g["mp_h"] if side == "home" else g["mp_a"]
                model_p = g["p_home"] if side == "home" else g["p_away"]
                if lo <= edge < hi:
                    pnl = compute_pnl(side, g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
                    band_bets.append(pnl)
                    actual_win = (g["home_won"] == 1) if side == "home" else (g["home_won"] == 0)
                    if actual_win:
                        wins += 1
                    exp_wins += model_p
        if band_bets:
            n = len(band_bets)
            profit = sum(band_bets)
            roi = 100 * profit / n
            win_rate = 100 * wins / n
            exp_win_rate = 100 * exp_wins / n
            print(f"  [{lo*100:.0f}%–{hi*100:.0f}%)    {n:>10} {win_rate:>6.1f}% {exp_win_rate:>8.1f}% "
                  f"{profit:>+8.2f} {roi:>+7.2f}%")

    # D. Calibration: p_home vs actual win rate in deciles
    print("\n  D. Model calibration — p_home deciles vs actual home win rate:")
    import math
    probs = [(g["p_home"], g["home_won"]) for g in games]
    probs.sort(key=lambda x: x[0])
    n = len(probs)
    decile_size = n // 10
    print(f"  {'p_home range':<20} {'N':>5} {'Pred%':>7} {'Actual%':>8} {'Diff':>6}")
    print("  " + "-" * 50)
    for i in range(10):
        bucket = probs[i * decile_size: (i + 1) * decile_size]
        pred_avg = sum(p for p, _ in bucket) / len(bucket)
        actual_avg = sum(w for _, w in bucket) / len(bucket)
        diff = actual_avg - pred_avg
        lo_p = bucket[0][0]
        hi_p = bucket[-1][0]
        bar = "▲" if diff > 0.02 else ("▼" if diff < -0.02 else "≈")
        print(f"  [{lo_p:.3f}–{hi_p:.3f}]     {len(bucket):>5} {pred_avg*100:>6.1f}% {actual_avg*100:>7.1f}% "
              f"{diff*100:>+5.1f}% {bar}")

    # ─── Summary ──────────────────────────────────────────────────────────
    print_separator("SUMMARY")
    # Overall betting stats (any positive edge)
    total_bets, total_profit = 0, 0.0
    wins_count = 0
    for g in games:
        for side in ("home", "away"):
            edge = g["edge_home"] if side == "home" else g["edge_away"]
            if edge >= 0:
                pnl = compute_pnl(side, g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
                total_bets += 1
                total_profit += pnl
                if pnl > 0:
                    wins_count += 1

    print(f"\n  Universe:       {len(games)} games (2024-03-20 → 2026-05-07)")
    print(f"  Bets (edge≥0):  {total_bets} bets placed")
    print(f"  Win rate:       {100*wins_count/total_bets:.1f}%")
    print(f"  Total profit:   {total_profit:+.2f} units")
    print(f"  Overall ROI:    {100*total_profit/total_bets:+.2f}%")

    bets_10, profit_10 = 0, 0.0
    for g in games:
        best_edge, best_side = None, None
        for side in ("home", "away"):
            e = g["edge_home"] if side == "home" else g["edge_away"]
            if e >= EDGE_THRESHOLD and (best_edge is None or e > best_edge):
                best_edge, best_side = e, side
        if best_side:
            pnl = compute_pnl(best_side, g["home_won"], g["ml_home_pin"], g["ml_away_pin"])
            bets_10 += 1
            profit_10 += pnl

    print(f"\n  Edge ≥ 10%:     {bets_10} bets")
    print(f"  Profit:         {profit_10:+.2f} units")
    print(f"  ROI:            {100*profit_10/bets_10:+.2f}%" if bets_10 else "  No bets")
    print(f"  Max drawdown:   -{max_dd:.2f} units over {n_bets} bets")


if __name__ == "__main__":
    main()
