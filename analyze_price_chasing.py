#!/usr/bin/env python3
"""
FPL Price Chasing Analysis

Analyzes whether chasing price rises in Fantasy Premier League is a good strategy.
Key question: Do players who rise in price actually score more points the following week?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Seasons available in vaastav's repo
SEASONS = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def fetch_season_data(season: str) -> pd.DataFrame:
    """Fetch merged gameweek data for a season from vaastav's repo."""
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df['season'] = season
        print(f"  ✓ {season}: {len(df):,} rows, {df['name'].nunique()} players")
        return df
    except Exception as e:
        print(f"  ✗ {season}: {e}")
        return pd.DataFrame()


def load_all_data() -> pd.DataFrame:
    """Load data from all available seasons."""
    print("Fetching FPL data from vaastav's repo...")
    dfs = []
    for season in SEASONS:
        df = fetch_season_data(season)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("Could not fetch any season data!")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(combined):,} gameweek observations across {len(dfs)} seasons")
    return combined


def calculate_price_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price changes and next-week points for each player-gameweek.

    Price in FPL is stored as value/10 (e.g., 100 = £10.0m)
    """
    # Ensure we have the required columns
    required_cols = ['name', 'value', 'total_points', 'season']
    gw_col = 'GW' if 'GW' in df.columns else 'round'

    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        raise ValueError(f"Missing columns: {missing}")

    # Create unique player identifier per season
    df = df.copy()
    df['player_season'] = df['name'] + '_' + df['season']
    df['gw'] = df[gw_col]

    # Sort by player and gameweek
    df = df.sort_values(['player_season', 'gw'])

    # Calculate price change from previous GW
    df['prev_value'] = df.groupby('player_season')['value'].shift(1)
    df['price_change'] = df['value'] - df['prev_value']

    # Get next week's points (what we're trying to predict)
    df['next_gw_points'] = df.groupby('player_season')['total_points'].shift(-1)

    # Get previous week's points (form indicator)
    df['prev_gw_points'] = df.groupby('player_season')['total_points'].shift(1)

    # Remove first and last GW for each player (no prev/next data)
    df = df.dropna(subset=['price_change', 'next_gw_points'])

    # Categorize price changes
    df['price_category'] = pd.cut(
        df['price_change'],
        bins=[-np.inf, -0.5, -0.01, 0.01, 0.5, np.inf],
        labels=['Big Fall (>£0.05m)', 'Small Fall', 'Stable', 'Small Rise', 'Big Rise (>£0.05m)']
    )

    return df


def analyze_price_vs_points(df: pd.DataFrame) -> dict:
    """Analyze the relationship between price changes and next-week points."""
    results = {}

    # 1. Overall correlation
    corr, p_value = stats.pearsonr(df['price_change'], df['next_gw_points'])
    results['correlation'] = {'r': corr, 'p_value': p_value}

    # 2. Points by price category
    category_stats = df.groupby('price_category', observed=True)['next_gw_points'].agg([
        'mean', 'median', 'std', 'count'
    ]).round(2)
    results['by_category'] = category_stats

    # 3. Compare risers vs fallers
    risers = df[df['price_change'] > 0]['next_gw_points']
    fallers = df[df['price_change'] < 0]['next_gw_points']
    stable = df[df['price_change'] == 0]['next_gw_points']

    # T-test: risers vs stable
    if len(risers) > 0 and len(stable) > 0:
        t_stat, t_pval = stats.ttest_ind(risers, stable)
        results['risers_vs_stable'] = {
            'risers_mean': risers.mean(),
            'stable_mean': stable.mean(),
            'difference': risers.mean() - stable.mean(),
            't_statistic': t_stat,
            'p_value': t_pval
        }

    # 4. Form persistence (does this week's points predict next week?)
    form_corr, form_p = stats.pearsonr(
        df['total_points'].dropna(),
        df['next_gw_points'].loc[df['total_points'].notna()]
    )
    results['form_persistence'] = {'r': form_corr, 'p_value': form_p}

    # 5. Previous week's points vs next week's (regression to mean)
    valid_prev = df.dropna(subset=['prev_gw_points'])
    if len(valid_prev) > 0:
        reg_corr, reg_p = stats.pearsonr(valid_prev['prev_gw_points'], valid_prev['next_gw_points'])
        results['regression_to_mean'] = {'r': reg_corr, 'p_value': reg_p}

    return results


def analyze_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze price chasing effectiveness by position."""
    if 'position' not in df.columns:
        return pd.DataFrame()

    position_analysis = []
    for pos in df['position'].unique():
        pos_df = df[df['position'] == pos]
        risers = pos_df[pos_df['price_change'] > 0]['next_gw_points']
        stable = pos_df[pos_df['price_change'] == 0]['next_gw_points']

        if len(risers) > 30 and len(stable) > 30:
            position_analysis.append({
                'position': pos,
                'risers_mean': risers.mean(),
                'stable_mean': stable.mean(),
                'advantage': risers.mean() - stable.mean(),
                'n_risers': len(risers),
                'n_stable': len(stable)
            })

    return pd.DataFrame(position_analysis)


def analyze_playing_players_only(df: pd.DataFrame) -> dict:
    """
    CRITICAL: Compare only players who actually played (minutes > 0).

    This removes the confound that stable-price players include many
    bench players who score 0 points, making risers look artificially better.
    """
    if 'minutes' not in df.columns:
        return {}

    # Filter to only players who played in the CURRENT gameweek
    # (i.e., they were playing when the price change happened)
    playing = df[df['minutes'] > 0].copy()

    results = {}

    # Correlation among playing players only
    corr, p_val = stats.pearsonr(playing['price_change'], playing['next_gw_points'])
    results['correlation_playing_only'] = {'r': corr, 'p_value': p_val}

    # Compare risers vs stable among players who played
    risers = playing[playing['price_change'] > 0]['next_gw_points']
    stable = playing[playing['price_change'] == 0]['next_gw_points']
    fallers = playing[playing['price_change'] < 0]['next_gw_points']

    results['playing_only_comparison'] = {
        'risers_mean': risers.mean() if len(risers) > 0 else 0,
        'stable_mean': stable.mean() if len(stable) > 0 else 0,
        'fallers_mean': fallers.mean() if len(fallers) > 0 else 0,
        'risers_n': len(risers),
        'stable_n': len(stable),
        'fallers_n': len(fallers),
        'difference': (risers.mean() - stable.mean()) if len(risers) > 0 and len(stable) > 0 else 0
    }

    # T-test for playing players
    if len(risers) > 30 and len(stable) > 30:
        t_stat, t_pval = stats.ttest_ind(risers, stable)
        results['playing_only_comparison']['t_pvalue'] = t_pval

    return results


def create_visualizations(df: pd.DataFrame, results: dict):
    """Create visualizations for the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FPL Price Chasing Analysis: Is Chasing Price Rises Worth It?', fontsize=14, fontweight='bold')

    # 1. Box plot: Points by price category
    ax1 = axes[0, 0]
    order = ['Big Fall (>£0.05m)', 'Small Fall', 'Stable', 'Small Rise', 'Big Rise (>£0.05m)']
    existing_cats = [c for c in order if c in df['price_category'].unique()]
    sns.boxplot(data=df, x='price_category', y='next_gw_points', ax=ax1, order=existing_cats)
    ax1.set_xlabel('Price Change Category')
    ax1.set_ylabel('Next Gameweek Points')
    ax1.set_title('Next GW Points by Price Change Category')
    ax1.tick_params(axis='x', rotation=30)

    # Add means as text
    for i, cat in enumerate(existing_cats):
        mean_val = df[df['price_category'] == cat]['next_gw_points'].mean()
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f'μ={mean_val:.2f}', ha='center', fontsize=9)

    # 2. Scatter plot with regression line
    ax2 = axes[0, 1]
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax2.scatter(sample['price_change'], sample['next_gw_points'], alpha=0.1, s=10)

    # Add regression line
    z = np.polyfit(df['price_change'], df['next_gw_points'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['price_change'].min(), df['price_change'].max(), 100)
    ax2.plot(x_line, p(x_line), "r-", linewidth=2, label=f"r = {results['correlation']['r']:.4f}")
    ax2.set_xlabel('Price Change (£0.1m units)')
    ax2.set_ylabel('Next Gameweek Points')
    ax2.set_title('Price Change vs Next GW Points')
    ax2.legend()
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 3. Mean points comparison bar chart
    ax3 = axes[1, 0]
    cat_means = df.groupby('price_category', observed=True)['next_gw_points'].mean()
    cat_means = cat_means.reindex(existing_cats)
    colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1f77b4'][:len(existing_cats)]
    bars = ax3.bar(range(len(cat_means)), cat_means.values, color=colors)
    ax3.set_xticks(range(len(cat_means)))
    ax3.set_xticklabels([c.replace(' (>£0.05m)', '\n(>£0.05m)') for c in existing_cats], rotation=0, fontsize=9)
    ax3.set_ylabel('Mean Next GW Points')
    ax3.set_title('Average Next-Week Points by Price Movement')
    ax3.axhline(y=df['next_gw_points'].mean(), color='black', linestyle='--', label='Overall Mean')
    ax3.legend()

    # Add value labels on bars
    for bar, val in zip(bars, cat_means.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 4. Distribution comparison
    ax4 = axes[1, 1]
    risers = df[df['price_change'] > 0]['next_gw_points']
    stable = df[df['price_change'] == 0]['next_gw_points']
    fallers = df[df['price_change'] < 0]['next_gw_points']

    ax4.hist(stable, bins=30, alpha=0.5, label=f'Stable (n={len(stable):,})', density=True, color='gray')
    ax4.hist(risers, bins=30, alpha=0.5, label=f'Risers (n={len(risers):,})', density=True, color='green')
    ax4.hist(fallers, bins=30, alpha=0.5, label=f'Fallers (n={len(fallers):,})', density=True, color='red')
    ax4.set_xlabel('Next Gameweek Points')
    ax4.set_ylabel('Density')
    ax4.set_title('Distribution of Next-Week Points')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('price_chasing_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved visualization to: price_chasing_analysis.png")


def print_results(results: dict, df: pd.DataFrame):
    """Print analysis results in a readable format."""
    print("\n" + "="*70)
    print("FPL PRICE CHASING ANALYSIS RESULTS")
    print("="*70)

    # Overall correlation
    print("\n📊 CORRELATION: Price Change → Next Week Points")
    print("-"*50)
    r = results['correlation']['r']
    p = results['correlation']['p_value']
    print(f"   Pearson correlation: r = {r:.4f}")
    print(f"   P-value: {p:.2e}")
    if abs(r) < 0.1:
        print("   → Very weak/negligible correlation")
    elif abs(r) < 0.3:
        print("   → Weak correlation")
    else:
        print("   → Moderate correlation")

    # Points by category
    print("\n📈 NEXT-WEEK POINTS BY PRICE CATEGORY")
    print("-"*50)
    print(results['by_category'].to_string())

    # Risers vs Stable comparison
    if 'risers_vs_stable' in results:
        print("\n⚖️  RISERS vs STABLE PLAYERS")
        print("-"*50)
        rvs = results['risers_vs_stable']
        print(f"   Price risers avg next-week points:  {rvs['risers_mean']:.2f}")
        print(f"   Stable price avg next-week points:  {rvs['stable_mean']:.2f}")
        print(f"   Difference: {rvs['difference']:+.2f} points")
        print(f"   T-test p-value: {rvs['p_value']:.2e}")
        if rvs['p_value'] < 0.05:
            print(f"   → Statistically significant difference!")
        else:
            print(f"   → NOT statistically significant")

    # Form persistence
    print("\n🔄 FORM PERSISTENCE (This Week → Next Week)")
    print("-"*50)
    fp = results['form_persistence']
    print(f"   Correlation: r = {fp['r']:.4f}")
    print(f"   → This week's points {'weakly' if abs(fp['r']) < 0.2 else 'moderately'} predict next week")

    # CRITICAL: Playing players only analysis
    if 'playing_only_comparison' in results:
        print("\n⚽ CRITICAL: PLAYING PLAYERS ONLY (minutes > 0)")
        print("-"*50)
        print("   (This removes bench fodder who inflate stable-price averages)")
        poc = results['playing_only_comparison']
        print(f"\n   Among players who actually PLAYED:")
        print(f"   • Risers avg next-week pts:    {poc['risers_mean']:.2f} (n={poc['risers_n']:,})")
        print(f"   • Stable avg next-week pts:    {poc['stable_mean']:.2f} (n={poc['stable_n']:,})")
        print(f"   • Fallers avg next-week pts:   {poc['fallers_mean']:.2f} (n={poc['fallers_n']:,})")
        print(f"\n   Difference (risers - stable): {poc['difference']:+.2f} points")

        if 'correlation_playing_only' in results:
            cpo = results['correlation_playing_only']
            print(f"   Correlation (playing only):   r = {cpo['r']:.4f}")

        if 't_pvalue' in poc:
            sig = "YES" if poc['t_pvalue'] < 0.05 else "NO"
            print(f"   Statistically significant:    {sig} (p={poc['t_pvalue']:.2e})")

    # Key insight
    print("\n" + "="*70)
    print("🎯 KEY INSIGHT")
    print("="*70)

    # Use playing-only data for fairer comparison
    if 'playing_only_comparison' in results:
        poc = results['playing_only_comparison']
        risers_mean = poc['risers_mean']
        stable_mean = poc['stable_mean']
        diff = poc['difference']
        corr = results.get('correlation_playing_only', {}).get('r', results['correlation']['r'])
    else:
        risers_mean = df[df['price_change'] > 0]['next_gw_points'].mean()
        stable_mean = df[df['price_change'] == 0]['next_gw_points'].mean()
        diff = risers_mean - stable_mean
        corr = results['correlation']['r']

    if abs(diff) < 0.5:
        print(f"""
   🚨 PRICE CHASING IS MOSTLY A TRAP!

   When comparing ONLY players who actually play (removing bench fodder):
   • Risers average:  {risers_mean:.2f} pts next week
   • Stable average:  {stable_mean:.2f} pts next week
   • Difference:      {diff:+.2f} pts (basically negligible)

   The correlation (r={corr:.3f}) shows price changes explain almost
   NOTHING about next-week performance.

   WHY THIS HAPPENS:
   1. Price rises reflect PAST performance, not future
   2. FPL is highly random week-to-week (luck dominates)
   3. You're buying at the TOP of form (regression to mean)

   💡 BETTER STRATEGIES:
   • Focus on fixtures (home vs away, opponent strength)
   • Use expected stats (xG, xA) over actual returns
   • Consider underlying performance, not price movements
   • Buy BEFORE price rises (when you spot form early)
""")
    else:
        direction = "more" if diff > 0 else "fewer"
        print(f"""
   Price risers score {direction} points on average ({risers_mean:.2f} vs {stable_mean:.2f}).

   BUT the correlation is only r={corr:.3f} - price changes explain very
   little of the variance in next-week points.

   The {abs(diff):.1f} point difference sounds significant but:
   • High variance means this gap is often noise
   • You're paying extra price for uncertain returns
   • Regression to mean will catch up eventually

   💡 RECOMMENDATION: Don't chase price rises as your primary strategy.
      Focus on fixtures, expected stats (xG, xA), and playing time.
""")


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("FPL PRICE CHASING ANALYSIS")
    print("Is buying players after price rises a good strategy?")
    print("="*70 + "\n")

    # Load data
    df = load_all_data()

    # Calculate price changes
    print("\nCalculating price changes and next-week points...")
    df = calculate_price_changes(df)
    print(f"Analyzing {len(df):,} player-gameweek observations with valid price change data")

    # Run analysis
    print("\nRunning statistical analysis...")
    results = analyze_price_vs_points(df)

    # CRITICAL: Analyze only playing players (removes bench fodder confound)
    playing_results = analyze_playing_players_only(df)
    results.update(playing_results)

    # Position analysis
    pos_analysis = analyze_by_position(df)
    if not pos_analysis.empty:
        print("\n📋 ANALYSIS BY POSITION")
        print("-"*50)
        print(pos_analysis.to_string(index=False))

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, results)

    # Print results
    print_results(results, df)

    # Save detailed results
    summary_df = df.groupby('price_category', observed=True).agg({
        'next_gw_points': ['mean', 'median', 'std', 'count'],
        'total_points': 'mean',
        'price_change': 'mean'
    }).round(3)
    summary_df.to_csv('price_chasing_summary.csv')
    print("\n✓ Saved detailed summary to: price_chasing_summary.csv")

    return df, results


if __name__ == "__main__":
    df, results = main()
