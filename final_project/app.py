"""
GUI for Job Referral Network Inequality Model
Sidebar sliders control all model parameters.
Main panel shows the live network (node color = group x employment status,
edge width = tie strength) alongside three time-series panels.
"""

import solara
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from model import JobReferralModel

# Reactive state
model_state = solara.reactive(None)
step_count  = solara.reactive(0)

# Layout cache: recomputed only when node set changes (prevents jumping nodes)
_cached_pos = {}

# Parameter reactive variables
n_agents                 = solara.reactive(80)
homophily                = solara.reactive(0.8)
pct_high_status          = solara.reactive(0.5)
weak_tie_prob            = solara.reactive(0.05)
initial_employed         = solara.reactive(0.3)
initial_employed_low     = solara.reactive(0.0)
referral_willingness     = solara.reactive(0.6)
ingroup_bias             = solara.reactive(1.2)
acceptance_prob          = solara.reactive(0.5)
job_loss_prob            = solara.reactive(0.01)
new_tie_prob             = solara.reactive(0.05)
tie_growth_rate          = solara.reactive(0.02)
tie_decay_rate           = solara.reactive(0.03)
isolation_threshold      = solara.reactive(5)
max_degree               = solara.reactive(15)


def make_model():
    return JobReferralModel(
        n_agents                 = n_agents.value,
        homophily                = homophily.value,
        pct_high_status          = pct_high_status.value,
        weak_tie_prob            = weak_tie_prob.value,
        initial_employed_pct     = initial_employed.value,
        initial_employed_pct_low = initial_employed_low.value,
        referral_willingness     = referral_willingness.value,
        ingroup_bias             = ingroup_bias.value,
        acceptance_prob          = acceptance_prob.value,
        job_loss_prob            = job_loss_prob.value,
        new_tie_prob             = new_tie_prob.value,
        tie_growth_rate          = tie_growth_rate.value,
        tie_decay_rate           = tie_decay_rate.value,
        isolation_threshold      = isolation_threshold.value,
        max_degree               = max_degree.value,
        seed                     = 42,
    )


# Drawing helpers 

def draw_network(model):
    """
    Draw the co-evolving network.
    Node color encodes group + employment status.
    Edge thickness encodes tie strength (thicker = stronger trust).
    Layout is cached so nodes do not jump as edges are added or removed.
    """
    global _cached_pos
    G = model.G

    # Recompute layout only when node set changes (e.g. after Reset)
    if not _cached_pos or set(_cached_pos.keys()) != set(G.nodes()):
        _cached_pos = nx.spring_layout(G, seed=42, k=0.5, weight="weight")
    pos = _cached_pos

    fig, ax = plt.subplots(figsize=(6, 5))

    color_map = []
    for node in G.nodes():
        agents = model.grid.get_cell_list_contents([node])
        if agents:
            a = agents[0]
            if a.group == "high_status":
                color_map.append("steelblue" if a.employed else "lightblue")
            else:
                color_map.append("tomato" if a.employed else "lightsalmon")
        else:
            color_map.append("grey")

    weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    widths    = [w * 2.5 for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=80, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="lightgrey", ax=ax)

    legend_handles = [
        mpatches.Patch(color="steelblue",   label="High-status employed"),
        mpatches.Patch(color="lightblue",   label="High-status unemployed"),
        mpatches.Patch(color="tomato",      label="Low-status employed"),
        mpatches.Patch(color="lightsalmon", label="Low-status unemployed"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7)
    ax.set_title(f"Network at Step {step_count.value}\n(edge thickness = tie strength)", fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_time_series(model):
    """Three-panel time series: employment rates, gap, and network structure."""
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        return plt.figure()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: Employment rates
    axes[0].plot(df["Employment_Rate_High"], label="High-status", color="steelblue")
    axes[0].plot(df["Employment_Rate_Low"],  label="Low-status",  color="tomato")
    axes[0].fill_between(df.index,
                         df["Employment_Rate_High"],
                         df["Employment_Rate_Low"],
                         alpha=0.1, color="purple")
    axes[0].set_title("Employment Rate Over Time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    # Panel 2: Employment gap with rolling mean to smooth small-population noise
    gap = df["Employment_Gap"]
    axes[1].plot(gap, color="purple", alpha=0.25, linewidth=0.8, label="Raw")
    axes[1].plot(gap.rolling(window=10, min_periods=1).mean(),
                 color="purple", linewidth=2, label="10-step mean")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Employment Gap (High minus Low)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gap")
    axes[1].legend(fontsize=7)

    # Panel 3: Network structure co-evolution
    axes[2].plot(df["Cross_Group_Ties"],       label="Cross-group ties",     color="green")
    axes[2].plot(df["Within_Group_Ties_High"], label="Within high-status",   color="steelblue", linestyle="--")
    axes[2].plot(df["Within_Group_Ties_Low"],  label="Within low-status",    color="tomato",    linestyle="--")
    axes[2].set_title("Network Structure Over Time")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Number of Ties")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    return fig


# Solara Page 

@solara.component
def Page():
    model = model_state.value

    with solara.Sidebar():
        solara.Markdown("## Parameters")
        solara.SliderInt(   "Number of Agents",              value=n_agents,             min=20,  max=200, step=10)
        solara.SliderFloat( "Homophily",                     value=homophily,            min=0.0, max=1.0, step=0.05)
        solara.SliderFloat( "Pct High-Status",               value=pct_high_status,      min=0.1, max=0.9, step=0.05)
        solara.SliderFloat( "Weak Tie Probability",          value=weak_tie_prob,        min=0.0, max=0.5, step=0.01)
        solara.SliderFloat( "Initial Employment (High)",     value=initial_employed,     min=0.0, max=1.0, step=0.05)
        solara.SliderFloat( "Initial Employment (Low)",      value=initial_employed_low, min=0.0, max=1.0, step=0.05)
        solara.SliderFloat( "Referral Willingness",          value=referral_willingness, min=0.0, max=1.0, step=0.05)
        solara.SliderFloat( "In-group Bias",                 value=ingroup_bias,         min=1.0, max=2.0, step=0.05)
        solara.SliderFloat( "Acceptance Probability",        value=acceptance_prob,      min=0.0, max=1.0, step=0.05)
        solara.SliderFloat( "Job Loss Probability",          value=job_loss_prob,        min=0.0, max=0.1, step=0.005)
        solara.SliderFloat( "New Tie Probability",           value=new_tie_prob,         min=0.0, max=0.2, step=0.01)
        solara.SliderFloat( "Tie Growth Rate",               value=tie_growth_rate,      min=0.0,  max=0.05, step=0.005)
        solara.SliderFloat( "Tie Decay Rate",                value=tie_decay_rate,       min=0.0, max=0.1, step=0.005)
        solara.SliderInt(   "Isolation Threshold (steps)",   value=isolation_threshold,  min=1,   max=20,  step=1)
        solara.SliderInt(   "Max Degree",                    value=max_degree,           min=5,   max=40,  step=1)

        solara.Markdown("---")

        def on_reset():
            global _cached_pos
            _cached_pos = {}   # force layout recompute for new graph
            model_state.set(make_model())
            step_count.set(0)

        def on_step():
            if model_state.value is None:
                model_state.set(make_model())
            model_state.value.step()
            step_count.set(step_count.value + 1)

        def on_run():
            if model_state.value is None:
                model_state.set(make_model())
            for _ in range(20):
                model_state.value.step()
            step_count.set(step_count.value + 20)

        solara.Button("Reset / Initialize", on_click=on_reset, color="primary")
        solara.Button("Step",               on_click=on_step)
        solara.Button("Run 20 Steps",       on_click=on_run)

    # Main panel
    solara.Markdown("# Job Referral Network Inequality Model")
    solara.Markdown(
        "Agents interact pairwise each step. Referrals emerge from **trust-weighted "
        "interactions** between neighbors. The network **co-evolves** with employment "
        "status via triadic-closure tie formation and unemployment-driven tie atrophy. "
        "Set *Initial Employment (Low)* equal to *High* to run the emergence robustness check."
    )

    if model is None:
        solara.Info("Click Reset / Initialize in the sidebar to begin.")
        return

    with solara.Row():
        solara.Markdown(
            f"**Step:** {step_count.value}  |  "
            f"**High-status employed:** {model._employment_rate_high():.1%}  |  "
            f"**Low-status employed:** {model._employment_rate_low():.1%}  |  "
            f"**Gap:** {model._employment_gap():.1%}  |  "
            f"**Cross-group ties:** {model._cross_group_ties()}  |  "
            f"**Cross-group referrals (this step):** {model.cross_group_referrals_accepted}  |  "
            f"**Mean tie strength:** {model._mean_tie_strength():.2f}"
        )

    with solara.Row():
        with solara.Column():
            solara.FigureMatplotlib(draw_network(model))
        with solara.Column():
            solara.FigureMatplotlib(draw_time_series(model))