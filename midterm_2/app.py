"""
app.py — Solara/Mesa GUI for the Axelrod (1997) culture model.

Run with:
    solara run app.py
  (from the midterm_2 directory)

The visualization includes:
  1. Space component: each cell is colored by a deterministic hash of its
     cultural profile.  Cells sharing the SAME culture get the SAME color,
     so contiguous same-color patches directly show cultural regions.
  2. Stable-regions plot: tracks the count of homogeneous contiguous regions
     over simulated time (paper's Table 2 / Figure 2 outcome).
  3. Sliders/selectors for all variable parameters reported in the paper.

VISUALIZATION NOTE: color scheme vs. the paper's Figure 1:
    The paper's Figure 1 depicts cultural SIMILARITY between adjacent cells as
    edge shading (black ≤ 20%, white = 100%).  Drawing per-edge shading
    requires custom rendering not directly supported by Mesa's grid renderer.
    I instead color each CELL by its full cultural profile (via hash), which
    reveals the same regional structure: a stable region appears as a solid
    block of one color.  This is the standard alternative visualization used
    in replications of Axelrod's model.

IMPROVISATION NOTE: render_interval default:
    model.step() advances exactly ONE event (one site activation), faithful
    to the paper.  However, stability on a 10x10 grid arrives after ~70,000
    events, so I set render_interval=1000 by default so the GUI redraws
    every 1,000 events.  The user can change this
    via the built-in "Render Interval" slider in the Controls panel.
"""

import matplotlib.colors as mcolors

from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
    Slider,
)
from mesa.visualization.components import AgentPortrayalStyle

from model import CultureModel


# ---------------------------------------------------------------------------
# Color palette — visually distinct colors keyed by culture hash
# ---------------------------------------------------------------------------

# Golden-ratio spacing in HSV gives maximally distinct hues.
_N_COLORS = 256
_PALETTE_HEX = [
    mcolors.to_hex(mcolors.hsv_to_rgb(((i * 0.618033988749895) % 1.0, 0.75, 0.9)))
    for i in range(_N_COLORS)
]


# ---------------------------------------------------------------------------
# Agent portrayal
# ---------------------------------------------------------------------------

def agent_portrayal(agent) -> AgentPortrayalStyle:
    """
    Color each site by its cultural profile.

    Identical cultures → identical color → same-colored blobs = cultural
    regions.  Square markers (marker="s") fill the cell with no gaps,
    reproducing the grid appearance of the paper's Figure 1.
    """
    color = _PALETTE_HEX[hash(agent.culture_tuple()) % _N_COLORS]
    return AgentPortrayalStyle(
        color=color,
        marker="s",
        size=80,
        edgecolors="none",
    )


# ---------------------------------------------------------------------------
# Post-process: clean up axes for the grid view
# ---------------------------------------------------------------------------

def post_process_space(ax):
    """Remove tick marks; set equal aspect so cells are square."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cultural Regions (same color = same culture)", fontsize=9)


# ---------------------------------------------------------------------------
# Model parameters (all varied in the paper)
# ---------------------------------------------------------------------------

model_params = {
    # Fixed seed exposed as text input (matches boid/virus examples)
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    # Grid size — paper default 10×10; also tests 5×5 and 15×15
    "width": Slider(label="Grid Width",  value=10, min=5, max=30, step=5),
    "height": Slider(label="Grid Height", value=10, min=5, max=30, step=5),
    # Cultural complexity — paper varies F ∈ {5,10,15} and q ∈ {5,10,15}
    "num_features": Slider(
        label="Features (F)", value=5, min=2, max=15, step=1
    ),
    "num_traits": Slider(
        label="Traits per Feature (q)", value=10, min=2, max=15, step=1
    ),
    # Neighborhood size — paper tests 4, 8, 12
    "neighborhood_size": {
        "type": "Select",
        "value": 4,
        "values": [4, 8, 12],
        "label": "Neighborhood Size (4/8/12)",
    },
    # Boundary topology — paper default is bounded (torus=False);
    # also tests toroidal wrapping (p. 214-215)
    "torus": {
        "type": "Select",
        "value": False,
        "values": [False, True],
        "label": "Toroidal Grid",
    },
}


# Plot component: stable regions over time
#
# IMPROVISATION NOTE — x-axis label:
#   make_plot_component uses Mesa's internal collection index as the x-axis
#   (labeled "Step" by default). Each collection = 1 sweep = width × height
#   events, so the values shown are sweep counts, not raw event counts.
#   I relabel the axis to make this clear.

def post_process_plot(ax):
    # NOTE: ax.set_xlabel() is intentionally omitted.
    # Mesa 3. version resets the x-axis label to "Step" after post_process returns,
    # silently overriding any custom xlabel. The y-label and title are not
    # reset by Mesa, so those apply correctly. The x-axis will always read
    # "Step" (= sweep index, where 1 sweep = width × height events).
    ax.set_ylabel("# Cultural Regions")
    ax.set_title("Cultural Regions Over Time")


RegionsPlot = make_plot_component("StableRegions", post_process=post_process_plot)

# Assemble the Solara page

# Create initial model instance 
model = CultureModel()

# Build space renderer with matplotlib backend (cleaner for fixed grids)
renderer = SpaceRenderer(model, backend="matplotlib")
renderer.draw_agents(agent_portrayal=agent_portrayal)
renderer.post_process = post_process_space

page = SolaraViz(
    model,
    renderer,
    components=[RegionsPlot],
    model_params=model_params,
    name="Axelrod (1997) — Culture Dissemination",
    # render_interval=1000 means the GUI redraws every 1,000 steps.
    # Each step = 1 event, so this ~ 1,000 events between visual updates.
    # Keeps the GUI responsive without changing the model's event semantics.
    render_interval=1000,
)
page  # noqa: required by `solara run`
