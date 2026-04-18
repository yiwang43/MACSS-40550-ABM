import solara
from model import SchellingModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)

## Define agent portrayal
## MODIFICATION: encoding uses both color (racial/ethnic group) and
## marker size (income class) so both dimensions are visible on the grid.
##   Color:  blue = type 1 (group one),  red = type 0 (group two)
##   Size:   large (100) = high-income,  small (40) = low-income
def agent_portrayal(agent):
    return {
        "color": "tab:blue" if agent.type == 1 else "tab:red",
        "marker": "s",
        ## MODIFICATION: size encodes income class
        "size": 100 if agent.income == 1 else 40,
    }

## Enumerate variable parameters
## MODIFICATION: added sliders for the four new income/mobility parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": {
        "type": "SliderInt",
        "value": 30, "label": "Width",
        "min": 5, "max": 100, "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": 30, "label": "Height",
        "min": 5, "max": 100, "step": 1,
    },
    "density": {
        "type": "SliderFloat",
        "value": 0.7, "label": "Population Density",
        "min": 0.0, "max": 1.0, "step": 0.01,
    },
    "group_one_share": {
        "type": "SliderFloat",
        "value": 0.7, "label": "Share Type 1 Agents",
        "min": 0.0, "max": 1.0, "step": 0.01,
    },
    "radius": {
        "type": "SliderInt",
        "value": 1, "label": "Vision Radius",
        "min": 1, "max": 5, "step": 1,
    },
    ## MODIFICATION: income-stratified homophily thresholds
    "homophily_low": {
        "type": "SliderFloat",
        "value": 0.3, "label": "Homophily mean — low-income",
        "min": 0.0, "max": 1.0, "step": 0.05,
    },
    "homophily_high": {
        "type": "SliderFloat",
        "value": 0.6, "label": "Homophily mean — high-income",
        "min": 0.0, "max": 1.0, "step": 0.05,
    },
    ## MODIFICATION: share of agents assigned to high-income class
    "high_income_pc": {
        "type": "SliderFloat",
        "value": 0.3, "label": "Share high-income agents",
        "min": 0.0, "max": 1.0, "step": 0.05,
    },
    ## MODIFICATION: search budget for high-income selective mobility
    "search_budget": {
        "type": "SliderInt",
        "value": 10, "label": "High-income search budget",
        "min": 1, "max": 50, "step": 1,
    },
}

## Instantiate model
schelling_model = SchellingModel()

## Plot: overall share happy (base model reporter, kept for comparability)
HappyPlot = make_plot_component({"share_happy": "tab:green"})

## MODIFICATION: plot happiness separately by income class
IncomeHappyPlot = make_plot_component({
    "share_happy_high_income": "tab:blue",
    "share_happy_low_income":  "tab:orange",
})

## MODIFICATION: plot income dissimilarity index over time
DissimilarityPlot = make_plot_component({"income_dissimilarity": "tab:purple"})

## Space component
SpaceGraph = make_space_component(agent_portrayal, draw_grid=False)

## Assemble page
page = SolaraViz(
    schelling_model,
    components=[SpaceGraph, HappyPlot, IncomeHappyPlot, DissimilarityPlot],
    model_params=model_params,
    name="Schelling + Income Sorting",
)
page