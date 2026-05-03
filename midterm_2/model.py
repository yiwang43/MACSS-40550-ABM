"""
model.py for CultureModel implementing Axelrod (1997).

Key parameters (all varied in the paper):
  - width, height : grid dimensions (paper uses 10 x 10 as default)
  - num_features  : F, number of cultural features (paper: 5, 10, 15)
  - num_traits    : q, trait alternatives per feature (paper: 5, 10, 15)
  - neighborhood  : von Neumann-4 (paper default), Moore-8, or extended-12

IMPLEMENTATION NOTES:
  Scheduling (one event = one random activation):
      Axelrod (footnote 5, p. 209) explicitly states: "The simulation is done
      one event at a time to avoid any artifacts of synchronous activation."
      Mesa's default RandomActivation scheduler activates ALL agents each
      step, which is synchronous and inconsistent with the paper.

      I therefore implement a custom "one-event-per-step" scheduler: in
      each model step: (1) draw one random agent, (2) draw one random
      neighbor for that agent, and (3) call only that agent's step().
      This replicates the paper's asynchronous, event-driven logic.

  Torus/boundary:
      The paper uses a bounded grid with absorbing boundaries (p. 208):
      "Sites on the edge of the map have only three neighbors, and sites in
      the corners have only two neighbors."  We default to torus=False.
      The paper also tests a toroidal variant (p. 214-215) and finds the same
      qualitative pattern; this is available via the `torus` parameter.

  Termination / stability detection:
      The paper runs for a fixed number of events but also notes the process
      reaches a stable absorbing state.  For the GUI we run indefinitely and
      expose a `is_stable` flag once no pair of adjacent sites can interact
      (i.e., every neighbouring pair is identical or fully different).
"""

import mesa
from mesa import DataCollector
from mesa.space import SingleGrid

try:
    from .agents import CultureAgent  # package import (e.g., solara run)
except ImportError:
    from agents import CultureAgent  # direct script execution


def count_stable_regions(model: "CultureModel") -> int:
    """
    Count distinct contiguous regions of identical culture (BFS/flood-fill).

    A "stable region" in the paper (p. 210) is a maximal connected set of
    sites sharing an identical culture, where connectivity is defined by the
    same neighborhood topology used for interactions.

    IMPLEMENTATION NOTE: The paper defines regions as groups of contiguous
    sites with IDENTICAL culture (not just similar).  I use 4-connectivity
    (von Neumann neighborhood) for region counting regardless of the
    interaction neighborhood size, following the paper's visual description
    (Figure 1 and surrounding text), which uses side-adjacent cells for
    region delineation.
    """
    visited = set()
    regions = 0
    agents = {agent.pos: agent for agent in model.agents}

    for pos, agent in agents.items():
        if pos in visited:
            continue
        # BFS over sites with identical culture
        queue = [pos]
        visited.add(pos)
        regions += 1
        while queue:
            cx, cy = queue.pop()
            # 4-connected neighbours
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if (nx, ny) in visited or model.grid.out_of_bounds((nx, ny)):
                    continue
                neighbor_agent = agents.get((nx, ny))
                if neighbor_agent and neighbor_agent.culture == agent.culture:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return regions


def check_stability(model: "CultureModel") -> bool:
    """
    Return True if the model has reached an absorbing state.

    Absorbing state (paper p. 210): every pair of neighboring sites is either
    identical (similarity=1, no change possible) or completely different
    (similarity=0, no interaction possible).

    IMPLEMENTATION NOTE: uses the model's actual interaction neighborhood:
        We delegate to model._get_neighbors() so the stability check respects
        neighborhood_size (4, 8, or 12).  Hardcoding 4-connectivity here would
        prematurely declare stability when neighborhood_size=8 or 12, because
        diagonal / extended neighbors with partial similarity can still interact
        and would go unchecked.
    """
    for agent in model.agents:
        for neighbor in model._get_neighbors(agent):
            if 0.0 < agent.cultural_similarity(neighbor) < 1.0:
                return False  # at least one active boundary exists
    return True


class CultureModel(mesa.Model):
    """
    Axelrod (1997) culture dissemination model.

    Parameters
    width, height : int
        Grid dimensions.  Paper default: 10x10.
    num_features : int
        Number of cultural features (F).  Paper varies: 5, 10, 15.
    num_traits : int
        Traits per feature (q).  Paper varies: 5, 10, 15.
    neighborhood_size : int
        4 = von Neumann (paper default), 8 = Moore, 12 = extended diamond.
    torus : bool
        Whether grid wraps (paper default False, also tests True).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        num_features: int = 5,
        num_traits: int = 10,
        neighborhood_size: int = 4,
        torus: bool = False,
        seed=None,
    ):
        super().__init__(rng=seed) 

        self.width = width
        self.height = height
        self.num_features = num_features
        self.num_traits = num_traits
        self.neighborhood_size = neighborhood_size

        # Fixed grid — no movement in the model (paper p. 208)
        self.grid = SingleGrid(width, height, torus=torus)

        # current_neighbor is set by the model before activating an agent
        # so that agents.step() can read it without re-drawing
        self.current_neighbor: CultureAgent | None = None

        # Track total events (= number of (site, neighbor) pairs drawn)
        self.event_count: int = 0

        # Stability flag — set once no further change is possible
        self.is_stable: bool = False

        # Place agents 
        for x in range(width):
            for y in range(height):
                agent = CultureAgent(self, num_features, num_traits)
                self.grid.place_agent(agent, (x, y))

        # Data collection 
        # IMPLEMENTATION NOTE: sampling frequency:
        #   count_stable_regions() does an O(N) BFS every call. Running it
        #   once per event (= 100 times per "round" on a 10×10 grid) is
        #   expensive and produces a noisy per-event plot.  We instead collect
        #   one sample per sweep (width × height events), matching the
        #   "Events/Site" time axis used in the paper's Figure 3 and keeping
        #   the GUI responsive.  The _sweep_count tracks how many full sweeps
        #   have elapsed and drives the x-axis of the plot.
        self._sweep_count: int = 0
        self.datacollector = DataCollector(
            model_reporters={
                "StableRegions": count_stable_regions,
                "EventCount": lambda m: m.event_count,
                "IsStable": lambda m: int(m.is_stable),
            }
        )

    # Neighbor selection helpers
    def _get_neighbors(self, agent: CultureAgent):
        """
        Return the list of neighboring agents for a given agent, respecting
        the chosen neighborhood_size.

        Neighborhood sizes (paper p. 212-213):
          4  — von Neumann (N, E, S, W)                      [paper default]
          8  — Moore (adds NE, NW, SE, SW diagonals)
          12 — Extended diamond (Moore-8 + N2, E2, S2, W2)

        IMPLEMENTATION NOTE:
            Mesa's get_neighbors(moore=True/False) only supports 4 and 8.
            For the 12-neighbor "diamond" neighborhood we implement manually:
            the 8 Moore neighbors plus the 4 sites two steps along cardinal
            directions (as described in the paper, p. 212).
        """
        x, y = agent.pos

        if self.neighborhood_size == 4:
            # von Neumann: 4 cardinal directions only
            return list(
                self.grid.get_neighbors((x, y), moore=False, include_center=False)
            )
        elif self.neighborhood_size == 8:
            # Moore: 8 surrounding cells
            return list(
                self.grid.get_neighbors((x, y), moore=True, include_center=False)
            )
        elif self.neighborhood_size == 12:
            # Extended diamond: Moore-8 + 4 two-step cardinal sites
            moore_8 = set(
                self.grid.get_neighbors((x, y), moore=True, include_center=False)
            )
            extra_positions = [(x+2, y), (x-2, y), (x, y+2), (x, y-2)]
            extra = []
            for pos in extra_positions:
                if not self.grid.out_of_bounds(pos):
                    cell_contents = self.grid.get_cell_list_contents([pos])
                    extra.extend(cell_contents)
            return list(moore_8) + extra
        else:
            raise ValueError(
                f"neighborhood_size must be 4, 8, or 12; got {self.neighborhood_size}"
            )

    # One event = one (active site, neighbor) pair
    def step(self):
        """
        Advance the model by ONE event (one random site activation).

        Axelrod (p. 208, Step 1): "At random, pick a site to be active, and
        pick one of its neighbors."

        IMPLEMENTATION NOTE on event granularity:
            The paper counts time in single events.  One call to model.step()
            here corresponds to exactly ONE such event, matching the paper's
            notation (e.g., "after 20,000 events" in Figure 1).  The GUI
            advances multiple events per visual frame; see server.py for the
            steps_per_second / advance logic.

        If the model has already reached stability, this method is a no-op.
        """
        if self.is_stable:
            return

        # Step 1: draw a random active site
        all_agents = list(self.agents)
        active_agent = self.random.choice(all_agents)

        # Step 1 (cont.): draw one random neighbor
        neighbors = self._get_neighbors(active_agent)
        if not neighbors:
            # Corner/edge agent with no neighbors (shouldn't happen in normal grids)
            return

        chosen_neighbor = self.random.choice(neighbors)
        self.current_neighbor = chosen_neighbor

        # Step 2: let the active agent attempt interaction
        active_agent.step()
        self.event_count += 1

        # Once per sweep (width × height events): check stability and collect
        # a data sample.  This avoids calling the O(N) BFS on every single
        # event, keeping the GUI fast and the plot readable (one point per
        # sweep rather than one per event).
        if self.event_count % (self.width * self.height) == 0:
            self.is_stable = check_stability(self)
            self._sweep_count += 1
            self.datacollector.collect(self)

    def run_until_stable(self, max_events: int = 10_000_000):
        """
        Run the model until it reaches a stable absorbing state.

        Parameters
        ----------
        max_events : int
            Safety cap to avoid infinite loops on very large grids.
        """
        while not self.is_stable and self.event_count < max_events:
            self.step()
        return self.event_count
