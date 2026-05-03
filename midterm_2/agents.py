"""
agents.py — CultureAgent for Axelrod (1997) culture dissemination model.

Each agent occupies a fixed site on a grid and holds a "culture" represented
as a tuple of integer traits across F features, where each trait is drawn
from {0, 1, ..., q-1} (q = number of traits per feature).

Paper reference: Axelrod, R. (1997). The Dissemination of Culture: A Model
with Local Convergence and Global Polarization. Journal of Conflict Resolution,
41(2), 203-226.
"""

from mesa import Agent


class CultureAgent(Agent):
    """
    Represents a single site/village in the Axelrod culture model.

    Attributes
    culture : list[int]
        The agent's current cultural profile — a list of F integer traits,
        one per feature. Each trait is in {0, ..., q-1}.

    IMPLEMENTATION NOTE — the "active site" convention:
        Axelrod specifies (p. 209) that "the activated site, rather than its
        neighbor, is the one that may undergo change." This ensures equal
        expected activation across all sites, including edge/corner sites that
        have fewer neighbors. step() alters self.culture, it never the neighbor's.
    """

    def __init__(self, model, num_features: int, num_traits: int):
        """
        Parameters
        model : CultureModel
            The parent model instance.
        num_features : int
            F — number of cultural features (dimensions).
        num_traits : int
            q — number of possible trait values per feature (0 to q-1).
        """
        super().__init__(model)
        # Randomly initialise each feature to a trait value in {0, ..., q-1}
        self.culture = [
            self.model.random.randrange(num_traits)
            for _ in range(num_features)
        ]

    # Core social-influence step (Axelrod, p. 208-209, Steps 1 & 2)
    def step(self):
        """
        Execute one social-influence event with this agent as the active site.

        Procedure (verbatim from paper, p. 208):
            Step 2. With probability equal to their cultural similarity, these
            two sites interact. An interaction consists of selecting at random
            a feature on which the active site and its neighbor differ (if
            there is one) and changing the active site's trait on this feature
            to the neighbor's trait on this feature.

        IMPLEMENTATION NOTES:
        1. The neighbor is chosen by the model scheduler before step() is
           called; here i just read self.model.current_neighbor which the
           model sets immediately before activating this agent.  This avoids
           selecting a neighbor independently in each agent step and lets the
           model control the random draw (consistent with Axelrod's footnote 4).

        2. Axelrod (footnote 4, p. 209) gives an equivalent but more efficient
           algorithm: draw a random feature f; if c(s,f) == c(n,f) AND there
           exists at least one differing feature, then perform the interaction
           on a randomly chosen differing feature. This is mathematically
           equivalent to "probability = similarity" because the probability
           that a random feature f will be shared equals the cultural similarity.
           I use the explicit probability formulation for clarity and
           faithfulness to the verbal description, then select a differing
           feature at random.
        """
        neighbor = self.model.current_neighbor
        if neighbor is None:
            return  # should not happen; defensive guard

        F = len(self.culture)

        # Compute cultural similarity: fraction of features with equal traits
        shared = sum(
            1 for f in range(F) if self.culture[f] == neighbor.culture[f]
        )
        similarity = shared / F  # in [0.0, 1.0]

        # If identical (similarity = 1.0): interact but nothing changes.
        # If fully different (similarity = 0.0): no interaction.
        # Otherwise: interact with probability = similarity.

        if self.model.random.random() < similarity:
            # Find all features on which the two sites differ
            differing = [
                f for f in range(F) if self.culture[f] != neighbor.culture[f]
            ]
            if differing:
                # Pick one differing feature at random and copy neighbor's trait
                chosen_feature = self.model.random.choice(differing)
                self.culture[chosen_feature] = neighbor.culture[chosen_feature]

    # Utility
    def cultural_similarity(self, other: "CultureAgent") -> float:
        """
        Return fraction of features shared with another agent (0.0-1.0).
        """
        F = len(self.culture)
        return sum(1 for f in range(F) if self.culture[f] == other.culture[f]) / F

    def culture_tuple(self) -> tuple:
        """Return culture as a hashable tuple (used for region counting)."""
        return tuple(self.culture)
