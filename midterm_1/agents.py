from mesa import Agent

class SchellingAgent(Agent):
    ## Initiate agent instance, inherit model trait from parent class
    def __init__(self, model, agent_type, income):
        super().__init__(model)
        ## Set agent type (racial/ethnic group, as in base model)
        self.type = agent_type
        ## MODIFICATION: assign income class (0 = low-income, 1 = high-income)
        self.income = income
        ## MODIFICATION: draw agent-specific homophily threshold from a class-specific normal distribution, clipped to [0, 1].
        ## High-income agents draw from a distribution centered at model.homophily_high; low-income from model.homophily_low.
        ## This creates within-class variance while preserving cross-class differences in average preferences.
        mean_h = model.homophily_high if income == 1 else model.homophily_low
        raw_h = model.random.gauss(mean_h, 0.10)
        self.desired_share_alike = max(0.0, min(1.0, raw_h))

    ## Define basic decision rule
    def move(self):
        ## Get list of neighbors within range of sight
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, radius=self.model.radius, include_center=False
        )
        ## Count neighbors of same type as self
        similar_neighbors = len([n for n in neighbors if n.type == self.type])
        ## If an agent has any neighbors, calculate share of same type
        if (valid_neighbors := len(neighbors)) > 0:
            share_alike = similar_neighbors / valid_neighbors
        else:
            share_alike = 0

        ## If unhappy, move using income-stratified mobility rule
        if share_alike < self.desired_share_alike:
            ## MODIFICATION: income-stratified mobility rule. High-income agents (income == 1) do not move randomly.
            ## Instead, they sample `search_budget` candidate empty cells and relocate to the one whose current neighbors contain the highest share of high-income agents
            ## for example, they sort "upward" into wealthier areas.
            ## Low-income agents fall back to the base model's random move, representing constrained residential mobility.
            if self.income == 1:
                empty_cells = list(self.model.grid.empties)
                if not empty_cells:
                    return  # nowhere to go; stay put
                ## Sample up to search_budget candidate destinations
                n = min(self.model.search_budget, len(empty_cells))
                candidates = self.random.sample(empty_cells, n)
                ## Score each candidate by share of high-income neighbors
                best_pos = max(candidates, key=self._high_income_score)
                self.model.grid.move_agent(self, best_pos)
            else:
                ## Low-income: random move (identical to base model)
                self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1

    ## MODIFICATION: helper to score a candidate cell by high-income neighbor share
    def _high_income_score(self, pos):
        """
        Return the fraction of neighbors at `pos` who are high-income.
        Used by high-income agents to select a preferred destination cell.
        A score of 0 is returned if the candidate cell has no neighbors.
        """
        neighbors = self.model.grid.get_neighbors(
            pos, moore=True, radius=self.model.radius, include_center=False
        )
        if not neighbors:
            return 0.0
        return sum(1 for n in neighbors if n.income == 1) / len(neighbors)