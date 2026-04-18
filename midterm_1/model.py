from mesa import Model
from mesa.space import SingleGrid
from agents import SchellingAgent
from mesa.datacollection import DataCollector


def _dissimilarity_index(model):
    """
    MODIFICATION: Compute the income dissimilarity index D at each step.
    D measures the degree to which high- and low-income agents are unevenly
    distributed across local neighborhoods (Moore neighborhoods of each agent).
    D = 0 means perfect integration; D = 1 means complete income segregation.
    Used as a dependent variable to track whether the selective-mobility rule
    accelerates income sorting over and above racial segregation.
    """
    agents = list(model.agents)
    n_high = sum(1 for a in agents if a.income == 1)
    n_low  = len(agents) - n_high
    if n_high == 0 or n_low == 0:
        return 0.0
    total = 0.0
    for a in agents:
        neighbors = model.grid.get_neighbors(
            a.pos, moore=True, radius=a.model.radius, include_center=False
        )
        local_high = sum(1 for n in neighbors if n.income == 1)
        local_low  = len(neighbors) - local_high
        total += abs(local_high / n_high - local_low / n_low)
    return round(0.5 * total, 4)


class SchellingModel(Model):
    ## Define initiation, requiring all needed parameter inputs
    def __init__(
        self,
        width=30,
        height=30,
        density=0.7,
        desired_share_alike=0.5,   # kept for GUI label compatibility; see note below
        group_one_share=0.7,
        radius=1,
        ## MODIFICATION: new parameters for income-stratified homophily and mobility
        homophily_low=0.3,         # mean threshold for low-income agents
        homophily_high=0.6,        # mean threshold for high-income agents
        high_income_pc=0.3,        # share of agents assigned high-income class
        search_budget=10,          # candidate cells evaluated by high-income movers
        seed=None,
    ):
        ## Inherit seed trait from parent class and ensure seed is integer
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        ## Core grid/population parameters (unchanged from base model)
        self.width = width
        self.height = height
        self.density = density
        self.group_one_share = group_one_share
        self.radius = radius

        ## MODIFICATION: store income-stratified homophily parameters on model
        ## so agents can read them during __init__ (see agents.py)
        self.homophily_low  = homophily_low
        self.homophily_high = homophily_high
        self.high_income_pc = high_income_pc
        self.search_budget  = search_budget

        ## NOTE: `desired_share_alike` is no longer a single global threshold;
        ## each agent samples its own threshold from a class-specific distribution
        ## (see SchellingAgent.__init__). The parameter is kept in the signature
        ## for GUI compatibility but is not used in agent creation.

        ## Create grid
        self.grid = SingleGrid(width, height, torus=True)

        ## Instantiate global happiness tracker
        self.happy = 0

        ## Define data collector
        ## MODIFICATION: added income-stratified happiness reporters and
        ## the income dissimilarity index as a model-level outcome variable
        self.datacollector = DataCollector(
            model_reporters={
                "happy": "happy",
                "share_happy": lambda m: (
                    (m.happy / len(m.agents)) * 100 if len(m.agents) > 0 else 0
                ),
                ## MODIFICATION: % happy among high-income agents
                "share_happy_high_income": lambda m: (
                    sum(1 for a in m.agents if a.income == 1 and
                        self._is_happy(a)) /
                    max(1, sum(1 for a in m.agents if a.income == 1)) * 100
                ),
                ## MODIFICATION: % happy among low-income agents
                "share_happy_low_income": lambda m: (
                    sum(1 for a in m.agents if a.income == 0 and
                        self._is_happy(a)) /
                    max(1, sum(1 for a in m.agents if a.income == 0)) * 100
                ),
                ## MODIFICATION: income dissimilarity index
                "income_dissimilarity": _dissimilarity_index,
            }
        )

        ## Place agents randomly around the grid
        for _cont, pos in self.grid.coord_iter():
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < group_one_share else 0
                ## MODIFICATION: assign income class via Bernoulli draw
                income = 1 if self.random.random() < high_income_pc else 0
                self.grid.place_agent(SchellingAgent(self, agent_type, income), pos)

        ## Initialize datacollector
        self.datacollector.collect(self)

    def _is_happy(self, agent):
        """
        MODIFICATION: Re-evaluate happiness for an agent in place, used
        only inside lambda reporters (agents.happy is reset each step before
        the data collector runs, so we compute on demand here).
        """
        neighbors = self.grid.get_neighbors(
            agent.pos, moore=True, radius=self.radius, include_center=False
        )
        if not neighbors:
            return False
        similar = sum(1 for n in neighbors if n.type == agent.type)
        return (similar / len(neighbors)) >= agent.desired_share_alike

    ## Define a step: reset global happiness tracker, agents move, collect data
    def step(self):
        self.happy = 0
        self.agents.shuffle_do("move")
        self.datacollector.collect(self)
        ## Run model until all agents are happy
        self.running = self.happy < len(self.agents)