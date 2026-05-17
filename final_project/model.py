"""
Job Referral Network Inequality Model — Model Definition
=========================================================
Two groups of workers are embedded in a homophilic social network.
Each step, agents interact pairwise: employed agents refer unemployed
neighbors, tie strength co-evolves with interaction history, and
long-term unemployment causes network atrophy.

The macro outcome — a persistent employment gap between high-status and
low-status groups — emerges from local referral dynamics even when both
groups start from equal employment levels, demonstrating true emergence
rather than initial-condition inheritance.

Authors: Yi Wang
Course:  MACS 40550 Agent-Based Modeling
"""

import networkx as nx
import numpy as np
import mesa
from mesa.space import NetworkGrid
from mesa import DataCollector

from agents import WorkerAgent


class JobReferralModel(mesa.Model):
    """
    Job Referral Network Inequality Model.

    Agents actively interact with neighbors each step. Referrals emerge from
    pairwise trust-weighted interactions rather than top-down BFS propagation.
    The network co-evolves with employment status via triadic-closure tie
    formation and unemployment-driven tie atrophy.

    Parameters
    ----------
    n_agents : int
        Total number of agents.
    homophily : float [0, 1]
        Probability that a new tie forms within the same group.
    pct_high_status : float [0, 1]
        Proportion of agents in the high_status group.
    initial_employed_pct : float [0, 1]
        Share of high_status agents who start employed.
    initial_employed_pct_low : float [0, 1]
        Share of low_status agents who start employed. Default 0 preserves the
        canonical inequality story; set equal to initial_employed_pct for a
        robustness check proving the gap emerges from network dynamics alone.
    weak_tie_prob : float [0, 1]
        Probability of seeding an initial cross-group weak tie per agent.
    referral_willingness : float [0, 1]
        Base probability an employed agent passes a referral to an unemployed
        neighbor, scaled by tie strength each interaction.
    ingroup_bias : float [1, ∞)
        Multiplier applied to referral_prob for same-group pairs. 1.0 = no
        in-group favoritism; values >1 encode trust-based in-group preference.
        Default 1.2 (calibrated to produce moderate bias consistent with
        audit-study evidence; see Pager 2003).
    acceptance_prob : float [0, 1]
        Probability an unemployed agent accepts a referral when offered.
    job_loss_prob : float [0, 1]
        Probability an employed agent loses their job each step.
    new_tie_prob : float [0, 1]
        Probability an employed agent forms a new workplace tie per step.
    tie_growth_rate : float
        Amount tie strength increases per active interaction (max 1.0).
        Default 0.02 keeps variance alive across a 60-step run: ties reach
        ~0.7 at equilibrium rather than saturating at 1.0 by step 200.
    tie_decay_rate : float
        Tie strength decrease per step for long-term unemployed agents.
        Social isolation is the only source of tie decay — employed agents
        maintain their ties through continued interaction.
    isolation_threshold : int
        Steps unemployed before tie decay begins.
    max_degree : int
        Maximum number of ties an agent will form. Caps network density so
        homophily's effect on segregation persists at long run rather than
        washing out as the graph approaches full connectivity.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_agents                 = 100,
        homophily                = 0.8,
        pct_high_status          = 0.5,
        initial_employed_pct     = 0.3,
        initial_employed_pct_low = 0.0,
        weak_tie_prob            = 0.05,
        referral_willingness     = 0.6,
        ingroup_bias             = 1.2,
        acceptance_prob          = 0.5,
        job_loss_prob            = 0.01,
        new_tie_prob             = 0.05,
        tie_growth_rate          = 0.02,
        tie_decay_rate           = 0.03,
        isolation_threshold      = 5,
        max_degree               = 15,
        seed                     = None,
    ):
        super().__init__(seed=seed)

        self.n_agents                 = n_agents
        self.homophily                = homophily
        self.pct_high_status          = pct_high_status
        self.initial_employed_pct     = initial_employed_pct
        self.initial_employed_pct_low = initial_employed_pct_low
        self.weak_tie_prob            = weak_tie_prob
        self.referral_willingness     = referral_willingness
        self.ingroup_bias             = ingroup_bias
        self.acceptance_prob          = acceptance_prob
        self.job_loss_prob            = job_loss_prob
        self.new_tie_prob             = new_tie_prob
        self.tie_growth_rate          = tie_growth_rate
        self.tie_decay_rate           = tie_decay_rate
        self.isolation_threshold      = isolation_threshold
        self.max_degree               = max_degree

        # Cumulative count of cross-group referral acceptances (tracked in agents)
        self.cross_group_referrals_accepted = 0

        # Build network and grid
        self.G    = nx.Graph()
        self._build_network()
        self.grid = NetworkGrid(self.G)
        self._create_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "Employment_Rate_High":           self._employment_rate_high,
                "Employment_Rate_Low":            self._employment_rate_low,
                "Employment_Gap":                 self._employment_gap,
                "Mean_Tie_Strength":              self._mean_tie_strength,
                "Cross_Group_Ties":               self._cross_group_ties,
                "Within_Group_Ties_High":         self._within_group_ties_high,
                "Within_Group_Ties_Low":          self._within_group_ties_low,
                "Mean_Steps_Unemployed_Low":      self._mean_steps_unemployed_low,
                "Cross_Group_Referrals_Accepted": lambda m: m.cross_group_referrals_accepted,
            },
            agent_reporters={
                "Employed":                "employed",
                "Group":                   "group",
                "Steps_Unemployed":        "steps_unemployed",
                "Referrals_Given":         "referrals_given",
                "Referral_Offers_Received": "referral_offers_received",
                "Degree":                  lambda a: self.G.degree(a.node_id),
            },
        )

        self.running = True
        self.datacollector.collect(self)

    # ── Network construction ───────────────────────────────────────────────

    def _build_network(self):
        """
        Build a homophilic weighted network.

        Each agent attempts avg_degree tie formations. Each tie is formed
        within-group with probability = homophily, cross-group otherwise.
        Initial within-group ties start with weight 0.5 (moderate familiarity).
        Seeded cross-group weak ties start with weight 0.2 (low familiarity).
        """
        avg_degree = 6

        n_high               = int(self.n_agents * self.pct_high_status)
        self.high_nodes      = list(range(n_high))
        self.low_nodes       = list(range(n_high, self.n_agents))
        self._high_nodes_set = set(self.high_nodes)   # O(1) membership test

        self.G.add_nodes_from(self.high_nodes, group="high_status")
        self.G.add_nodes_from(self.low_nodes,  group="low_status")

        for node in range(self.n_agents):
            is_high    = node in self._high_nodes_set
            same_group = self.high_nodes if is_high else self.low_nodes
            diff_group = self.low_nodes  if is_high else self.high_nodes

            for _ in range(avg_degree // 2):
                pool = ([n for n in same_group if n != node]
                        if self.random.random() < self.homophily
                        else diff_group)
                if pool:
                    target = self.random.choice(pool)
                    if not self.G.has_edge(node, target):
                        self.G.add_edge(node, target, weight=0.5)

        # Seed cross-group weak ties
        for node in range(self.n_agents):
            if self.random.random() < self.weak_tie_prob:
                is_high    = node in self._high_nodes_set
                diff_group = self.low_nodes if is_high else self.high_nodes
                if diff_group:
                    target = self.random.choice(diff_group)
                    if not self.G.has_edge(node, target):
                        self.G.add_edge(node, target, weight=0.2)

    # ── Agent creation ─────────────────────────────────────────────────────

    def _create_agents(self):
        """
        Instantiate agents with group-specific initial employment rates.
        Setting initial_employed_pct_low > 0 allows robustness checks where
        both groups start from equal footing, isolating the network mechanism.
        """
        for node_id in self.G.nodes():
            group = self.G.nodes[node_id]["group"]
            agent = WorkerAgent(self, group, node_id)
            if group == "high_status":
                agent.employed = self.random.random() < self.initial_employed_pct
            else:
                agent.employed = self.random.random() < self.initial_employed_pct_low
            self.grid.place_agent(agent, node_id)

    # ── Model step ────────────────────────────────────────────────────────

    def step(self):
        """
        Advance one model step.
        shuffle_do activates each agent in random order each tick.
        Tie decay is handled at the agent level in decay_weak_ties(),
        called from WorkerAgent.step() for long-term unemployed agents only.
        cross_group_referrals_accepted is reset each step so the reporter
        captures a per-step flow, not a cumulative stock.
        Data is collected last.
        """
        self.cross_group_referrals_accepted = 0
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    # ── Reporter helpers ───────────────────────────────────────────────────

    def _get_agents_by_group(self, group):
        return [a for a in self.agents if a.group == group]

    def _employment_rate_high(self):
        agents = self._get_agents_by_group("high_status")
        return sum(a.employed for a in agents) / len(agents) if agents else 0.0

    def _employment_rate_low(self):
        agents = self._get_agents_by_group("low_status")
        return sum(a.employed for a in agents) / len(agents) if agents else 0.0

    def _employment_gap(self):
        return self._employment_rate_high() - self._employment_rate_low()

    def _mean_tie_strength(self):
        weights = [d["weight"] for _, _, d in self.G.edges(data=True) if "weight" in d]
        return float(np.mean(weights)) if weights else 0.0

    def _cross_group_ties(self):
        return sum(
            1 for u, v in self.G.edges()
            if self.G.nodes[u].get("group") != self.G.nodes[v].get("group")
        )

    def _within_group_ties_high(self):
        return sum(
            1 for u, v in self.G.edges()
            if self.G.nodes[u].get("group") == "high_status"
            and self.G.nodes[v].get("group") == "high_status"
        )

    def _within_group_ties_low(self):
        return sum(
            1 for u, v in self.G.edges()
            if self.G.nodes[u].get("group") == "low_status"
            and self.G.nodes[v].get("group") == "low_status"
        )

    def _mean_steps_unemployed_low(self):
        agents = [a for a in self._get_agents_by_group("low_status") if not a.employed]
        return float(np.mean([a.steps_unemployed for a in agents])) if agents else 0.0
