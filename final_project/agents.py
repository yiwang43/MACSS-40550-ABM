"""
EMERGENCE SOURCE
Emergence comes from three interacting agent-level mechanisms:

1. ACTIVE PAIRWISE INTERACTION: Every step, each agent contacts a random
   neighbor and a referral interaction occurs. Whether the referral is passed
   depends on tie strength (trust) between the two agents, not a global BFS.
   Thousands of local decisions aggregate into a macro employment gap.

2. CO-EVOLVING NETWORK: Tie strength is not fixed. It grows when two agents
   interact repeatedly and decays when they do not. Employed agents also form
   new ties through triadic closure (friends-of-friends), so the employed
   cluster densifies from within. The network structure and employment status
   are mutually constitutive.

3. JOB LOSS + PATH DEPENDENCE: Employed agents can lose their job each step.
   This keeps dynamics live. Early stochastic employment events matter
   disproportionately because they seed the referral clusters that grow over
   time — small early differences compound into large, persistent gaps.

The macro outcome (the size and persistence of the employment gap) is NOT
derivable from the parameters analytically. It emerges from who happens to
interact with whom in early steps, whether a low-status agent forms a strong
tie with an employed high-status agent before the high-status cluster closes
in on itself, and whether job loss severs key bridging ties at critical moments.
This is true even when both groups start from the same employment level, so
homophily alone generates the gap from identical initial conditions.

"""

import mesa


class WorkerAgent(mesa.Agent):
    """
    A job-seeking or employed worker embedded in a social network.

    Each step the agent:
      1. Contacts one random neighbor and attempts a referral interaction.
      2. Potentially forms a new tie via triadic closure with a neighboring
         employed agent (workplace contact mechanism).
      3. If unemployed, increments unemployment counter and may have weak
         ties atrophy due to disuse.
      4. If employed, may lose the job with probability = model.job_loss_prob.

    Parameters:
    node_id : int
        The graph node this agent occupies (used for all network operations).
    group : str
        Social group, "high_status" or "low_status".
    employed : bool
        Current employment status.
    steps_unemployed : int
        Consecutive steps spent unemployed (resets on employment).
    referrals_given : int
        Total referral offers this agent has extended to neighbors.
    referral_offers_received : int
        Total referral offers this agent has received (offer, not necessarily
        accepted — tracked to distinguish network access from job outcomes).
    """

    def __init__(self, model, group, node_id):
        super().__init__(model)
        self.node_id                 = node_id   # graph node identifier
        self.group                   = group
        self.employed                = False
        self.steps_unemployed        = 0
        self.referrals_given         = 0
        self.referral_offers_received = 0

    # Core pairwise interaction 
    def interact_with_neighbor(self, neighbor):
        """
        Pairwise interaction between self and a chosen neighbor.

        If self is employed and neighbor is unemployed:
          - Self passes a referral with probability proportional to tie strength.
          - Same-group referrals receive a trust bonus controlled by
            model.ingroup_bias (>1 = in-group favoritism, 1 = no bias).
          - If the referral is passed, neighbor accepts with probability
            equal to model.acceptance_prob.
          - Cross-group acceptances are tallied on the model for reporting.
          - Tie strength between self and neighbor increases on every interaction.

        If both are unemployed:
          - No referral occurs but tie strength still grows from social contact,
            improving future referral probability if either becomes employed.

        Parameters
        neighbor : WorkerAgent
        """
        edge_data    = self.model.G.get_edge_data(self.node_id, neighbor.node_id)
        tie_strength = edge_data.get("weight", 0.5) if edge_data else 0.5

        # Referral attempt: only employed agents can refer unemployed neighbors
        if self.employed and not neighbor.employed:
            referral_prob = tie_strength * self.model.referral_willingness

            # In-group trust bonus controlled by the ingroup_bias parameter
            if self.group == neighbor.group:
                referral_prob = min(1.0, referral_prob * self.model.ingroup_bias)

            if self.random.random() < referral_prob:
                self.referrals_given                 += 1
                neighbor.referral_offers_received    += 1

                if self.random.random() < self.model.acceptance_prob:
                    neighbor.employed         = True
                    neighbor.steps_unemployed = 0

                    # Track cross-group referral acceptances for model reporter
                    if self.group != neighbor.group:
                        self.model.cross_group_referrals_accepted += 1

        # Tie strength grows on every interaction regardless of referral outcome or
        # employment status — contact itself builds familiarity and future trust
        new_strength = min(1.0, tie_strength + self.model.tie_growth_rate)
        if self.model.G.has_edge(self.node_id, neighbor.node_id):
            self.model.G[self.node_id][neighbor.node_id]["weight"] = new_strength

    def try_form_workplace_tie(self):
        """
        Employed agents may form a new tie with an employed stranger encountered
        through a shared contact (triadic closure / friends-of-friends).

        Restricting candidates to the 2-hop neighborhood is sociologically
        realistic — workplace introductions happen through mutual acquaintances,
        not random global encounters. It is also O(degree²) rather than O(n),
        keeping batch runs tractable.

        A max_degree cap prevents the network from saturating over long runs.
        Without it, every employed agent eventually connects to every other,
        erasing the homophily signal that drives the research question.

        Homophily biases tie formation within the 2-hop candidate pool:
        with probability = homophily the agent draws from same-group candidates,
        otherwise from cross-group candidates.
        """
        if not self.employed:
            return

        # Degree cap: stop forming new ties once socially saturated
        if self.model.G.degree(self.node_id) >= self.model.max_degree:
            return

        # Build friends-of-friends set: neighbors' neighbors not already connected
        neighbors = set(self.model.G.neighbors(self.node_id))
        fof = set()
        for nb in neighbors:
            for nb2 in self.model.G.neighbors(nb):
                if nb2 != self.node_id and nb2 not in neighbors:
                    fof.add(nb2)

        # Filter to employed agents in the fof pool
        candidates = []
        for n in fof:
            cell = self.model.grid.get_cell_list_contents([n])
            if cell and cell[0].employed:
                candidates.append(cell[0])

        if not candidates:
            return

        same_group = [a for a in candidates if a.group == self.group]
        diff_group = [a for a in candidates if a.group != self.group]

        # Homophily biases which sub-pool to draw from
        if same_group and self.random.random() < self.model.homophily:
            pool = same_group
        elif diff_group:
            pool = diff_group
        else:
            pool = same_group

        if pool and self.random.random() < self.model.new_tie_prob:
            new_contact = self.random.choice(pool)
            self.model.G.add_edge(self.node_id, new_contact.node_id, weight=0.3)

    def decay_weak_ties(self):
        """
        Long-term unemployed agents lose tie strength with neighbors as
        social distance grows when their worlds diverge. Ties that fall
        below the minimum weight threshold are removed entirely, representing
        network atrophy from social isolation.

        Decay only begins after isolation_threshold steps of unemployment
        so that short spells of unemployment do not immediately sever ties.
        """
        if self.steps_unemployed < self.model.isolation_threshold:
            return

        for nb_id in list(self.model.G.neighbors(self.node_id)):
            edge_data = self.model.G.get_edge_data(self.node_id, nb_id)
            if edge_data is None:
                continue
            new_weight = edge_data.get("weight", 0.5) - self.model.tie_decay_rate
            if new_weight <= 0.05:
                self.model.G.remove_edge(self.node_id, nb_id)
            else:
                self.model.G[self.node_id][nb_id]["weight"] = new_weight

    # Agent step
    def step(self):
        """
        One agent time step with four sequential actions:

        1. Contact a random neighbor and interact (possible referral + tie growth).
        2. If employed, try to form a new workplace tie via triadic closure.
        3. If unemployed, increment counter and decay ties if past isolation threshold.
        4. If employed, roll for job loss.
        """

        # 1. Pairwise interaction with a random neighbor
        neighbors = list(self.model.G.neighbors(self.node_id))
        if neighbors:
            contact_id   = self.random.choice(neighbors)
            contact_list = self.model.grid.get_cell_list_contents([contact_id])
            if contact_list:
                self.interact_with_neighbor(contact_list[0])

        # 2. Workplace tie formation via triadic closure
        self.try_form_workplace_tie()

        # 3. Unemployment dynamics and tie atrophy
        if not self.employed:
            self.steps_unemployed += 1
            self.decay_weak_ties()

        # 4. Job loss so steps_unemployed resets to 0: a fresh unemployment spell begins
        if self.employed and self.random.random() < self.model.job_loss_prob:
            self.employed         = False
            self.steps_unemployed = 0
