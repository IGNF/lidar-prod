# Strategy to find optimal decision thresholds for Building validation

## Motivations

As described in section [Building Validation](./production_process.md) of the production process, the decision to validate or not a group of candidate buildings is based on several decision thresholds. Those thresholds represents different levels of confidence, for different sources of data. 

They may depend on the AI model which produces the probabilities as well as on the rule-based classification from which the clusters of candidates are derived. They are highly coupled. For instance, if a lower probability is required at the point level to be confirmed as a building (threshold `C1`), we might require a higher percentage of confirmed points in a cluster of candidates (thresholds `C2`) to validate it. There must therefore be optimized jointly. 

These thresholds define how much we automate decisions, but also the quantity of errors we may introduce: there is a balance to be found between `recall` (proportion of buildings group that were confirmed), `precision` (proportion of buildings among confirmed groups), and `automation` (proportion of groups for which a decision was made i.e. that are not flagged as "unsure"). 

## Strategy

We approach the choice of decisions thresholds as a constrained multi-objectives hyperparameters optimization.
We use the [NSGA-II](https://doi.org/10.1109/4235.996017) algorithm from the optuna optimization library.

The constraints are defined empirically: recall>=98% and precision>=98%. The genetic algorithms search two maximize the three objectives, but focuses the search to solutions that meet these criteria.

After a chosen number of generations, the genetic algorithms outputs the [Pareto front](https://en.wikipedia.org/wiki/Pareto_front) i.e. the set of Pareto efficient solutions for which no objective criterion could be increased without another one being reduced. Among Pareto-efficient solutions compliant with the constraint, the final solution is the set of thresholds that maximizes the production of precision, recall, and automation.