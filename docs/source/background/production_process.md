# Production process used to transform point clouds classification

The end goal of the tool is to edit the input (rules-based) classification as much as we confidently can, and to highlight remaining areas of uncertainty for human inspection.

**Input**: point cloud that went through a first geometric algorithm that identified `candidates building points` based on geometric rules (e.g. plane surfaces, above 1.5m of the ground, etc.), and for which a semantic segmentation model produced a point-level probability of being a building. The default name for this extra dimension is `building`. You can leverage this [package for aerial lidar deep learning segmentation](https://github.com/IGNF/lidar-deep-segmentation).

## A) Building Validation

**Goal**: Confirm or refute groups of candidate building points when possible, mark them as unsure elsewise.

1) Clustering of _candidate buildings points_ into connected components.
2) Point-level decision
   1) Identification of points with ambiguous probability: `high entropy` if entropy >= E1 
   2) Identification of points that are `overlayed` by a building vector from the database.
   3) Decision at the point-level based on probabilities : 
      1) `confirmed` if:
         1) p>=`C1`, or
         2) `overlayed` and p>= (`C1` * `Cr`), where `Cr` is a relaxation factor that reduces the confidence we require to confirm when a point overlayed by a building vector. 
      2) `refuted` if (1-p) >= `R1`
3) Group-level decision :
    1) Uncertain due to high entropy: if proportion of `high entropy` points >=  `E2`
    2) Confirmation: if proportion of `confirmed` points >= `C2` OR if proportion of `overlayed` points >= `O1`
    3) Refutation: if proportion of `refuted` points >= `R2` AND proportion of `overlayed` points < `O1`
    4) Uncertainty: elsewise (this is a safeguard: uncertain groups are supposed to be already captured via their entropy)
4) Update of the point cloud classification

Decision thresholds `E1`, `E2` , `C1`, `C2`, `R1`, `R2`, `O1` are chosen via a [multi-objective hyperparameter optimization](/background/building_validation_optimization.md) that aims to maximize automation, precision, and recall of the decisions. 
Current performances on a 15km² validation dataset, expressed as percentages of clusters, are:
- Automation=91%
- Precision=98.5%
- Recall=98.1%.

![](/img/LidarBati-BuildingValidationM7.1V2.0.png)

## B) Building Completion

**Goal**: Confirm points that were too isolated to make up a group but have high-enough probability nevertheless (e.g. walls)

Among  _candidate buildings points_ that have not been clustered in previous step due, identify those which nevertheless meet the requirement to be `confirmed`.
Cluster them together with previously confirmed building points in a relaxed, vertical fashion (higher tolerance, XY plan).
For each cluster, if some points were confirmed, the others are considered to belong to the same building, and are 
therefore confirmed as well.

![](/img/LidarBati-BuildingCompletion.png)


## C) Building Identification

**Goal**: Highlight potential buildings that were missed by the rule-based algorithm, for human inspection. 

Among points that were **not** _candidate buildings points_ identify those which meet the requirement to be `confirmed`, and cluster them.

This clustering defines a LAS extra dimensions (`Group`) which indexes newly found cluster that may be some missed buildings.

![](/img/LidarBati-BuildingIdentification.png)


