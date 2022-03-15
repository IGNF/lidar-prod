from dataclasses import dataclass
import logging
import os
import os.path as osp
import pdal

log = logging.getLogger(__name__)


class BuildingIdentifier:
    """Logic of building validation.
    
    Points that were not found by rule-based algorithms are clustered but with a high-enough deep learning probability of
    being a building are clustered into candidate groups of buildings.

    High enough means :
    - p>=min_building_proba
    OR if point fall in a building vector from the BDUni:
    - p>=(min_building_proba*min_building_proba_multiplier_if_bd_uni_overlay).
    """

    def __init__(
        self,
        candidate_buildings_codes: int = [202],
        min_building_proba: float = 0.75,
        min_building_proba_multiplier_if_bd_uni_overlay: float = 0.5,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.candidate_buildings_codes = candidate_buildings_codes
        self.data_format = data_format
        self.min_building_proba = min_building_proba
        self.min_building_proba_multiplier_if_bd_uni_overlay = min_building_proba_multiplier_if_bd_uni_overlay

    def run(self, in_f: str, out_f: str):
        """Application.

        Transform cloud at `in_f` following identification logic, and save it to
        `out_f`

        Args:
            in_f (str): path to input LAS file with a building probability channel
            out_f (str): path for saving updated LAS file.

        Returns:
            _type_: returns `out_f` for potential terminal piping.

        """
        log.info(f"Applying Building Identification to file \n{in_f}")
        log.info("Clustering of points with high building proba")
        self.update(in_f, out_f)
        return out_f

    def update(self, in_f: str, out_f: str):
        """Identify potential buildings in a new channel, excluding former candidates from
        search based on their group ID. ClusterID needs to be reset to avoid unwanted merge
        of information from previous VuildingValidation clustering.

        """
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader(
            in_f,
            type="readers.las",
            nosrs=True,
            override_srs=self.data_format.crs_prefix + str(self.data_format.crs),
        )
        non_candidates = (
            "("
            + f"{self.data_format.las_channel_names.macro_candidate_building_groups} == 0"
            + ")"
        )
        p_heq_threshold = f"(building>={self.min_building_proba})"
        p_heq_threshold_under_bd_uni = f"(building>={self.min_building_proba * self.min_building_proba_multiplier_if_bd_uni_overlay})"
        where =  f"{non_candidates} && ({p_heq_threshold} | {p_heq_threshold_under_bd_uni})"
        pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        pipeline |= pdal.Filter.ferry(
            dimensions=f"{self.data_format.las_channel_names.cluster_id}=>{self.data_format.las_channel_names.ai_building_identified}"
        )
        pipeline |= pdal.Writer(
            type="writers.las", filename=out_f, forward="all", extra_dims="all"
        )
        os.makedirs(osp.dirname(out_f), exist_ok=True)
        pipeline.execute()
