from dataclasses import dataclass
import logging
import os
import os.path as osp
import pdal

log = logging.getLogger(__name__)


class BuildingIdentifier:
    """Logic of building validation.

    Points that were not found by rule-based algorithms but which have a high-enough probability of
    being a building are clustered into candidate groups of buildings.

    High enough probability means :
    - p>=min_building_proba
    OR, IF point fall in a building vector from the BDUni:
    - p>=(min_building_proba*min_frac_confirmation_factor_if_bd_uni_overlay).
    """

    def __init__(
        self,
        min_building_proba: float = 0.75,
        min_frac_confirmation_factor_if_bd_uni_overlay: float = 1.0,
        cluster=None,
        data_format=None,
    ):
        self.cluster = cluster
        self.data_format = data_format
        self.min_building_proba = min_building_proba
        self.min_frac_confirmation_factor_if_bd_uni_overlay = (
            min_frac_confirmation_factor_if_bd_uni_overlay
        )

    def run(self, in_f: str, out_f: str) -> str:
        """Application.

        Transform cloud at `in_f` following identification logic, and save it to
        `out_f`

        Args:
            in_f (str): path to input LAS file with a building probability channel
            out_f (str): path for saving updated LAS file.

        Returns:
            str:  `out_f`

        """
        log.info(f"Applying Building Identification to file \n{in_f}")
        log.info("Clustering of points with high building proba.")
        self.prepare(in_f, out_f)
        return out_f

    def prepare(self, in_f: str, out_f: str) -> None:
        """Identify potential buildings in a new channel, excluding former candidates from
        search based on their group ID. ClusterID needs to be reset to avoid unwanted merge
        of information from previous VuildingValidation clustering.

        Args:
            in_f (str): input LAS
            out_f (str): output LAS
        """
        pipeline = pdal.Pipeline()
        pipeline |= pdal.Reader(in_f, type="readers.las")
        non_candidates = (
            f"({self.data_format.las_dimensions.candidate_buildings_flag} == 0)"
        )
        p_heq_threshold = f"(building>={self.min_building_proba})"
        A = f"(building>={self.min_building_proba * self.min_frac_confirmation_factor_if_bd_uni_overlay})"
        B = f"({self.data_format.las_dimensions.uni_db_overlay} > 0)"
        p_heq_modified_threshold_under_bd_uni = f"({A} && {B})"
        where = f"{non_candidates} && ({p_heq_threshold} || {p_heq_modified_threshold_under_bd_uni})"
        pipeline |= pdal.Filter.cluster(
            min_points=self.cluster.min_points,
            tolerance=self.cluster.tolerance,
            is3d=self.cluster.is3d,
            where=where,
        )
        # Always move and reset ClusterID to avoid conflict with later tasks.
        pipeline |= pdal.Filter.ferry(
            dimensions=f"{self.data_format.las_dimensions.cluster_id}=>{self.data_format.las_dimensions.ai_building_identified}"
        )
        pipeline |= pdal.Filter.assign(
            value=f"{self.data_format.las_dimensions.cluster_id} = 0"
        )

        pipeline |= pdal.Writer(
            type="writers.las",
            filename=out_f,
            forward="all",
            extra_dims="all",
            minor_version=4,
            dataformat_id=8,
        )
        os.makedirs(osp.dirname(out_f), exist_ok=True)
        pipeline.execute()
