from lidar_prod.tasks.basic_identification_optimization import BasicIdentifierOptimizer

LAS_SUBSET_FILE_VEGETATION = "tests/files/436000_6478000.subset.postIA.las"


def test_basic_identifier_optimizer(vegetation_unclassifed_hydra_cfg):
    basic_identifier_optimizer = BasicIdentifierOptimizer(
        vegetation_unclassifed_hydra_cfg,
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"][
            "ai_vegetation_proba"
        ],
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"][
            "ai_vegetation_unclassified_groups"
        ],
        vegetation_unclassifed_hydra_cfg["data_format"]["codes"]["vegetation"],
        vegetation_unclassifed_hydra_cfg["data_format"]["las_dimensions"][
            "classification"
        ],
        vegetation_unclassifed_hydra_cfg["basic_identification"][
            "vegetation_nb_trials"
        ],
        list(
            vegetation_unclassifed_hydra_cfg["data_format"]["codes"][
                "vegetation_target"
            ].values()
        ),
    )
    basic_identifier_optimizer.optimize()
    trial = basic_identifier_optimizer.study.best_trial
    assert trial.value > 0.9  # IoU value
