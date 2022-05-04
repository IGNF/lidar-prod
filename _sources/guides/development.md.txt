# Developer's guide

## Code versionning

Package version follows semantic versionning conventions and is defined in `setup.py`. 

Releases are generated when new high-level functionnality are implemented (e.g. a new step in the production process), with a documentation role. Production-ready code is fast-forwarded in the `prod` branch when needed.

## Tests

Tests can be run in an activated environment with.

```bash
conda activate lidar_prod
python -m pytest
```

One test depends on a large, non-versionned file (665MB), which is accessible from the self-hosted action runner, but not publicly available at the moment. The absence of the file makes the test xfail so that it is not required for local development.

# Continuous Integration (CI)

New features are developped in ad-hoc branches, and merged in the `dev` branch, where they may be accumulated until stability. 
CI tests are run for pull request to merge on either `dev` or `main` branches, and only for branches that are flagged as "ready for review" in github.

# Continuous Delivery (CD)



Additionally, in the CICD workflow, the app is run on two point clouds (a subset and the large, non-versionned file mentionned in [Tests](#Tests)). Outputs are saved to the self-hosted action runner, and can be inspected to get a qualitative sense of the performance.

