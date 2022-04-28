# Developer's guide

## Use application from source

Instead of installing the package, you can run it from python sources directly
```bash
# activate an env matching ./bash/setup_env.sh requirements.
conda activate lidar_prod
python lidar_prod/run.py paths.src_las=[/path/to/file.las]
```

## CICD and versions

New features are staged in the `dev` branch, and CICD workflow is run when a pull requets to merge is created.
In Actions, check the output of a full evaluation on a single LAS to spot potential regression. The app is also run 
on a subset of a LAS, which can be visually inspected before merging - there can always be surprises.

Package version follows semantic versionning conventions and is defined in `setup.py`. 

Releases are generated when new high-level functionnality are implemented (e.g. a new step in the production process), with a documentation role. Production-ready code is fast-forwarded in the `prod` branch when needed. 