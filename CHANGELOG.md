# main

### 1.9.14
- Be robust to pgsql2shp warnings when dealing with empty tables (i;e. no buildings).

### 1.9.13
- Run CICD operations for all branches prefixed with "staging-"

### 1.9.12
- Instantiate bd_uni_connection_params externally to fix BV optimization.

### 1.9.11
- Hide credentials to the BD Uni.

### 1.9.10
- Add missing "reservoir" category to the BD uni request, alongside buildings.

### 1.9.9
- Update PDAL version to V2.5.1.

### 1.9.8
- Keep confidence channel in the "reliability" LAS channel.

### 1.9.7
- Update PDAL version to V2.5.0.

### 1.9.6
- Deal with cases where there are no building over the cloud, by creating an empty shapefile.
