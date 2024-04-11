# Overview of the process

A las file goes through several steps, which alter its content.
At first we have a `Raw` las file, sent to another process (`Myria3D`) that infers the probabilities of various classes.
From there, the las file is used a first time by this module to decide if the points are vegetation, unclassified or something else, and sent to an external process, a `rule-based segmentation`.
The las file is used a second time by this module to decide if the points are buildings or not.

## schema of the overall process
The arrows represent dimensions in the las file, where they come from and where they are consumed.
```{mermaid}
	sequenceDiagram
		participant Raw
		participant AI inference (Myria3D)
		participant vegetation and unclassified detection
		participant Rule based segmentation
		participant Building module
		actor Human inspection
		Raw->>AI inference (Myria3D): Intensity
		Raw->>AI inference (Myria3D): ReturnNumber
		Raw->>AI inference (Myria3D): NumberOfReturns
		Raw->>AI inference (Myria3D): Red, Green, Blue
		Raw->>AI inference (Myria3D): Infrared
		AI inference (Myria3D)->>vegetation and unclassified detection: vegetation
		AI inference (Myria3D)->>vegetation and unclassified detection: unclassified
		vegetation and unclassified detection->>Rule based segmentation: Group (1: vegetation, 3: unclassified)
		note right of Rule based segmentation: uses vegetation and unclassified
		AI inference (Myria3D)->>Building module: building
		AI inference (Myria3D)->>Building module: entropy
		Building module->>Human inspection: Group
		Raw-->>vegetation and unclassified detection: Classification
		Raw-->>Building module: Classification
		note right of Building module: Building module updates Classification
		Building module->>Human inspection: Classification
```
