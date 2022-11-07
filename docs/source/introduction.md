Lidar-Prod is a production library which aims at augmenting rule- based semantic segmentation algorithms with deep learning neural network predictions (probabilities) and a public building vector database (BDUni). Its main entry-points are:

- `application.py`: The application takes a point cloud and update its Classification dimension based on its deep learning predictions and a public geographic database.
- `optimization.py`: The right balance between automation of decision and error is found via a multi-objective optimization of of the decision thresholds, by means of a simple genetic algorithm.

Our strategy is to fuse together different sources of informations (rule- based classification, deep learning predictions, databases), so that we can ensure a high-quality classification while minimizing the need for human correction. Deep learning probabilities might also be used to highlight area of uncertainty, or to spot elements that were missed by the other approaches.

Right now, the class `building` is the only one that is addressed. The extension to other classes is dependent on the training of multiclass AI model, which requires high quality training datasets that are currently being produced.