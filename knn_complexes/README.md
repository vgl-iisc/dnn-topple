## Data Format

This directory should have a subdirectory for every target (model, dataset), with the following structure:

```
<Model>_<Dataset>
|   |   <split1>
|   |   |   |   adj_<underscore-separated tags>_<k>.txt
|   |   |   |   ...
|   |   <split2>
```

Tags typically store epoch information (`e{i}`, `best`, or `last`) etc.

Each of the files is plaintext, with each line containing the relevant data for one datapoint, space-separated.
For example, each line of a loss file contains a single value, whereas each line of a vectors file contains as many values as the layer's embedding dimension.
As such, there are exactly as many lines as there are datapoints in the split.

### Replicability
- Every (model, dataset) pairs should correspond to a training run in `training/`, and every dataset (download URL) + split should be documented in `datasets/`
- The datasets should be shuffled with a fixed seed, which should also be documented in `datasets/`
- `training/` should also document mappings from best and last to epoch numbers for each run