## Data Format

This directory should have a subdirectory for every target (model, dataset), with the following structure:

```
<Model>_<Dataset>
|   Losses
|   |   <split1>
|   |   |   |   loss_<underscore-separated tags>.txt
|   |   |   |   ...
|   |   <split2>
|   |   ...
|   Penultimate_Tensors
|   |   <split1>
|   |   |   vectors_<underscore-separated tags>.txt
|   |   |   ...
|   |   ...
|   <n-k>_Tensors
|   |   Same structure as Penultimate_Tensors, but stores vectors from previous layers
|   Predictions
|   |   <split1>
|   |   |   predictions_<underscore-separated tags>.txt
|   |   |   ...
|   |   ...
|   <Model>_<underscore-separated tags>.pth (model weights)
```

Tags typically store epoch information (`e{i}`, `best`, or `last`) etc.

Each of the files is plaintext, with each line containing the relevant data for one datapoint, space-separated.
For example, each line of a loss file contains a single value, whereas each line of a vectors file contains as many values as the layer's embedding dimension.
As such, there are exactly as many lines as there are datapoints in the split.

A collated `preds.csv` file may also be present, with the columns:
`Epoch_No,Split,Image_Index,Image_Function_value,Original_Label,Predicted_Label,Correct`

Where `Correct` is either `TRUE` or `FALSE`.

### Replicability
- Every (model, dataset) pairs should correspond to a training run in `training/`, and every dataset (download URL) + split should be documented in `datasets/`
- The datasets should be shuffled with a fixed seed, which should also be documented in `datasets/`
- `training/` should also document mappings from best and last to epoch numbers for each run