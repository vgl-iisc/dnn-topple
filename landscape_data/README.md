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
|   Tensors
|   |   <split1>
|   |   |   vectors_<underscore-separated tags>.txt
|   |   |   ...
|   |   ...
|   Predictions
|   |   <split1>
|   |   |   predictions_<underscore-separated tags>.txt
|   |   |   ...
|   |   ...
|   <Model>_<underscore-separated tags>.pth (model weights)
```

Tags: `a{layer_index_from_back}_e{epoch_no}`. For instance, if the activations are coming from the penultimate layer, `a{1}`, if they're from the layer before that `a{2}` etc. It works like negative indexing.

Each of the files is plaintext, with each line containing the relevant data for one datapoint, space-separated.
For example, each line of a loss file contains a single value, whereas each line of a vectors file contains as many values as the layer's embedding dimension.
As such, there are exactly as many lines as there are datapoints in the split.

A collated `preds.csv` file may also be present, with the columns:
`Epoch_No,Layer,Split,Image_Index,Image_Function_value,Original_Label,Predicted_Label,Correct`

Where `Correct` is either `TRUE` or `FALSE`.

### Replicability
- Every (model, dataset) pairs should correspond to a training run in `training/`, and every dataset (download URL) + split should be documented in `datasets/`
- The datasets should be shuffled with a fixed seed, which should also be documented in `datasets/`
- `training/` should also document mappings from best and last to epoch numbers for each run