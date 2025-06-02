### Code Structure and Usage (Density-Based Approach)

The code is organized into Jupyter notebooks for clarity and ease of use.

#### Notebooks Overview:

* **`train_paper_unet.ipynb`**:

  * Contains the training process and results for the density-based approach.
  * Before running this notebook, label files must be created. Detailed instructions are provided in the README file located in the dataset directory.

#### Pretrained Model:

* The file `dens_paper_pretrained.keras` is a modified version of the model presented in the following paper:

  *Kraft, M.; Aszkowski, P.; Pieczyński, D.; Fularz, M. Low-Cost Thermal Camera-Based Counting Occupancy Meter Facilitating Energy Saving in Smart Buildings. Energies 2021, 14, 4542.*
  [https://doi.org/10.3390/en14154542](https://doi.org/10.3390/en14154542)

* The modification made is to support variable input sizes instead of a fixed input. The original weights trained by the authors were loaded into this model.



