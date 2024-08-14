# AggCVAE_thesis_github
Deep learning and MCMC inference for mapping HIV Prevalence in Zambia 2018. The code in this repository builds on the work from ES: https://github.com/MLGlobalHealth/aggVAE.

This repository is organised as follows:

- `aggGP.ipynb`: code for the AggGP model.
- `aggVAE_indiv.ipynb`:  code for the AggVAE model applied to administrative units 1 and 2 separately.
- `aggCVAE_gpu_indiv_admin.ipynb`: code for the AggCVAE model applied to administrative units 1 and 2 separately.
- `aggCVAE_all_admin.ipynb`:  code for the AggCVAE model encoding administrative units 1 and 2 jointly.
- `aggCVAE_thesis_figures.ipynb`: code to produce comparison plots between aggVAE, aggCVAE and aggGP in the thesis.

The code relies heavily on the `numpyro` probabilistic programming library for Bayesian hierarchical modelling and MCMC inference, as well as `JAX` for deep learning. 
It is recommended that the reader familiarises themselves with the corresponding documentation before using the code:
- `numpyro`: https://num.pyro.ai/en/latest/index.html#introductory-tutorials
- `JAX`: https://jax.readthedocs.io/en/latest/

For the models, each notebook follows a similar structure:

1. Import libraries
2. Set the administrative level (if applicable)
3. Read shapefiles of boundaries of the administrative units
4. Read HIV prevalence data
5. Merge the shape and prevalence data
6. Create the computational grid
   - Create a regular grid
   - Add points to administrative units with insufficient gridpoints 
8. Train a VAE/CVAE algorithm with `JAX` (if applicable) 
   - Define preliminary functions (covariance matrix of GP, indicator matrix M)
   - Define the model architecture (model configurations, MLP layers, etc.)
   - Define the aggregated GP prior to be approximated by the VAE/CVAE with `numpyro`
   - Define functions needed for training (dataloader, training step, validation step, loss function)
   - Train the VAE/CVAE
   - Save the decoder and losses
   - Evaluate how well the VAE/CVAE has learnt from the aggregated GP prior, i.e. comparing aggregated GP prior samples before training and VAE/CVAE approximations after training (spatial maps and 95% BCI intervals)
9. Define the HIV prevalence model
10. Fit the prevalence model using MCMC inference with `numpyro`
    - Save the MCMC iterations
    - Create a table of MCMC summary results with Elapsed Time, Average ESS and R-hat statistics
    - Plot posterior distributions and traceplots of hyperparameters (if applicable)
11. Analyse model predictions
    - Map HIV prevalence spatially
    - Scatterplot of estimated prevalence against observed prevalence
    - Calculate 95 % BCI intervals of the posterior distribution of the estimated HIV prevalence
    - Calculate performance metrics (MAE, RMSE, Pearson's and Spearman's correlation coefficients between observations and predictions)


The VAE and CVAE algorithms were trained using a GPU. To run the code on CPU, comment out the line `jax.config.update('jax_platform_name', 'gpu')` at the start of each notebook in the Importing Libraries section.

Alternatively, the trained decoders, MCMC iterations and saved losses provided in the `decoders`, `MCMC` and `losses` folders can be used directly to reproduce the findings in the thesis. To do this, the VAE/CVAE training step, MCMC inference and savings of decoder and MCMC objects should be commented out. This is indicated more precisely in the code. Please note that the MCMC files for both AggGP models presented in the thesis were not included in the Github repository due to file size limitations - instead they were submitted as part of the .zip submission file.



   


 
