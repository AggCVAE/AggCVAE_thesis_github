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

1. Importing libraries
2. Reading shapefiles of boundaries of the administrative units
3. Reading HIV prevalence data
4. Merging the shape and prevalence data
5. Training a VAE/CVAE algorithm with `JAX` (if applicable) 
   a. Defining preliminary functions (covariance matrix of GP, indicator matrix M)
   b. Defining the model architecture (model configurations, MLP layers, etc.)
   c. Defining the aggregated GP prior to be approximated by the VAE/CVAE using `numpyro`
   d. Defining functions needed for training (dataloader, training step, validation step, loss function)
   e. Training the VAE/CVAE
   f. Saving the decoder and losses
   g. Evaluating how well the VAE/CVAE has learnt from the aggregated GP prior, i.e. comparing aggregated GP prior samples before training and VAE/CVAE approximations after training (spatial maps and 95% BCI intervals)
6. Defining the HIV prevalence model
7. Fitting the prevalence model using MCMC inference with `numpyro`
   a. Saving the MCMC iterations
   b. Creating a table of MCMC summary results with Elapsed Time, Average ESS and R-hat statistics
   c. Plotting posterior distributions and traceplots of hyperparameters (if applicable)
8. Analysing model predictions
   a. Mapping HIV prevalence spatially
   b. Scatterplot of estimated prevalence against observed prevalence
   c. Calculating performance metrics (MAE, RMSE, Pearson's and Spearman's correlation coefficients between observations and predictions)

   
   
   


 
