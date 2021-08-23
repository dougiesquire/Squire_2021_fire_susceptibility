Code respository for:
=====================

    Likelihood of unprecedented drought and fire weather during Australia’s 2019/2020 megafires
    by Dougal T. Squire, Doug Richardson, James S. Risbey, Amanda S. Black, Vassili Kitsios, Richard J. Matear, Didier Monselesan, Thomas S. Moore and Carly R. Tozer    

Notebooks are numbered according to the order in which they should be run.

Figures are generated in the following notebooks:
 - **Fig 1**: `0_get_region.ipynb`
 - **Fig 2**: `3_historical_precedent.ipynb`
 - **Fig 3**: `4_fidelity_testing.ipynb`
 - **Fig 4**: `4_fidelity_testing.ipynb`
 - **Fig 5**: `5_likelihoods.ipynb`
 - **Fig 6**: `6_conditioning.ipynb`
 - **Fig 7**: `6_conditioning.ipynb`
 - **Fig 8**: `5_likelihoods.ipynb`

To clean up before publishing:
 - remove figure checks from `1.0_prepare_index_data.ipynb` comparing old and new data
 - keep only PBSCluster code cells
 - decide whether including masking of dt or not and edit code accordingly
