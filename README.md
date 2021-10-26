# Machine learning prediction of <sup>27</sup>Al ss-NMR C<sub>Q</sub> for crystalline materials 

--------------------------------------------------------------------------------------------

## Introduction

In the field of solid-state Nuclear Magnetic Resonance (NMR), materials are measured to yield critical parameters which can be used to detect the local geometry of the subject structure. A popular example is the isotropic chemical shift which is widely used to determine the structural difference between difference chemical sites for both solid-state and liquid-state materials. For solid materials specifically, there are more parameters than isotropic chemical shift that people can get from the NMR spectrums because of the remaining many body interactions such as dipolar interactions and quadrupolar interactions. 

Experimentally quadrupolar interactions can be measure in terms of a value called the quadrupolar coupling constant (C<sub>Q</sub>). C<sub>Q</sub> is a value derived from the electronic field gradient (EFG) tensor and is directly correlated to the broadening of the spectrum.

![spectrum_cq](./figures/spectrum_cq.png)

**Figure 1** NMR spectrum with difference value of C<sub>Q</sub>

The goal of this model is to predict the C<sub>Q</sub> value from electronic field gradient (EFG) tensor of <sup>27</sup>Al containing solid materials. 

## Dataset

The dataset for the training was obtained from the Materials Project[1]. The dataset is consist of 1800 <sup>27</sup>Al containing structures of 4, 5 and 6 coordination. Most Al materials have a local geometry of tetrahedron (T:4), trigonal-bipyramidical (T:5) or octahedral (O:6), thus these are the geometries considered in this dataset. The distribution of geometries is showed below:

![geo_dist](./figures/geo_dist.png)

**Figure 2** Geometry distribution of the <sup>27</sup>Al dataset. 

All the structures in the dataset was accompanied with density functional theory (DFT) calculated NMR parameters for all sites, the calculations were perform with VASP[2] in a high-throughput manner. 

## Feature generation 

The structures are transformed into two sets of features for the sub-stream of model training, the structural features and the elemental properties features. *More details will be added later.* 

## Result

To assess the performance of the features and the overall accuracy of the model, we also included a benchmarking model based on the popular feature Smooth Overlap of Atomic Positions (SOAP)[3]. 

It is showed that the model based on our features managed to predict the CQ value for the test set with R<sup>2</sup>=0.97 and RMSE = 0.7 MHz which is a better result compared with the model based on SOAP (R<sup>2</sup>=0.92).  

Here are the prediction results for this model and SOAP model on the test set.  

![test_result](./figures/test_r2_98_chemenv_split.png)

**Figure 3** Random forest prediction of CQ with the features from this work.  

![test_result_SOAP](./figures/27Al_RF_testset_SOAP_101521.png)

**Figure 4** Random forest prediction of CQ with SOAP. 

### References

* [1]: https://materialsproject.org/

* [2]: https://www.vasp.at/

* [3]: https://singroup.github.io/dscribe/0.3.x/tutorials/soap.html