# Random Forest prediction of <sup>27</sup>Al ss-NMR C<sub>Q</sub> for crystalline materials 

--------------------------------------------------------------------------------------------

## Introduction

The goal of this model is to predict the CQ value from electronic field gradient (EFG) tensor of <sup>27</sup>Al containing solid materials. The algorithm chosen here is random forest and the model is trained on a dataset of totally around 1800 structures.  

The structures are transformed into two sets of features for the sub-stream of model training, the structural features and the elemental properties features. It is showed that the model based on our features managed to predict the CQ value for the test set with R<sup>2</sup>=0.97 and RMSE = 0.7 MHz which beats the model based on SOAP (R<sup>2</sup>=0.92).  

Here are the prediction results for this model and SOAP model.  

![test_result](./figures/test_r2_98_chemenv_split.png)

**Figure 1** Random forest prediction of CQ with the features from this work.  

![test_result_SOAP](./figures/27Al_RF_testset_SOAP_101521.png)

**Figure 2** Random forest prediction of CQ with SOAP.  
