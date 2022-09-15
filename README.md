# SasakiMLSaliency

#DESCRIPTION
This repository acompanies the paper 'Topological Upper Bound on Critical Volume of Toric Sasaki-Einstein Manifolds'. The problem considered there is to determine the critical (minimised w.r.t. associated Reeb vector) volumes of Sasaki-Eintein manifolds arising in Calabi-Yau cones constructed from the 3d reflexive polytopes.  

Experiment 1 - the critical volumes are learned from various quantities associated to the geometric objects involved, including the properties of the toric diagrams and topology.

Experiment 2 - the number of non-zero entries in the critical Reeb vector ('degree of Reeb action') is learned from similar set of features

It contains the code for classification and regression neural networks and gradient saliency analysis, largely adapted from https://github.com/edhirst/PolytopeML and https://github.com/deepmind/mathematics_conjectures. There are two experiments as described in the main paper that can be viewed. 
Note: the results and plots will not be exactly reproduced as in the paper but should be nevertheless similar.

#DEPENDENCIES
The code is mostly Keras/Tensorflow-based, with some minor tools taken from SciKitLearn, as well as other standard Python libraries.
All the datasets used are contained in the associated pickle files. 

#TO RUN
Download MLP_Saliency_GIT.py together with the X_dict and Y_dict pickle files, containing input and output data resp.

Comment/uncomment the demo function calls in the line 520 of MLP_Saliency_GIT.py to run the corresponding experiment.
