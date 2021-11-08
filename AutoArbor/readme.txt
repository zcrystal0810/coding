AutoArbor is a python based package used to separate a complete tree structure, mainly used for splitting a neuron reconstruction result, into several dense clusters. 
In order to analyze with more biological meanings, some postprocessing steps have added into the original script. You can either use the original one or the revised one.
Through the post-processing, we have done the following things:
1. Soma arbor part contains both soma and dendrite
2. Long passing fiber is cut/removed
3. Group small clusters, e.g some with only one branch

The command of using the package is as below:
python <AutoArbor script> --filename <swcfile>
