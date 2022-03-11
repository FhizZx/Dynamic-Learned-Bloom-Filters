[![Screenshot3](Dynamic_Learned_Bloom_Filters-1.jpg "Screenshot3")]()

cache - pytorch cached data such as model weights or
        hessians

code_archive - deprecated code but there might be some useful bits
               in there; most of the it is meant for 1D synthetic
               experiments but can probably be adapted
               for multidimensional data without much effort

data - datasets used for training

images - contains most of the plots for the 1D synthetic
         experiments

//
Useful source code:

utils - code for computing gradients/hessians/fishers for models,
        also some code for model update tuning but probably needs
        to be slightly rewritten

binary_dataset - a custom pytorch dataset that should hopefully
                 be able to serve most purposes for benchmarking

bloom_filters - contains the code for a simple online learned BF

online_learned_models - gives a template for what methods
                        a learned model used for the OnlineLBF should have, also has a sample
                        online learned model with (naive) inverse
                        fisher gradient product updates

urldata_example - example of using the supplied code
                  for an experiment on the URL dataset,
                  comparing subsequent IHGP updates with
                  retraining from scratch
