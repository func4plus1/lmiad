## References

(2020) [The Well Archicted Framework - Machine Learning Lens](https://d1.awsstatic.com/whitepapers/architecture/wellarchitected-Machine-Learning-Lens.pdf)

(2019) [The Scientific Method in the Science of Machine Learning](https://arxiv.org/pdf/1904.10922.pdf)

(2011)[Philosophy and the practice of Bayesian statistics](http://www.stat.columbia.edu/~gelman/research/unpublished/philosophy.pdf)

(1998) [A Beginners Guide to the Mathematics of Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.3556&rep=rep1&type=pdf&fbclid=IwAR1jmZJ3FBqcNLKbkyKXV3nZ3LaT12M6RoE4gW70TUrLrp-7WJe5LvjQkn4)

## Notes 

### [The Well Archicted Framework - Machine Learning Lens](https://d1.awsstatic.com/whitepapers/architecture/wellarchitected-Machine-Learning-Lens.pdf)

The Machine Learning Jouney often has three major outposts:

* AI Services
  *  Fully managed services that enable you to quickly add ML capabilities to your workloads using API calls.
  * Services at this level are based on pre-trained or automatically trained machine learning and deep learning models, so that you don’t need ML knowledge to use them.
* ML Services
  * Managed services and resources for machine learning to developers, data scientists, and researchers.
  * Label data, build, train, deploy, and operate custom ML models without having to worry about the underlying infrastructure needs.
  *  heavy lifting of infrastructure management is managed by the cloud vendor, so that your data science teams can focus on what they do best.
* ML Frameworks and Infrastructure 
  * Design your own tools and workflows to build, train, tune, and deploy models, working at the framework and infrastructure level. 
  
Questions to ask:

MLOPS 01: How have you prepared your team to operate and support a machine learning workload? 
MLOPS 02: How are you documenting model creation activities? 
MLOPS 03: How are you tracking model lineage?
MLOPS 04: How have you automated the development and deployment pipeline for your ML workload?
MLOPS 05: How are you monitoring and logging model hosting activities?
MLOPS 06: How do you know when to retrain ML models with new or updated data?
MLOPS 07: How do you incorporate learnings between iterations of model development, model training, and model hosting?

MLSEC 01: How do you control access to your ML workload?
MLSEC 02 : How are you protecting and monitoring access to sensitive data used in your ML workloads?
MLSEC 03: How are you protecting trained ML models?

MLREL 01: How do you manage changes to your machine learning models and prediction endpoints?
MLREL 02: How are changes to ML models coordinated across your workload?
MLREL 03: How are you scaling endpoints hosting models for predictions? 
MLREL 04: How do you recover from failure or inadvertent loss of a trained ML model? 
MLREL 05: How do you recover from failure or inadvertent loss of model hosting resources? 

MLPER 01: How are you choosing the most appropriate instance type for training and hosting your models?
MLPER 02: How do you scale your ML workload while maintaining optimal performance?

MLCOST 01: How do you optimize data labeling costs?
MLCOST 02: How do you optimize costs during ML experimentation?
MLCOST 03: How do you select the most cost optimal resources for ML training?
MLCOST 04: How do you optimize cost for ML Inference?

### [The Scientific Method in the Science of Machine Learning](https://arxiv.org/pdf/1904.10922.pdf)

> We conjecture that grounding ML research in statistically sound hypothesis testing with careful control of nuisance parameters may encourage the publication of advances that
stand the test of time.

> Proper application of the scientific method can help researchers understand factors of variation in experimental outcomes, as well as the dynamics of
components in ML models, which would aid in ensuring robust performance in real world systems.

> Starting from the assumption that there exists accessible ground truth, the scientific method is a systematic framework for experimentation that allows researchers to make objective statements about
phenomena and gain knowledge of the fundamental workings of a system under investigation. 

> Central to this framework is the formulation of a scientific hypothesis and an expectation that can be falsified through experiments and statistical methods... failure to include these steps is likely to lead to unscientific findings.

> “If the hypothesis is right, then I should expect to observe...”

> At the base of scientific research lies the notion that an experimental outcome is a random variable, and that appropriate statistical machinery must be employed to estimate the properties of its
distribution... Since abundant sampling of observations might be prohibitive due to resource constraints, the role of statistical uncertainties accompanying the measurement becomes vital to interpret the result.

### Philosophy and the Practice of Bayesian Statistics

> Bayesian statistics or “inverse probability”—starting with a prior distribution, getting data, and moving to the posterior distribution—is associated with an inductive approach of learning about the general from particulars.

> Rather than testing and attempted falsification, learning proceeds more smoothly: an accretion of evidence is summarized by a posterior distribution, and scientific process is associated with the rise and fall in the posterior probabilities of various models.

> We think most of this received view of Bayesian inference is wrong. Bayesian methods are no more inductive than any other mode of statistical inference.

> The goal of model checking, then, is not to demonstrate the foregone conclusion of falsity as such, but rather to learn how, in particular, this model fails.

> When we find such particular failures, they tell us how the model must be improved; when severe tests cannot find them, the inferences we draw about those aspects of the real world from our fitted model become more credible.

> In designing a good test for model checking, we are interested in finding particular errors which, if present, would mess up particular inferences,
and devise a test statistic which is sensitive to this sort of mis-specification.

> What we are advocating, then, is what Cox and Hinkley (1974) call “pure significance testing,” in which certain of the model’s implications are compared directly to the data, rather than entering into a contest with some alternative model.

> A model is a story of how the data could have been generated; the fitted model should therefore be able to generate synthetic data that look like the real data; failures to do so in important ways indicate faults in the model.

> There are technical problems with methods that purport to determine the posterior probability of models, most notably that in models with continuous parameters, aspects of the model that have essentially no effect on posterior inferences within a model can have huge effects on the comparison of posterior probability among models.

> Complex models can and should be checked and falsified.
