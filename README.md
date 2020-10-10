## References

(2020) [The Well Archicted Framework - Machine Learning Lens](https://d1.awsstatic.com/whitepapers/architecture/wellarchitected-Machine-Learning-Lens.pdf)

(2020) [Language Models Are Few Shot Learners](https://arxiv.org/pdf/2005.14165.pdf) | [video explanation](https://www.youtube.com/watchv=SY5PvZrJhLE&feature=youtu.be&ab_channel=YannicKilcher)

(2019) [The Scientific Method in the Science of Machine Learning](https://arxiv.org/pdf/1904.10922.pdf)

(2019) [Federated Machine Learning: Concepts and Applications](https://arxiv.org/pdf/1902.04885.pdf)

(2019) [Federated Machine Learning Concept and Application)](https://arxiv.org/pdf/1902.04885.pdf)

(2019) [Patient Clustering Improves Efficiency of Federated Machine Learning to predict mortality and hospital stay time using distributed Electronic Medical
Records](https://arxiv.org/ftp/arxiv/papers/1903/1903.09296.pdf)

(2019) [Towards Federated Learning At Scale: System Design](https://arxiv.org/pdf/1902.01046.pdf)

(2019) [Federated learning for mobile keyboard prediction](https://arxiv.org/pdf/1811.03604.pdf)

(2019) [Managing Machine Learning Projects - Balance Potential With the Need for Guardrails](https://d1.awsstatic.com/whitepapers/aws-managing-ml-projects.pdf)

(2018) [Optimizing Revenue Over Data Driven Assortments](https://arxiv.org/pdf/1708.05510.pdf)

(2018) [Leaf: A Federated Learning Benchmark](https://arxiv.org/pdf/1812.01097.pdf)

(2018) [Applied Federated Learning: Improving Google Keyboard Query Suggestions](https://arxiv.org/pdf/1812.02903.pdf)

(2017) [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 

(2016) [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/pdf/1610.02527.pdf)

(2016) [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629.pdf)

(2012) [Approximate Nearest Neighbor: Towards Reducing the Curse of Dimensionality](https://www.theoryofcomputing.org/articles/v008a014/v008a014.pdf)

(2011) [Philosophy and the practice of Bayesian statistics](http://www.stat.columbia.edu/~gelman/research/unpublished/philosophy.pdf)

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

**Operation Excellence** For Machine Learning:
  * Establish cross functional teams
  * Identify the end-to-end architecture and operational model early
  * Continuously monitor and measure ML workloads
  * Establish a model retraining strategy
  * Document machine learning discovery activities and findings
  * Version machine learning inputs and artifacts
  * Automate machine learning deployment pipelines
 
**Security** for Machine Leanring:
  * Restrict Access to ML systems
  * Ensure Data Governance
  * Enforce Data Lineage
  * Enforce Regulatory Compliance
  
**Reliability** for Machine Learning: 
  *  Manage changes to model inputs through automation
  * Train once and deploy across environments

**Performance Efficiency** for Machine Learning:
  * Optimize compute for your ML workload
  * Define latency and network bandwidth performance requirements for your models
  * Continuously monitor and measure system performance

**Cost Optimization** for Machine Learning:
  * Automate to reduce cost of ownership
  * Experiment with small datasets
  * Right size training and model hosting instances
  * Account for inference architecture based on consumption patterns
  * Define overall ROI and opportunity cost
  
  
Questions to ask:

* MLOPS 01: How have you prepared your team to operate and support a machine learning workload? 
* MLOPS 02: How are you documenting model creation activities? 
* MLOPS 03: How are you tracking model lineage?
* MLOPS 04: How have you automated the development and deployment pipeline for your ML workload?
* MLOPS 05: How are you monitoring and logging model hosting activities?
* MLOPS 06: How do you know when to retrain ML models with new or updated data?
* MLOPS 07: How do you incorporate learnings between iterations of model development, model training, and model hosting?
<br>

* MLSEC 01: How do you control access to your ML workload?
* MLSEC 02 : How are you protecting and monitoring access to sensitive data used in your ML workloads?
* MLSEC 03: How are you protecting trained ML models?
<br>

* MLREL 01: How do you manage changes to your machine learning models and prediction endpoints?
* MLREL 02: How are changes to ML models coordinated across your workload?
* MLREL 03: How are you scaling endpoints hosting models for predictions? 
* MLREL 04: How do you recover from failure or inadvertent loss of a trained ML model? 
* MLREL 05: How do you recover from failure or inadvertent loss of model hosting resources? 
<br>

* MLPER 01: How are you choosing the most appropriate instance type for training and hosting your models?
* MLPER 02: How do you scale your ML workload while maintaining optimal performance?
<br>

* MLCOST 01: How do you optimize data labeling costs?
* MLCOST 02: How do you optimize costs during ML experimentation?
* MLCOST 03: How do you select the most cost optimal resources for ML training?
* MLCOST 04: How do you optimize cost for ML Inference?

The end-to-end machine learning process includes the following phases:
* Business Goal Identification
* ML Problem Framing
* Data Collection and Integration
* Data Preparation
* Data Visualization and Analytics
* Feature Engineering
* Model Training
* Model Evaluation
* Business Evaluation
* Production Deployment (Model Deployment and Model Inference)

![End To End ML Pipeline According to Amazon](end2endml.png)

![Blue Green Deployment](bluegreendeployment.png)

<br>

**Canary Deployment**

![Canary Deployment](canarydeployment.png)


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

### (2011) [Philosophy and the practice of Bayesian statistics](http://www.stat.columbia.edu/~gelman/research/unpublished/philosophy.pdf)

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

---

# (2019) [Federated Machine Learning Concept and Application)](https://arxiv.org/pdf/1902.04885.pdf)

**Secure Multi-party Computation**

provide security proof in a well-defined simulation framework to guarantee complete zeroknowledge, that is, each party knows nothing except its input and output.

**Differential Privacy**

Adding noise to the data, or using generalization methods to obscurecertain sensitive attributes until the third party cannot distinguish the individual, thereby makingthe data impossible to be restore to protect user privacy

**Homomorphic Encryption**

Homomorphic Encryption adopted to protect user data privacy through parameter exchange under the encryption mechanism during machine learning. Unlike differential privacy protection, the data and the model itself are not transmitted, nor can they be guessed by the other party’s data.

**Horizontal federated learning** or sample-based federated learning, 

introduced in the scenarios that data sets share the same feature space but different samples

A horizontal federated learning system typically assumes honest participantsand security against a honest-but-curious server

example two banks in different regions 

k participants with the same data structure collaborativelylearn a machine learning model with the help of a parameter or cloud server

* Step 1: participants locally compute training gradients, mask a selection of gradients withencryption  differential privacy  or secret sharing techniques, and send maskedresults to server
* Step 2: Server performs secure aggregation without learning information about any participant;
* Step 3: Server send back the aggregated results to participants
* Step 4: Participants update their respective model with the decrypted gradients

**Vertical federated learning** or feature-based federated learning

 Is applicable to the cases that two data sets share the same sample ID space but differ in feature space. 
 
 example a bank and an ecommerce company in the same city have the same sample probably
 
 At the end of learning, each party only holds the model parameters associated to its own features, therefore at inference time, the two parties also need to collaborate to generate output.
 
 Part 1.Encrypted entity alignment. 
 
 Part 2. Encrypted model training: 
 
 * Step 1: collaborator C creates encryption pairs, send public key to A and B
 * Step 2: A and B encrypt and exchange the intermediate results for gradient and loss calculations
 * Step 3: A and B computes encrypted gradients and adds additional mask, respectively, and B also computes encrypted loss; A and B send encrypted values to C
 * Step 4: C decrypts and send the decrypted gradients and loss back to A and B; A and B unmask the gradients, update the model parameters accordingly.
 
**Federated Transfer Learning** applies to the scenarios that the two data sets differ not only in samples but also in feature space

a common representation between the two feature space is learned using the limited common sample sets andl ater applied to obtain predictions for samples with only one-side features.

## Important Code

Annoy [Repository](https://github.com/spotify/annoy)

### Concepts

Cross Validation 
* [Video Explanation](https://www.youtube.com/embed/TIgfjmp-4BA)
* K-Fold (Leave One Out) [Video Explained](https://www.youtube.com/embed/6dbrR-WymjI)
