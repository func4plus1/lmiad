## References

(2020) [The Well Archicted Framework - Machine Learning Lens](https://d1.awsstatic.com/whitepapers/architecture/wellarchitected-Machine-Learning-Lens.pdf)

(2019) [The Scientific Method in the Science of Machine Learning](https://arxiv.org/pdf/1904.10922.pdf)

(2011)[Philosophy and the practice of Bayesian statistics](http://www.stat.columbia.edu/~gelman/research/unpublished/philosophy.pdf)

(1998) [A Beginners Guide to the Mathematics of Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.3556&rep=rep1&type=pdf&fbclid=IwAR1jmZJ3FBqcNLKbkyKXV3nZ3LaT12M6RoE4gW70TUrLrp-7WJe5LvjQkn4)

## Notes 

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
