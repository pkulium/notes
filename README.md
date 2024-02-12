- **Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach**
    >We considered the Federated Learning (FL) problem in the heterogeneous case, and studied a
personalized variant of the classic FL formulation in which our goal is to find a proper initialization
model for the users that can be quickly adapted to the local data of each user after the training phase.
We highlighted the connections of this formulation with Model-Agnostic Meta-Learning (MAML),
and showed how the decentralized implementation of MAML, which we called Per-FedAvg, can be
used to solve the proposed personalized FL problem. We also characterized the overall complexity of
Per-FedAvg for achieving first-order optimality in nonconvex settings. 


- **Attack of the Tails: Yes, You Really Can Backdoor Federated Learning**
    > We first establish that if a model is vulnerable to inference-time attacks in the
form adversarial examples [28–32], then, under mild conditions, the same model will be vulnerable to
backdoor training-time attacks. If these backdoors are crafted properly (i.e., targeting low probability,
or edge-case samples), then they can also be hard to detect

- **Inverting Gradients - How easy is it to break privacy in federated learning?**
    > We shed light into possible avenues of attack, analyze the ability to
reconstruct the input to any fully connected layer analytically, propose a general optimization-based
attack based on cosine similarity of gradients, and discuss its effectiveness for different types of
architectures and scenarios. In contrast to previous work we show that even deep, nonsmooth networks
trained with ImageNet-sized data such as modern computer vision architectures like ResNet-152
are vulnerable to attacks - even when considering trained parameter vectors. Our experimental
results clearly indicate that privacy is not an innate property of collaborative learning algorithms like
federated learning, and that secure applications to be closely investigated on a case-by case basis for
their potential of leaking private information. Provable differential privacy possibly remains the only
way to guarantee security, even for aggregated gradients of larger batches of data points.

- **Ensemble Distillation for Robust Model Fusion in Federated Learning**
    > We propose a distillation framework for robust federated model fusion, which allows for heterogeneous client models and data, and is robust to the choices of neural architectures. We show in extensive numerical experiments on various CV/NLP datasets (CIFAR-10/100, ImageNet, AG News, SST2) and settings (heterogeneous models and/or data) that the server model can be trained much faster, requiring fewer communication rounds than any existing FL technique.

- **Federated Learning with Only Positive Labels**
    > We studied a novel learning setting, federated learning with only positive labels, and proposed an algorithm that can learn a high-quality classification model without requiring negative instance and label pairs. The idea is to impose a geometric regularization on the server side to make all class embeddings spreadout. We justified the proposed method both theoretically and empirically.



- **How To Backdoor Federated Learning**
    > Federated learning employs secure aggregation to protect confidentiality of participants’ local models and thus cannot prevent our attack by detecting anomalies in participants’ contributions to the joint model. To demonstrate that anomaly detection would not have been effective in any case, we also develop and evaluate a generic constrain-and-scale technique that incorporates the evasion of defenses into the attacker’s loss function during training.


- **FLAME: Taming Backdoors in Federated Learning**
    > We present FLAME, a resilient aggregation framework for FL that eliminates the impact of backdoor attacks while maintaining the benign performance of the aggregated model. This is achieved by three modules: DP-based noising of model updates to remove backdoor contributions, automated model clustering approach to identify and eliminate potentially poisoned model updates, and model weight clipping before aggregation to limit the impact of malicious model updates on the aggregation result. The last two modules can significantly reduce the amount of random noise required by DP noising for backdoor elimination. 

- **Demystifying Why Local Aggregation Helps: Convergence Analysis of Hierarchical SGD**
    > In this work, we frst introduce a new notion of “upward” and “downward” divergences. We then use it to conduct a novel analysis to obtain a worst-case convergence upper bound for two-level H-SGD with non-IID data, non-convex objective function, and stochastic gradient. By extending this result to the case with random grouping, we observe that this convergence upper bound of H-SGD is between the upper bounds of two single-level local SGD settings, with the number of local iterations equal to the local and global update periods in H-SGD, respectively. We refer to this as the “sandwich behavior”


- **Defending against Backdoors in Federated Learning with Robust Learning Rate**
    > To prevent backdoor attacks, we propose a lightweight defense that requires minimal change to the FL protocol. At a high level, our defense is based on carefully adjusting the aggregation server’s learning rate, per dimension and per round, based on the sign information of agents’ updates.


- **DBA: DISTRIBUTED BACKDOOR ATTACKS AGAINST FEDERATED LEARNING**
    > We propose a novel distributed backdoor attack strategy DBA on FL and show that DBA is more persistent and effective than centralized backdoor attack. Based on extensive experiments, we report a prominent phenomenon that although each adversarial party is only implanted with a local trigger pattern via DBA, their assembled pattern (i.e., global trigger) attains significantly better attack performance on the global model compared with the centralized attack


- **Blind Backdoors in Deep Learning Models**
    > We investigate a new vector for backdoor attacks: code poisoning. Machine learning pipelines include code from open-source and proprietary repositories, managed via build and integration tools. Code management platforms are known vectors for malicious code injection, enabling attackers to directly modify source and binary code

- **SparseFed: Mitigating Model Poisoning Attacks in Federated Learning with Sparsification**
    > In this work, we present SparseFed, a new optimization algorithm for federated learning that can train high-quality models under these constraints while greatly mitigating model poisoning attacks. We describe SparseFed in detail in Section 2, but the main idea is intuitive: at each round, participating devices compute an update on their local data and clip the update. The server computes the aggregate gradient, and only updates the topk highest magnitude elements.

- **Data-Free Knowledge Distillation for Heterogeneous Federated Learning**
    >In this paper, we propose an FL paradigm that enables efficient knowledge distillation to address user heterogeneity without requiring any external data. Extensive empirical experiments, guided by theoretical implications, have shown that our proposed approach can benefit federated learning with better generalization performance using less communication rounds.
    
- **On Large-Cohort Training for Federated Learning**
    >In this work we explore the benefits and limitations of large-cohort training in federated learning. As
    discussed in Sections 3.5 and 5, focusing on the number of communication rounds often obscures
    the data efficiency of a method. This in turn impacts many metrics important to society, such as
    total energy consumption or total carbon emissions. While we show that large-cohort training can
    negatively impact such metrics by reducing data-efficiency (see Section 3.5 and Appendix B.5), a
    more specialized focus on these issues is warranted

- **A Reputation Mechanism Is All You Need: Collaborative Fairness and Adversarial Robustness in Federated Learning**
    >We propose a Robust and Fair Federated Learning (RFFL) framework to simultaneously achieve collaborative fairness and adversarial robustness. RFFL utilizes a reputation system to iteratively calculate participants’ contributions and reward participants accordingly with different models of performance commensurate with their contributions.

- **PERSONALIZED FEDERATED LEARNING WITH FIRST ORDER MODEL OPTIMIZATION**
    >We propose a flexible federated learning framework that allows clients to personalize to specific target data distributions irrespective of their available local training data. 2. Withinthisframework,weintroduceamethodtoefficientlycalculatetheoptimalweighted combination of uploaded models as a personalized federated update 3. Our method strongly outperforms other methods in non-IID federated learning settings.

- **HETEROFL: COMPUTATION AND COMMUNICATION EFFICIENT FEDERATED LEARNING FOR HETEROGE- NEOUS CLIENTS**
    >In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation com- plexities and still produce a single global inference model

- **FEDBE: MAKING BAYESIAN MODEL ENSEMBLE APPLICABLE TO FEDERATED LEARNING**
    >In this paper, we propose a novel aggregation algorithm named FEDBE, which takes a Bayesian inference perspective by sampling
    higher-quality global models and combining them via Bayesian model Ensemble,
    leading to much robust aggregation. We show that an effective model distribution
    can be constructed by simply fitting a Gaussian or Dirichlet distribution to the local
    models.


- **HybridAlpha: An Efficient Approach for Privacy-Preserving Federated Learning**
    >In this paper, we propose HybridAlpha, an approach for privacy-preserving
    federated learning employing an SMC protocol based on functional
    encryption. This protocol is simple, efficient and resilient to participants dropping out. We evaluate our approach regarding the
    training time and data volume exchanged using a federated learning
    process to train a CNN on the MNIST data set

- **HybridAlpha: Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning**
    >We devise a new attack on distributed deep learning based on GANs. Our attack is more generic and effective than current information extraction mechanisms. The attack we devise is also effective when parameters are obfuscated via differential privacy. 


- **Personalized Federated Learning with Moreau Envelopes**
    >In this paper, we propose pFedMe as a personalized FL algorithm that can adapt to the statistical
diversity issue to improve the FL performance. Our approach makes use of the Moreau envelope
function which helps decompose the personalized model optimization from global model learning,
which allows pFedMe to update the global model similarly to FedAvg, yet in parallel to optimize the
personalized model w.r.t each client’s local data distribution


- **CRFL: Certifiably Robust Federated Learning against Backdoor Attacks**
    >In this paper, we propose pFedMe as a personalized FL algorithm that can adapt to the statistical
diversity issue to improve the FL performance. Our approach makes use of the Moreau envelope
function which helps decompose the personalized model optimization from global model learning,
which allows pFedMe to update the global model similarly to FedAvg, yet in parallel to optimize the
personalized model w.r.t each client’s local data distribution

 


- **DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection**
    > We present several new techniques (DDifs, NEUPs, Threshold Exceedings) to infer information about a model’s training data, identify similar models, and measure the homogeneity of model updates. By performing a deep inspection of the models’ structure and their predictions, DeepSight can effectively mitigate state-of-the-art poisoning attacks and is robust against sophisticated attacks, without degrading the performance of the aggregated model.


- **FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION**
    >  FedDyn is based on exact minimization, wherein at each round, each participating device, dynamically
updates its regularizer so that the optimal model for the regularized loss is in conformity with the
global empirical loss. Our approach is different from prior works that attempt to parallelize gradient
computation, and in doing so they tradeoff target accuracy with communications, and necessitate
inexact minimization

- **An Efficient Framework for Clustered Federated Learning**
    >  We address the problem of Federated Learning (FL) where users are distributed
and partitioned into clusters. This setup captures settings where different groups
of users have their own objectives (learning tasks) but by aggregating their data
with others in the same cluster (same learning task), they can leverage the strength
in numbers in order to perform more efficient Federated Learning. We propose
a new framework dubbed the Iterative Federated Clustering Algorithm (IFCA),
which alternately estimates the cluster identities of the users and optimizes model
parameters for the user clusters via gradient descent


- **FEDBN: FEDERATED LEARNING ON NON-IID FEATURES VIA LOCAL BATCH NORMALIZATION**
    >  This work proposes a novel federated learning aggregation method called FedBN that keeps the
local Batch Normalization parameters not synchronized with the global model, such that it mitigates
feature shifts in non-IID data. We provide convergence guarantees for FedBN in realistic federated
settings under the overparameterized neural networks regime, while also accounting for practical
issues

- **ADAPTIVE FEDERATED OPTIMIZATION**
    >  In this work, we propose federated versions of adaptive optimizers, including ADAGRAD, ADAM, and YOGI,
and analyze their convergence in the presence of heterogeneous data for general
nonconvex settings. Our results highlight the interplay between client heterogeneity
and communication efficiency.

- **Federated Learning on Non-IID Data Silos: An Experimental Study**
    >  In this paper, we study non-IID data as one key challenge in such distributed databases, and develop a benchmark named NIID- bench. Specifically, we introduce six data partitioning strate- gies which are much more comprehensive than the previous studies. Furthermore, we conduct comprehensive experiments to compare existing algorithms and demonstrate their strength and weakness.


- **Federated Optimization for Heterogeneous Networks**
    > We propose a federated optimization framework for heterogeneous networks, FedProx, which encompasses FedAvg. In order to characterize the convergence behavior of FedProx, we invoke a device dissimilarity assumption in the network. Under this assumption, we provide the first convergence guarantees for FedProx. Finally, we demonstrate that our theoretical assumptions reflect empirical performance, and that FedProx can improve the robustness and stability of convergence over FedAvg when data is heterogeneous across devices.

- **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**
    > We then proposed a new stochastic algorithm (SCAFFOLD) which overcomes gradient dissimilarity using control variates. We demonstrated the effectiveness of SCAFFOLD via strong convergence guarantees and empirical evaluations. Further, we showed that while SCAFFOLD is always at least as fast as SGD, it can be much faster depending on the Hessian dissimilarity in our data. Thus, different algorithms can take advantage of (and are limited by) different notions of dissimilarity. 

- **Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization**
    > This paper provides a general framework to analyze the convergence of federated heterogeneous optimization algorithms. It subsumes previously proposed methods such as FedAvg and FedProx and provides the first principled understanding of the solution bias and the convergence slowdown due to objective inconsistency. Using insights from this analysis, we propose FedNova, a normalized averaging method that eliminates objective inconsistency while preserving fast error convergence

- **Long-tail learning via logit adjustment**
    > Real-world classification problems typically exhibit an imbalanced or long-tailed label distribution, wherein many labels are associated with only a few samples. This poses a challenge for generalisation on such labels, and also makes naïve learning biased towards dominant labels. In this paper, we present two simple modifications of standard softmax cross-entropy training to cope with these challenges. Our techniques revisit the classic idea of logit adjustment based on the label frequencies, either applied post-hoc to a trained model, or enforced in the loss during training. Such adjustment encourages a large relative margin between logits of rare versus dominant labels. These techniques unify and generalise several recent proposals in the literature, while possessing firmer statistical grounding and empirical performance.

- **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss**
    > Deep learning algorithms can fare poorly when the training dataset suffers from
heavy class-imbalance but the testing criterion requires good generalization on less
frequent classes. We design two novel methods to improve performance in such
scenarios. First, we propose a theoretically-principled label-distribution-aware
margin (LDAM) loss motivated by minimizing a margin-based generalization
bound. This loss replaces the standard cross-entropy objective during training
and can be applied with prior strategies for training with class-imbalance such as
re-weighting or re-sampling. Second, we propose a simple, yet effective, training
schedule that defers re-weighting until after the initial stage, allowing the model to
learn an initial representation while avoiding some of the complications associated
with re-weighting or re-sampling


- **SHARPNESS-AWARE MINIMIZATION FOR EFFICIENTLY IMPROVING GENERALIZATION**
    >  Motivated by prior work connecting the geometry of the loss
landscape and generalization, we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness. In particular,
our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie
in neighborhoods having uniformly low loss; this formulation results in a minmax optimization problem on which gradient descent can be performed efficiently. We present empirical results showing that SAM improves model generalization across a variety of benchmark datasets (e.g., CIFAR-{10, 100}, ImageNet, finetuning tasks) and models, yielding novel state-of-the-art performance
for several. Additionally, we find that SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specifically target learning with noisy labels


- **InstaHide: Instance-hiding Schemes for Private Distributed Learning**
    > This paper introduces InstaHide, a simple encryption of training images, which can be plugged into an existing distributed deep learning pipeline. The encryption
is efficient and has minor effect on test accuracy.
InstaHide encrypts each training image with a
“one-time secret key” which consists of mixing
a number of randomly chosen images and applying a random pixel-wise mask. Other contributions of this paper include: (a) Using a large public dataset (e.g. ImageNet) for mixing during its
encryption, which improves security. (b) Experimental results to show effectiveness in preserving privacy against known attacks with only minor effects on accuracy. (c) Theoretical analysis
showing that successfully attacking privacy requires attackers to solve a difficult computational
problem. (d) Demonstrating that Mixup alone is
insecure (as contrary to recent proposals), by presenting some efficient attacks. (e) Release of a
challenge dataset1 to encourage new attacks

- **THE LOTTERY TICKET HYPOTHESIS:FINDING SPARSE, TRAINABLE NEURAL NETWORKS**
    > We find that a standard pruning technique naturally uncovers subnetworks whose
initializations made them capable of training effectively. Based on these results, we
articulate the lottery ticket hypothesis: dense, randomly-initialized, feed-forward
networks contain subnetworks (winning tickets) that—when trained in isolation—
reach test accuracy comparable to the original network in a similar number of
iterations. The winning tickets we find have won the initialization lottery: their
connections have initial weights that make training particularly effective.
We present an algorithm to identify winning tickets and a series of experiments
that support the lottery ticket hypothesis and the importance of these fortuitous
initializations. We consistently find winning tickets that are less than 10-20% of
the size of several fully-connected and convolutional feed-forward architectures
for MNIST and CIFAR10. Above this size, the winning tickets that we find learn
faster than the original network and reach higher test accuracy


- **Architecture Agnostic Federated Learning for Neural Networks**
    > This work introduces a novel framework, Federated Heterogeneous Neural Networks
(FedHeNN), that allows each client to build a
personalised model without enforcing a common
architecture across clients. This allows each client
to optimize with respect to local data and compute
constraints, while still benefiting from the learnings of other (potentially more powerful) clients.
The key idea of FedHeNN is to use the instancelevel representations obtained from peer clients
to guide the simultaneous training on each client.
The extensive experimental results demonstrate
that the FedHeNN framework is capable of learning better performing models on clients in both
the settings of homogeneous and heterogeneous
architectures across clients


- **Similarity of Neural Network Representations Revisited**
    > We examine methods for comparing neural network representations based on canonical
correlation analysis (CCA). We show that CCA
belongs to a family of statistics for measuring multivariate similarity, but that neither CCA nor any
other statistic that is invariant to invertible linear
transformation can measure meaningful similarities between representations of higher dimension
than the number of data points. We introduce
a similarity index that measures the relationship
between representational similarity matrices and
does not suffer from this limitation. This similarity index is equivalent to centered kernel alignment (CKA) and is also closely connected to CCA.
Unlike CCA, CKA can reliably identify correspondences between representations in networks
trained from different initializations


- **Grounding Representation Similarity with Statistical Testing**
    > To understand neural network behavior, recent works quantitatively compare different networks’ learned representations using canonical correlation analysis (CCA),
centered kernel alignment (CKA), and other dissimilarity measures. Unfortunately,
these widely used measures often disagree on fundamental observations, such
as whether deep networks differing only in random initialization learn similar
representations. These disagreements raise the question: which, if any, of these
dissimilarity measures should we believe? We provide a framework to ground
this question through a concrete test: measures should have sensitivity to changes
that affect functional behavior, and specificity against changes that do not. We
quantify this through a variety of functional behaviors including probing accuracy
and robustness to distribution shift, and examine changes such as varying random
initialization and deleting principal components. We find that current metrics exhibit different weaknesses, note that a classical baseline performs surprisingly well,
and highlight settings where all metrics appear to fail, thus providing a challenge
set for further improvement.


- **Grounding Representation Similarity with Statistical Testing**
    > To understand neural network behavior, recent works quantitatively compare different networks’ learned representations using canonical correlation analysis (CCA),
centered kernel alignment (CKA), and other dissimilarity measures. Unfortunately,
these widely used measures often disagree on fundamental observations, such
as whether deep networks differing only in random initialization learn similar
representations. These disagreements raise the question: which, if any, of these
dissimilarity measures should we believe? We provide a framework to ground
this question through a concrete test: measures should have sensitivity to changes
that affect functional behavior, and specificity against changes that do not. We
quantify this through a variety of functional behaviors including probing accuracy
and robustness to distribution shift, and examine changes such as varying random
initialization and deleting principal components. We find that current metrics exhibit different weaknesses, note that a classical baseline performs surprisingly well,
and highlight settings where all metrics appear to fail, thus providing a challenge
set for further improvement.


- **THE LAZY NEURON PHENOMENON: ON EMERGENCE OF ACTIVATION SPARSITY IN TRANSFORMERS**
    > This paper studies a curious phenomenon that machine learning model with Transformer architectures have sparse activation maps. By activation map we refer
to the intermediate output of the multi-layer perceptrons (MLPs) after a ReLU
activation function, and by “sparse” we mean that on average very few entries
(e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input to
MLP. Moreover, larger Transformers with more layers and wider MLP hidden
dimensions are sparser as measured by the percentage of nonzero entries. Through
extensive experiments we demonstrate that the emergence of sparsity is a prevalent
phenomenon that occurs for both natural language processing and vision tasks,
on both training and evaluation data, for Transformers of various configurations,
at layers of all depth levels. We discuss how sparsity immediately implies a way
to significantly reduce the FLOP count and improve efficiency for Transformers.
Moreover, we demonstrate perhaps surprisingly that enforcing an even sparser
activation via Top-k thresholding with a small k brings a collection of desired
properties, namely less sensitivity to noisy training data, more robustness to input
corruptions, and better calibration for their prediction confidence.


- **Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time**
    >  We hypothesize that contextual sparsity,
which are small, input-dependent sets of attention
heads and MLP parameters that yield approximately the same output as the dense model for a
given input, can address these issues. We show that
contextual sparsity exists, that it can be accurately
predicted, and that we can exploit it to speed up
LLM inference in wall-clock time without compromising LLM’s quality or in-context learning ability.
Based on these insights, we propose DEJAVU, a
system that uses a low-cost algorithm to predict
contextual sparsity on the fly given inputs to each
layer, along with an asynchronous and hardwareaware implementation that speeds up LLM
inference. We validate that DEJAVU can reduce the
inference latency of OPT-175B by over 2× compared to the state-of-the-art FasterTransformer,
and over 6× compared to the widely used Hugging
Face implementation, without compromising
model quality. 

- **LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation**
    > To reduce the size and complexity of these models, we propose LoSparse (Low-Rank and Sparse
approximation), a novel model compression technique that approximates a weight matrix by the
sum of a low-rank matrix and a sparse matrix. Our method combines the advantages of both
low-rank approximations and pruning, while avoiding their limitations. Low-rank approximation
compresses the coherent and expressive parts in neurons, while pruning removes the incoherent
and non-expressive parts in neurons. Pruning enhances the diversity of low-rank approximations,
and low-rank approximation prevents pruning from losing too many expressive neurons. We evaluate our method on natural language understanding, question answering, and natural language
generation tasks. 

- **PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance**
    > To reduce the model
size, researchers prune these models based on the
weights’ importance scores. However, such scores
are usually estimated on mini-batches during training, which incurs large variability/uncertainty due
to mini-batch sampling and complicated training
dynamics. As a result, some crucial weights could
be pruned by commonly used pruning methods
because of such uncertainty, which makes training unstable and hurts generalization. To resolve
this issue, we propose PLATON, which captures
the uncertainty of importance scores by upper
confidence bound (UCB) of importance estimation. In particular, for the weights with low importance scores but high uncertainty, PLATON
tends to retain them and explores their capacity.
We conduct extensive experiments with several
Transformer-based models on natural language
understanding, question answering and image
classification to validate the effectiveness of PLATON. 

- **LLM-Pruner: On the Structural Pruning of Large Language Models**
    >  we tackle the compression of LLMs within the bound of two constraints: being taskagnostic and minimizing the reliance on the original training dataset. Our method,
named LLM-Pruner, adopts structural pruning that selectively removes non-critical
coupled structures based on gradient information, maximally preserving the majority of the LLM’s functionality. To this end, the performance of pruned models
can be efficiently recovered through tuning techniques, LoRA, in merely 3 hours,
requiring only 50K data. We validate the LLM-Pruner on three LLMs, including
LLaMA, Vicuna, and ChatGLM, and demonstrate that the compressed models still
exhibit satisfactory capabilities in zero-shot classification and generation


- **Movement Pruning: Adaptive Sparsity by Fine-Tuning**
    > We propose the use of movement pruning, a simple, deterministic first-order weight
pruning method that is more adaptive to pretrained model fine-tuning. We give
mathematical foundations to the method and compare it to existing zeroth- and
first-order pruning methods. Experiments show that when pruning large pretrained
language models, movement pruning shows significant improvements in highsparsity regimes. When combined with distillation, the approach achieves minimal
accuracy loss with down to only 3% of the model parameters.

- **Pruning neural networks without any data by iteratively conserving synaptic flow**
    > We first mathematically
formulate and experimentally verify a conservation law that explains why existing
gradient-based pruning algorithms at initialization suffer from layer-collapse, the
premature pruning of an entire layer rendering a network untrainable. This theory
also elucidates how layer-collapse can be entirely avoided, motivating a novel
pruning algorithm Iterative Synaptic Flow Pruning (SynFlow). This algorithm can
be interpreted as preserving the total flow of synaptic strengths through the network
at initialization subject to a sparsity constraint. Notably, this algorithm makes
no reference to the training data and consistently competes with or outperforms
existing state-of-the-art pruning algorithms at initialization over a range of models


- **Winning the Lottery with Continuous Sparsification**
    >  We revisit fundamental aspects of pruning algorithms, pointing out missing ingredients
in previous approaches, and develop a method, Continuous Sparsification, which
searches for sparse networks based on a novel approximation of an intractable 0
regularization. We compare against dominant heuristic-based methods on pruning
as well as ticket search – finding sparse subnetworks that can be successfully
re-trained from an early iterate. Empirical results show that we surpass the state-ofthe-art for both objectives, across models and datasets, including VGG trained on
CIFAR-10 and ResNet-50 trained on ImageNet. In addition to setting a new standard for pruning, Continuous Sparsification also offers fast parallel ticket search,
opening doors to new applications of the Lottery Ticket Hypothesis.

- **CRAM: A COMPRESSION-AWARE MINIMIZER**
    > In this work we propose a new compression-aware minimizer dubbed CrAM that modifies the
optimization step in a principled way, in order to produce models whose local loss
behavior is stable under compression operations such as pruning. Thus, dense models trained via CrAM should be compressible post-training, in a single step, without
significant accuracy loss

- **M-FAC: Efficient Matrix-Free Approximations of Second-Order Information**
    >  In this work, we investigate
matrix-free, linear-time approaches for estimating Inverse-Hessian Vector Products
(IHVPs) for the case when the Hessian can be approximated as a sum of rank-one
matrices, as in the classic approximation of the Hessian by the empirical Fisher
matrix. We propose two new algorithms: the first is tailored towards network
compression and can compute the IHVP for dimension d, if the Hessian is given
as a sum of m rank-one matrices, using O(dm2
) precomputation, O(dm) cost for
computing the IHVP, and query cost O(m) for any single element of the inverse
Hessian. The second algorithm targets an optimization setting, where we wish to
compute the product between the inverse Hessian, estimated over a sliding window
of optimization steps, and a given gradient direction, as required for preconditioned
SGD

- **Momentum Provably Improves Error Feedback!**
    > e. In this work we address one of the most
pressing issues. In particular, in the canonical nonconvex setting, all known variants of EF rely on very
large batch sizes to converge, which can be prohibitive in practice. We propose a surprisingly simple
fix which removes this issue both theoretically, and in practice: the application of Polyak’s momentum
to the latest incarnation of EF due to Richtárik et al. [2021] known as EF21. Our algorithm, for which
we coin the name EF21-SGDM, improves the communication and sample complexities of previous
error feedback algorithms under standard smoothness and bounded variance assumptions, and does
not require any further strong assumptions such as bounded gradient dissimilarity. Moreover, we
propose a double momentum version of our method that improves the complexities even further

- **2Direction: Theoretically Faster Distributed Training with Bidirectional Communication Compression**
    > We develop a new and provably accelerated method,
which we call 2Direction, based on fast bidirectional compressed communication
and a new bespoke error-feedback mechanism which may be of independent interest. Indeed, we find that the EF and EF21-P mechanisms (Seide et al., 2014;
Gruntkowska et al., 2023) that have considerable success in the design of efficient
non-accelerated methods are not appropriate for accelerated methods. In particular, we prove that 2Direction improves the previous state-of-the-art communication complexity

- **Recovering Private Text in Federated Learning of Language Models**
    >  In this paper, we present a novel attack method FILM
for federated learning of language models (LMs). For the first time, we show
the feasibility of recovering text from large batch sizes of up to 128 sentences.
Unlike image-recovery methods that are optimized to match gradients, we take
a distinct approach that first identifies a set of words from gradients and then
directly reconstructs sentences based on beam search and a prior-based reordering
strategy. We conduct the FILM attack on several large-scale datasets and show
that it can successfully reconstruct single sentences with high fidelity for large
batch sizes and even multiple sentences if applied iteratively. 


- **Neurotoxin: Durable Backdoors in Federated Learning**
    > Prior work has shown that backdoors can be inserted into FL models, but these
backdoors are often not durable, i.e., they do not
remain in the model after the attacker stops uploading poisoned updates. Thus, since training
typically continues progressively in production
FL systems, an inserted backdoor may not survive until deployment. Here, we propose Neurotoxin, a simple one-line modification to existing
backdoor attacks that acts by attacking parameters
that are changed less in magnitude during training

- **DADAQUANT: DOUBLY-ADAPTIVE QUANTIZATION FOR COMMUNICATION-EFFICIENT FEDERATED LEARNING**
    > We find that dynamic adaptations of the quantization level
can boost compression without sacrificing model quality. First, we introduce a
time-adaptive quantization algorithm that increases the quantization level as training progresses. Second, we introduce a client-adaptive quantization algorithm
that assigns each individual client the optimal quantization level at every round.
Finally, we combine both algorithms into DAdaQuant, the doubly-adaptive quantization algorithm. 

- **Defending against Backdoors in Federated Learning with Robust Learning Rate**
    >  To prevent backdoor attacks, we propose a lightweight defense that requires minimal change to the
FL protocol. At a high level, our defense is based on carefully
adjusting the aggregation server’s learning rate, per dimension and per round, based on the sign information of agents’
updates. We first conjecture the necessary steps to carry a
successful backdoor attack in FL setting, and then, explicitly
formulate the defense based on our conjecture. Through experiments, we provide empirical evidence that supports our
conjecture, and we test our defense against backdoor attacks
under different settings. We observe that either backdoor is
completely eliminated, or its accuracy is significantly reduced.

- **FedNAR: Federated Optimization with Normalized Annealing Regularization**
    > In this paper, we first explore
the choices of weight decay and identify that weight decay value appreciably influences the convergence of existing FL algorithms. While preventing overfitting
is crucial, weight decay can introduce a different optimization goal towards the
global objective, which is further amplified in FL due to multiple local updates and
heterogeneous data distribution. To address this challenge, we develop Federated
optimization with Normalized Annealing Regularization (FedNAR), a simple yet
effective and versatile algorithmic plug-in that can be seamlessly integrated into
any existing FL algorithms. Essentially, we regulate the magnitude of each update
by performing co-clipping of the gradient and weight decay. We provide a comprehensive theoretical analysis of FedNAR’s convergence rate and conduct extensive
experiments on both vision and language datasets with different backbone federated
optimization algorithms. 

- **Federated Composite Optimization**
    > In this paper, we study the Federated Composite Optimization (FCO) problem, in which the
loss function contains a non-smooth regularizer. Such problems arise naturally in FL applications that
involve sparsity, low-rank, monotonicity, or more general constraints. We first show that straightforward
extensions of primal algorithms such as FedAvg are not well-suited for FCO since they suffer from the
“curse of primal averaging,” resulting in poor convergence. As a solution, we propose a new primal-dual
algorithm, Federated Dual Averaging (FedDualAvg), which by employing a novel server dual averaging
procedure circumvents the curse of primal averaging

- **Faster Single-loop Algorithms for Minimax Optimization without Strong Concavity**
    > Gradient descent ascent (GDA), the simplest
single-loop algorithm for nonconvex minimax optimization, is widely used in practical
applications such as generative adversarial
networks (GANs) and adversarial training.
Albeit its desirable simplicity, recent work
shows inferior convergence rates of GDA in
theory, even when assuming strong concavity
of the objective in terms of one variable. This
paper establishes new convergence results for
two alternative single-loop algorithms – alternating GDA and smoothed GDA – under
the mild assumption that the objective satisfies the Polyak- Lojasiewicz (PL) condition
about one variable.

- **Revisiting Weighted Aggregation in Federated Learning with Neural Networks**
    > . In this paper, we revisit the
weighted aggregation process and gain new insights into the training dynamics of FL. First, we
find that the sum of weights can be smaller than 1,
causing global weight shrinking effect (analogous
to weight decay) and improving generalization.
We explore how the optimal shrinking factor
is affected by clients’ data heterogeneity and
local epochs. Second, we dive into the relative
aggregation weights among clients to depict the
clients’ importance. We develop client coherence
to study the learning dynamics and find a critical
point that exists. Before entering the critical point,
more coherent clients play more essential roles
in generalization. Based on the above insights,
we propose an effective method for Federated
Learning with Learnable Aggregation Weights,
named as FEDLAW 

- **Are Emergent Abilities of Large Language Models a Mirage?**
    > Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due the
researcher’s choice of metric rather than due to fundamental changes in model
behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth,
continuous, predictable changes in model performance. We present our alternative
explanation in a simple mathematical model, then test it in three complementary
ways

- **A Single-Loop Accelerated Extra-Gradient Difference Algorithm with Improved Complexity Bounds for Constrained Minimax Optimization**
    > In this paper, we propose a novel extra-gradient difference acceleration algorithm
for solving constrained nonconvex-nonconcave (NC-NC) minimax problems. In
particular, we design a new extra-gradient difference step to obtain an important
quasi-cocoercivity property, which plays a key role to significantly improve the
convergence rate in the constrained NC-NC setting without additional structural
assumption. Then momentum acceleration is also introduced into our dual accelerating update step

- **Privacy Auditing with One (1) Training Run**
    > We propose a scheme for auditing differentially private machine learning systems
with a single training run. This exploits the parallelism of being able to add or remove
multiple training examples independently. We analyze this using the connection between differential privacy and statistical generalization, which avoids the cost of group
privacy. Our auditing scheme requires minimal assumptions about the algorithm and
can be applied in the black-box or white-box setting

- **Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection**
    > Neural sequence models based on the transformer architecture have demonstrated remarkable incontext learning (ICL) abilities, where they can perform new tasks when prompted with training and
test examples, without any parameter update to the model. This work advances the understandings of
the strong ICL abilities of transformers. We first provide a comprehensive statistical theory for transformers to perform ICL by deriving end-to-end quantitative results for the expressive power, in-context
prediction power, and sample complexity of pretraining. Concretely, we show that transformers can
implement a broad class of standard machine learning algorithms in context, such as least squares, ridge
regression, Lasso, convex risk minimization for generalized linear models (such as logistic regression), and
gradient descent on two-layer neural networks, with near-optimal predictive power on various in-context
data distributions. Using an efficient implementation of in-context gradient descent as the underlying
mechanism, our transformer constructions admit mild bounds on the number of layers and heads, and
can be learned with polynomially many pretraining sequences.
Building on these “base” ICL algorithms, intriguingly, we show that transformers can implement more
complex ICL procedures involving in-context algorithm selection, akin to what a statistician can do in
real life—A single transformer can adaptively select different base ICL algorithms—or even perform
qualitatively different tasks—on different input sequences, without any explicit prompting of the right
algorithm or task. We both establish this in theory by explicit constructions, and also observe this
phenomenon experimentally

- **Jailbroken: How Does LLM Safety Training Fail?**
    > Large language models trained for safety and harmlessness remain susceptible to
adversarial misuse, as evidenced by the prevalence of “jailbreak” attacks on early
releases of ChatGPT that elicit undesired behavior. Going beyond recognition of
the issue, we investigate why such attacks succeed and how they can be created.
We hypothesize two failure modes of safety training: competing objectives and
mismatched generalization. Competing objectives arise when a model’s capabilities
and safety goals conflict, while mismatched generalization occurs when safety
training fails to generalize to a domain for which capabilities exist. We use these
failure modes to guide jailbreak design and then evaluate state-of-the-art models,
including OpenAI’s GPT-4 and Anthropic’s Claude v1.3, against both existing and
newly designed attacks. We find that vulnerabilities persist despite the extensive
red-teaming and safety-training efforts behind these models. Notably, new attacks
utilizing our failure modes succeed on every prompt in a collection of unsafe
requests from the models’ red-teaming evaluation sets and outperform existing ad
hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity—that
safety mechanisms should be as sophisticated as the underlying model—and argues
against the idea that scaling alone can resolve these safety failure modes.

- **Sharpness Minimization Algorithms Do Not Only Minimize Sharpness To Achieve Better Generalization**
    > Despite extensive studies, the underlying reason as to why overparameterized neural networks can generalize remains elusive. Existing theory shows that common stochastic optimizers prefer flatter minimizers
of the training loss, and thus a natural potential explanation is that flatness implies generalization. This
work critically examines this explanation. Through theoretical and empirical investigation, we identify the
following three scenarios for two-layer ReLU networks: (1) flatness provably implies generalization; (2)
there exist non-generalizing flattest models and sharpness minimization algorithms fail to generalize, and
(3) perhaps most surprisingly, there exist non-generalizing flattest models, but sharpness minimization algorithms still generalize. Our results suggest that the relationship between sharpness and generalization subtly
depends on the data distributions and the model architectures and sharpness minimization algorithms do not
only minimize sharpness to achieve better generalization. This calls for the search for other explanations for the generalization of over-parameterized neural networks.

- **Abide by the Law and Follow the Flow: Conservation Laws for Gradient Flows**
    > Understanding the geometric properties of gradient descent dynamics is a key ingredient in deciphering the recent success of very large machine learning models. A striking observation is that trained over-parameterized models retain some
properties of the optimization initialization. This “implicit bias” is believed to be
responsible for some favorable properties of the trained models and could explain
their good generalization properties. The purpose of this article is threefold. First,
we rigorously expose the definition and basic properties of “conservation laws”,
which are maximal sets of independent quantities conserved during gradient flows
of a given model (e.g. of a ReLU network with a given architecture) with any
training data and any loss. Then we explain how to find the exact number of these
quantities by performing finite-dimensional algebraic manipulations on the Lie
algebra generated by the Jacobian of the model. Finally, we provide algorithms
(implemented in SageMath) to: a) compute a family of polynomial laws; b) compute the number of (not necessarily polynomial) conservation laws. We provide
showcase examples that we fully work out theoretically. Besides, applying the two
algorithms confirms for a number of ReLU network architectures that all known
laws are recovered by the algorithm, and that there are no other laws. Such computational tools pave the way to understanding desirable properties of optimization
initialization in large machine learning models

- **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
    > Language models are increasingly being deployed for general problem solving
across a wide range of tasks, but are still confined to token-level, left-to-right
decision-making processes during inference. This means they can fall short in
tasks that require exploration, strategic lookahead, or where initial decisions play
a pivotal role. To surmount these challenges, we introduce a new framework for
language model inference, “Tree of Thoughts” (ToT), which generalizes over the
popular “Chain of Thought” approach to prompting language models, and enables
exploration over coherent units of text (“thoughts”) that serve as intermediate steps
toward problem solving. ToT allows LMs to perform deliberate decision making
by considering multiple different reasoning paths and self-evaluating choices to
decide the next course of action, as well as looking ahead or backtracking when
necessary to make global choices. Our experiments show that ToT significantly
enhances language models’ problem-solving abilities on three novel tasks requiring
non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords.
For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only
solved 4% of tasks, our method achieved a success rate of 74%

- **Students Parrot Their Teachers: Membership Inference on Model Distillation**
    > Model distillation is frequently proposed as a technique to reduce the privacy leakage of machine
learning. These empirical privacy defenses rely on the intuition that distilled “student” models protect
the privacy of training data, as they only interact with this data indirectly through a “teacher” model.
In this work, we design membership inference attacks to systematically study the privacy provided
by knowledge distillation to both the teacher and student training sets. Our new attacks show that
distillation alone provides only limited privacy across a number of domains. We explain the success
of our attacks on distillation by showing that membership inference attacks on a private dataset can
succeed even if the target model is never queried on any actual training points, but only on inputs whose
predictions are highly influenced by training data. Finally, we show that our attacks are strongest when
student and teacher sets are similar, or when the attacker can poison the teacher set.

- **Learning Transformer Programs**
    > Recent research in mechanistic interpretability has attempted to reverse-engineer
Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of
providing complete, faithful descriptions of the underlying algorithms. In this
work, we introduce a procedure for training Transformers that are mechanistically
interpretable by design. We build on RASP [Weiss et al., 2021], a programming
language that can be compiled into Transformer weights. Instead of compiling
human-written programs into Transformers, we design a modified Transformer
that can be trained using gradient-based optimization and then be automatically
converted into a discrete, human-readable program. We refer to these models as
Transformer Programs. To validate our approach, we learn Transformer Programs
for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck-languages), and NLP tasks including
named entity recognition and text classification. The Transformer Programs can
automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To
demonstrate these advantages, we convert Transformers into Python programs
and use off-the-shelf code analysis tools to debug model errors and identify the
“circuits” used to solve different sub-problems. We hope that Transformer Programs
open a new path toward the goal of intrinsically interpretable machine learning


- **Bridging Discrete and Backpropagation: Straight-Through and Beyond**
    > Backpropagation, the cornerstone of deep learning, is limited to computing gradients for continuous variables. This limitation poses challenges for problems
involving discrete latent variables. To address this issue, we propose a novel approach to approximate the gradient of parameters involved in generating discrete
latent variables. First, we examine the widely used Straight-Through (ST) heuristic
and demonstrate that it works as a first-order approximation of the gradient. Guided
by our findings, we propose ReinMax, which achieves second-order accuracy by
integrating Heun’s method, a second-order numerical method for solving ODEs.
ReinMax does not require Hessian or other second-order derivatives, thus having
negligible computation overheads. Extensive experimental results on various tasks
demonstrate the superiority of ReinMax over the state of the art.

- **THE LAZY NEURON PHENOMENON: ON EMERGENCE**
    > This paper studies the curious phenomenon for machine learning models with Transformer architectures
that their activation maps are sparse. By activation map we refer to the intermediate output of the multi-layer
perceptrons (MLPs) after a ReLU activation function, and by “sparse” we mean that on average very few
entries (e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input to MLP. Moreover, larger
Transformers with more layers and wider MLP hidden dimensions are sparser as measured by the percentage
of nonzero entries. Through extensive experiments we demonstrate that the emergence of sparsity is a
prevalent phenomenon that occurs for both natural language processing and vision tasks, on both training and
evaluation data, for Transformers of various configurations, at layers of all depth levels, as well as for other
architectures including MLP-mixers and 2-layer MLPs. We show that sparsity also emerges using training
datasets with random labels, or with random inputs, or with infinite amount of data, demonstrating that
sparsity is not a result of a specific family of datasets. We discuss how sparsity immediately implies a way
to significantly reduce the FLOP count and improve efficiency for Transformers. Moreover, we demonstrate
perhaps surprisingly that enforcing an even sparser activation via Top-k thresholding with a small value of k
brings a collection of desired but missing properties for Transformers, namely less sensitivity to noisy training
data, more robustness to input corruptions, and better calibration for their prediction confidence.

- **Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models**  
    > Task arithmetic has recently emerged as a cost-effective and scalable approach to
edit pre-trained models directly in weight space: By adding the fine-tuned weights
of different tasks, the model’s performance can be improved on these tasks, while
negating them leads to task forgetting. Yet, our understanding of the effectiveness
of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight
disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show
that fine-tuning models in their tangent space by linearizing them amplifies weight
disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we
provide theoretical and empirical analyses of the neural tangent kernel (NTK) of
these models and establish a compelling link between task arithmetic and the spatial
localization of the NTK eigenfunctions


- **Private Everlasting Prediction**
    > A private learner is trained on a sample of labeled points and generates a hypothesis that can be
used for predicting the labels of newly sampled points while protecting the privacy of the training set
[Kasiviswannathan et al., FOCS 2008]. Research uncovered that private learners may need to exhibit
significantly higher sample complexity than non-private learners as is the case with, e.g., learning of
one-dimensional threshold functions [Bun et al., FOCS 2015, Alon et al., STOC 2019].
We explore prediction as an alternative to learning. Instead of putting forward a hypothesis, a predictor
answers a stream of classification queries. Earlier work has considered a private prediction model with
just a single classification query [Dwork and Feldman, COLT 2018]. We observe that when answering a
stream of queries, a predictor must modify the hypothesis it uses over time, and, furthermore, that it must
use the queries for this modification, hence introducing potential privacy risks with respect to the queries
themselves.
We introduce private everlasting prediction taking into account the privacy of both the training set and
the (adaptively chosen) queries made to the predictor. We then present a generic construction of private
everlasting predictors in the PAC model. The sample complexity of the initial training sample in our
construction is quadratic (up to polylog factors) in the VC dimension of the concept class. Our construction
allows prediction for all concept classes with finite VC dimension, and in particular threshold functions
with constant size initial training sample, even when considered over infinite domains, whereas it is known
that the sample complexity of privately learning threshold functions must grow as a function of the domain
size and hence is impossible for infinite domains

- **ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers**
    > In this work, we present an efficient and affordable post-training quantization approach to compress large
Transformer-based models, termed as ZeroQuant. ZeroQuant is an end-to-end quantization and inference
pipeline with three main components: (1) a fine-grained hardware-friendly quantization scheme for both
weight and activations; (2) a novel affordable layer-by-layer knowledge distillation algorithm (LKD) even
without the access to the original training data; (3) a highly-optimized quantization system backend
support to remove the quantization/dequantization overhead

- **PYHESSIAN: Neural Networks Through the Lens of the Hessian**
    > We present PYHESSIAN, a new scalable framework
that enables fast computation of Hessian (i.e., second-order
derivative) information for deep neural networks. PYHESSIAN
enables fast computations of the top Hessian eigenvalues, the Hessian trace, and the full Hessian eigenvalue/spectral density, and it
supports distributed-memory execution on cloud/supercomputer
systems and is available as open source [1]. This general framework can be used to analyze neural network models, including
the topology of the loss landscape (i.e., curvature information)
to gain insight into the behavior of different models/optimizers.
To illustrate this, we analyze the effect of residual connections
and Batch Normalization layers on the trainability of neural
networks. One recent claim, based on simpler first-order analysis,
is that residual connections and Batch Normalization make the
loss landscape “smoother”, thus making it easier for Stochastic
Gradient Descent to converge to a good solution. Our extensive
analysis shows new finer-scale insights, demonstrating that, while
conventional wisdom is sometimes validated, in other cases it is
simply incorrect. In particular, we find that Batch Normalization
does not necessarily make the loss landscape smoother, especially
for shallower network

- **Learning Transferable Features with Deep Adaptation Networks**
    > In this paper, we propose a new Deep Adaptation
Network (DAN) architecture, which generalizes
deep convolutional neural network to the domain
adaptation scenario. In DAN, hidden representations of all task-specific layers are embedded in a
reproducing kernel Hilbert space where the mean
embeddings of different domain distributions can
be explicitly matched. The domain discrepancy
is further reduced using an optimal multi-kernel
selection method for mean embedding matching.
DAN can learn transferable features with statistical guarantees, and can scale linearly by unbiased
estimate of kernel embedding. Extensive empirical evidence shows that the proposed architecture
yields state-of-the-art image classification error
rates on standard domain adaptation benchmarks

- **Certifying Some Distributional Robustness with Principled Adversarial Training**
    > Neural networks are vulnerable to adversarial examples and researchers have proposed many
heuristic attack and defense mechanisms. We address this problem through the principled lens
of distributionally robust optimization, which guarantees performance under adversarial input
perturbations. By considering a Lagrangian penalty formulation of perturbing the underlying
data distribution in a Wasserstein ball, we provide a training procedure that augments model
parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical
cost relative to empirical risk minimization. Furthermore, our statistical guarantees allow us
to efficiently certify robustness for the population loss. For imperceptible perturbations, our
method matches or outperforms heuristic approaches

- **A Closer Look at Memorization in Deep Networks**
    > We examine the role of memorization in deep
learning, drawing connections to capacity, generalization, and adversarial robustness. While
deep networks are capable of memorizing noise
data, our results suggest that they tend to prioritize learning simple patterns first. In our
experiments, we expose qualitative differences
in gradient-based optimization of deep neural
networks (DNNs) on noise vs. real data. We
also demonstrate that for appropriately tuned
explicit regularization (e.g., dropout) we can
degrade DNN training performance on noise
datasets without compromising generalization on
real data. Our analysis suggests that the notions
of effective capacity which are dataset independent are unlikely to explain the generalization
performance of deep networks when trained with
gradient based methods because training data itself plays an important role in determining the
degree of memorization

- **Visualizing Higher-Layer Features of a Deep Network**
    > Deep architectures have demonstrated state-of-the-art results in a variety of
settings, especially with vision datasets. Beyond the model definitions and the
quantitative analyses, there is a need for qualitative comparisons of the solutions
learned by various deep architectures. The goal of this paper is to find good qualitative interpretations of high level features represented by such models. To this end,
we contrast and compare several techniques applied on Stacked Denoising Autoencoders and Deep Belief Networks, trained on several vision datasets. We show
that, perhaps counter-intuitively, such interpretation is possible at the unit level,
that it is simple to accomplish and that the results are consistent across various
techniques. We hope that such techniques will allow researchers in deep architectures to understand more of how and why deep architectures work

- **Identifying and attacking the saddle point problem in high-dimensional non-convex optimization**
    > A central challenge to many fields of science and engineering involves minimizing
non-convex error functions over continuous, high dimensional spaces. Gradient descent
or quasi-Newton methods are almost ubiquitously used to perform such minimizations,
and it is often thought that a main source of difficulty for these local methods to find
the global minimum is the proliferation of local minima with much higher error than
the global minimum. Here we argue, based on results from statistical physics, random
matrix theory, neural network theory, and empirical evidence, that a deeper and more
profound difficulty originates from the proliferation of saddle points, not local minima,
especially in high dimensional problems of practical interest. Such saddle points are
surrounded by high error plateaus that can dramatically slow down learning, and give the
illusory impression of the existence of a local minimum. Motivated by these arguments,
we propose a new approach to second-order optimization, the saddle-free Newton method,
that can rapidly escape high dimensional saddle points, unlike gradient descent and
quasi-Newton methods. We apply this algorithm to deep or recurrent neural network
training, and provide numerical evidence for its superior optimization performance.


- **Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations**
    > We introduce a method to train Quantized Neural Networks (QNNs) — neural networks
with extremely low precision (e.g., 1-bit) weights and activations, at run-time. At traintime the quantized weights and activations are used for computing the parameter gradients.
During the forward pass, QNNs drastically reduce memory size and accesses, and replace
most arithmetic operations with bit-wise operations. As a result, power consumption is
expected to be drastically reduced. We trained QNNs over the MNIST, CIFAR-10, SVHN
and ImageNet datasets. The resulting QNNs achieve prediction accuracy comparable to
their 32-bit counterparts. For example, our quantized version of AlexNet with 1-bit weights
and 2-bit activations achieves 51% top-1 accuracy. Moreover, we quantize the parameter
gradients to 6-bits as well which enables gradients computation using only bit-wise operation. Quantized recurrent neural networks were tested over the Penn Treebank dataset, and
achieved comparable accuracy as their 32-bit counterparts using only 4-bits. Last but not
least, we programmed a binary matrix multiplication GPU kernel with which it is possible
to run our MNIST QNN 7 times faster than with an unoptimized GPU kernel, without
suffering any loss in classification accuracy. The QNN code is available online.


- **On the Number of Linear Regions of Deep Neural Networks**
    > We study the complexity of functions computable by deep feedforward neural networks with piecewise linear activations in terms of the symmetries and the number
of linear regions that they have. Deep networks are able to sequentially map portions of each layer’s input-space to the same output. In this way, deep models
compute functions that react equally to complicated patterns of different inputs.
The compositional structure of these functions enables them to re-use pieces of
computation exponentially often in terms of the network’s depth. This paper investigates the complexity of such compositional maps and contributes new theoretical
results regarding the advantage of depth for neural networks with piecewise linear
activation functions. In particular, our analysis is not specific to a single family of
models, and as an example, we employ it for rectifier and maxout networks. We
improve complexity bounds from pre-existing work and investigate the behavior
of units in higher layers


- **BinaryConnect: Training Deep Neural Networks with binary weights during propagations**
    >  As a result, there is much interest in research and development of dedicated hardware for Deep Learning (DL). Binary weights, i.e., weights
which are constrained to only two possible values (e.g. -1 or 1), would bring great
benefits to specialized DL hardware by replacing many multiply-accumulate operations by simple accumulations, as multipliers are the most space and powerhungry components of the digital implementation of neural networks. We introduce BinaryConnect, a method which consists in training a DNN with binary
weights during the forward and backward propagations, while retaining precision
of the stored weights in which gradients are accumulated. Like other dropout
schemes, we show that BinaryConnect acts as regularizer and we obtain near
state-of-the-art results with BinaryConnect on the permutation-invariant MNIST,
CIFAR-10 and SVHN

- **Efficient Algorithms for Smooth Minimax Optimization**
    > This paper studies first order methods for solving smooth minimax optimization
problems minx maxy g(x, y) where g(·, ·) is smooth and g(x, ·) is concave for each
x. In terms of g(·, y), we consider two settings – strongly convex and nonconvex –
and improve upon the best known rates in both

- **Near-Optimal Algorithms for Minimax Optimization**
    > This paper resolves a longstanding open question pertaining to the design of near-optimal first-order
algorithms for smooth and strongly-convex-strongly-concave minimax problems. Current state-ofthe-art first-order algorithms find an approximate Nash equilibrium using O˜(κx+κy) (Tseng, 1995)
or O˜(min{κx
√κy,
√κxκy}) (Alkousa et al., 2019) gradient evaluations, where κx and κy are the
condition numbers for the strong-convexity and strong-concavity assumptions. A gap still remains
between these results and the best existing lower bound Ω( ˜ √κxκy) (Ibrahim et al., 2019; Zhang
et al., 2019). This paper presents the first algorithm with O˜(
√κxκy) gradient complexity, matching the lower bound up to logarithmic factors. Our algorithm is designed based on an accelerated
proximal point method and an accelerated solver for minimax proximal steps. It can be easily extended to the settings of strongly-convex-concave, convex-concave, nonconvex-strongly-concave,
and nonconvex-concave functions. This paper also presents algorithms that match or outperform
all existing methods in these settings in terms of gradient complexity, up to logarithmic factors.


- **A Single-Loop Smoothed Gradient Descent-Ascent Algorithm for Nonconvex-Concave Min-Max Problems**
    > Nonconvex-concave min-max problem arises in many machine learning applications including
minimizing a pointwise maximum of a set of nonconvex functions and robust adversarial training
of neural networks. A popular approach to solve this problem is the gradient descent-ascent
(GDA) algorithm which unfortunately can exhibit oscillation in case of nonconvexity. In this
paper, we introduce a “smoothing” scheme which can be combined with GDA to stabilize the
oscillation and ensure convergence to a stationary solution. We prove that the stabilized GDA
algorithm can achieve an O(1/ǫ2
) iteration complexity for minimizing the pointwise maximum
of a finite collection of nonconvex functions. Moreover, the smoothed GDA algorithm achieves
an O(1/ǫ4
) iteration complexity for general nonconvex-concave problems. Extensions of this
stabilized GDA algorithm to multi-block cases are presented. To the best of our knowledge, this
is the first algorithm to achieve O(1/ǫ2
) for a class of nonconvex-concave problem. We illustrate
the practical efficiency of the stabilized GDA algorithm on robust training.


- **Extracting and Composing Robust Features with Denoising Autoencoders**
    > Previous work has shown that the difficulties in learning deep generative or discriminative models can be overcome by an initial unsupervised learning step that maps inputs to useful intermediate representations. We introduce and motivate a new training
principle for unsupervised learning of a representation based on the idea of making the
learned representations robust to partial corruption of the input pattern. This approach
can be used to train autoencoders, and these
denoising autoencoders can be stacked to initialize deep architectures. The algorithm can
be motivated from a manifold learning and
information theoretic perspective or from a
generative model perspective. Comparative
experiments clearly show the surprising advantage of corrupting the input of autoencoders on a pattern classification benchmark suite.

- **Training Deep Nets with Sublinear Memory Cost**
    > We propose a systematic approach to reduce the memory consumption of deep neural network training. Specifically, we design an algorithm that costs O(n) memory to train a n layer
network, with only the computational cost of an extra forward pass per mini-batch. As many of
the state-of-the-art models hit the upper bound of the GPU memory, our algorithm allows deeper
and more complex models to be explored, and helps advance the innovations in deep learning
research. We focus on reducing the memory cost to store the intermediate feature maps and gradients during training. Computation graph analysis is used for automatic in-place operation and
memory sharing optimizations. We show that it is possible to trade computation for memory
giving a more memory efficient training algorithm with a little extra computation cost. In the
extreme case, our analysis also shows that the memory consumption can be reduced to O(log n)
with as little as O(n log n) extra cost for forward computation. Our experiments show that we
can reduce the memory cost of a 1,000-layer deep residual network from 48G to 7G on ImageNet
problems. Similarly, significant memory cost reduction is observed in training complex recurrent
neural networks on very long sequences

- **On Calibration of Modern Neural Networks**
    > Confidence calibration – the problem of predicting probability estimates representative of the
true correctness likelihood – is important for
classification models in many applications. We
discover that modern neural networks, unlike
those from a decade ago, are poorly calibrated.
Through extensive experiments, we observe that
depth, width, weight decay, and Batch Normalization are important factors influencing calibration. We evaluate the performance of various
post-processing calibration methods on state-ofthe-art architectures with image and document
classification datasets. Our analysis and experiments not only offer insights into neural network learning, but also provide a simple and
straightforward recipe for practical settings: on
most datasets, temperature scaling – a singleparameter variant of Platt Scaling – is surprisingly effective at calibrating predictions.

 
 - **UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION**
    > Through extensive systematic experiments, we show how these traditional approaches fail to explain why large neural networks generalize well in practice.
Specifically, our experiments establish that state-of-the-art convolutional networks
for image classification trained with stochastic gradient methods easily fit a random labeling of the training data. This phenomenon is qualitatively unaffected
by explicit regularization, and occurs even if we replace the true images by completely unstructured random noise. We corroborate these experimental findings
with a theoretical construction showing that simple depth two neural networks already have perfect finite sample expressivity as soon as the number of parameters
exceeds the number of data points as it usually does in practice

- **Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time**
    > The conventional recipe for maximizing model
accuracy is to (1) train multiple models with various hyperparameters and (2) pick the individual model which performs best on a held-out
validation set, discarding the remainder. In this
paper, we revisit the second step of this procedure in the context of fine-tuning large pre-trained
models, where fine-tuned models often appear
to lie in a single low error basin. We show that
averaging the weights of multiple models finetuned with different hyperparameter configurations often improves accuracy and robustness. Unlike a conventional ensemble, we may average
many models without incurring any additional
inference or memory costs—we call the results
“model soups.” When fine-tuning large pre-trained
models such as CLIP, ALIGN, and a ViT-G pretrained on JFT, our soup recipe provides significant improvements over the best model in a hyperparameter sweep on ImageNet. The resulting ViT-G model, which attains 90.94% top-1
accuracy on ImageNet, achieved a new state of
the art. Furthermore, we show that the model
soup approach extends to multiple image classification and natural language processing tasks,
improves out-of-distribution performance, and improves zero-shot performance on new downstream
tasks. Finally, we analytically relate the performance similarity of weight-averaging and logitensembling to flatness of the loss and confidence
of the predictions, and validate this relation empirically

- **Certifying Some Distributional Robustness with Principled Adversarial Training**
    > Neural networks are vulnerable to adversarial examples and researchers have proposed many heuristic attack and defense mechanisms. We address this problem through the principled lens of distributionally robust optimization, which guarantees performance under adversarial input perturbations. By considering a Lagrangian penalty formulation of perturbing the underlying data distribution in a Wasserstein ball, we provide a training procedure that augments model parameter updates with worst-case perturbations of training data. For smooth losses, our procedure provably achieves moderate levels of robustness with little computational or statistical cost relative to empirical risk minimization. Furthermore, our statistical guarantees allow us to efficiently certify robustness for the population loss. For imperceptible perturbations, our method matches or outperforms heuristic approaches.

- **SOFT WEIGHT-SHARING FOR NEURAL NETWORK COMPRESSION**
    > In this paper, we show that competitive
compression rates can be achieved by using a version of ”soft weight-sharing”
(Nowlan & Hinton, 1992). Our method achieves both quantization and pruning in
one simple (re-)training procedure. This point of view also exposes the relation
between compression and the minimum description length (MDL) principle


- **Deep Leakage from Gradients**
    > we show that it is possible to obtain the
private training data from the publicly shared gradients. We name this leakage as
Deep Leakage from Gradient and empirically validate the effectiveness on both
computer vision and natural language processing tasks. Experimental results show
that our attack is much stronger than previous approaches: the recovery is pixelwise accurate for images and token-wise matching for texts. Thereby we want to
raise people’s awareness to rethink the gradient’s safety. We also discuss several
possible strategies to prevent such deep leakage. Without changes on training
setting, the most effective defense method is gradient pruning.

- **Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots**
    > In the field of natural language processing, the prevalent approach involves finetuning pretrained language models (PLMs) using local samples. Recent research
has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries
can embed malicious prediction behaviors by manipulating a few training samples.
In this study, our objective is to develop a backdoor-resistant tuning procedure that
yields a backdoor-free model, no matter whether the fine-tuning dataset contains
poisoned samples. To this end, we propose and integrate a honeypot module into
the original PLM, specifically designed to absorb backdoor information exclusively.
Our design is motivated by the observation that lower-layer representations in
PLMs carry sufficient backdoor features while carrying minimal information about
the original tasks. Consequently, we can impose penalties on the information
acquired by the honeypot module to inhibit backdoor creation during the finetuning process of the stem network. Comprehensive experiments conducted on
benchmark datasets substantiate the effectiveness and robustness of our defensive
strategy. Notably, these results indicate a substantial reduction in the attack success
rate ranging from 10% to 40% when compared to prior state-of-the-art methods


- **AUTODAN: GENERATING STEALTHY JAILBREAK PROMPTS ON ALIGNED LARGE LANGUAGE MODELS**
    >  Investigating jailbreak prompts can lead
us to delve into the limitations of LLMs and further guide us to secure them.
Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness
problems, as attacks depend on token-based algorithms to generate prompts that
are often semantically meaningless, making them susceptible to detection through
basic perplexity testing. In light of these challenges, we intend to answer this
question: Can we develop an approach that can automatically generate stealthy
jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive
evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength
in cross-model transferability, and cross-sample universality compared with the
baseline. Moreover, we also compare AutoDAN with perplexity-based defense
methods and show that AutoDAN can bypass them effectively

- **Reconstructive Neuron Pruning for Backdoor Defense**
    > In this paper, we propose a novel defense called Reconstructive Neuron Pruning (RNP) to expose
and prune backdoor neurons via an unlearning
and then recovering process. Specifically, RNP
first unlearns the neurons by maximizing the
model’s error on a small subset of clean samples and then recovers the neurons by minimizing the model’s error on the same data. In RNP,
unlearning is operated at the neuron level while
recovering is operated at the filter level, forming an asymmetric reconstructive learning procedure. We show that such an asymmetric process
on only a few clean samples can effectively expose and prune the backdoor neurons implanted
by a wide range of attacks, achieving a new stateof-the-art defense performance. Moreover, the
unlearned model at the intermediate step of our
RNP can be directly used to improve other backdoor defense tasks including backdoor removal,
trigger recovery, backdoor label detection, and
backdoor sample detection

- **DISTILLING COGNITIVE BACKDOOR PATTERNS WITHIN AN IMAGE**
    > This paper proposes a simple method to distill and detect backdoor patterns
within an image: Cognitive Distillation (CD). The idea is to extract the “minimal
essence” from an input image responsible for the model’s prediction. CD optimizes an input mask to extract a small pattern from the input image that can lead
to the same model output (i.e., logits or deep features). The extracted pattern can
help understand the cognitive mechanism of a model on clean vs. backdoor images and is thus called a Cognitive Pattern (CP). Using CD and the distilled CPs,
we uncover an interesting phenomenon of backdoor attacks: despite the various
forms and sizes of trigger patterns used by different attacks, the CPs of backdoor
samples are all surprisingly and suspiciously small. One thus can leverage the
learned mask to detect and remove backdoor examples from poisoned training
datasets. We conduct extensive experiments to show that CD can robustly detect a
wide range of advanced backdoor attacks. We also show that CD can potentially
be applied to help detect potential biases from face datasets

- **Universal and Transferable Adversarial Attacks on Aligned Language Models**
    > Because “out-of-the-box” large language models are capable of generating a great
deal of objectionable content, recent work has focused on aligning these models in an
attempt to prevent undesirable generation. While there has been some success at circumventing these measures—so-called “jailbreaks” against LLMs—these attacks have
required significant human ingenuity and are brittle in practice. Attempts at automatic
adversarial prompt generation have also achieved limited success. In this paper, we
propose a simple and effective attack method that causes aligned language models to
generate objectionable behaviors. Specifically, our approach finds a suffix that, when
attached to a wide range of queries for an LLM to produce objectionable content, aims
to maximize the probability that the model produces an affirmative response (rather
than refusing to answer). However, instead of relying on manual engineering, our approach automatically produces these adversarial suffixes by a combination of greedy
and gradient-based search techniques, and also improves over past automatic prompt
generation methods.
Surprisingly, we find that the adversarial prompts generated by our approach are
quite transferable, including to black-box, publicly released LLMs. Specifically, we train
an adversarial attack suffix on multiple prompts (i.e., queries asking for many different
types of objectionable content), as well as multiple models (in our case, Vicuna-7B and
13B). When doing so, the resulting attack suffix is able to induce objectionable content in the public interfaces to ChatGPT, Bard, and Claude, as well
as open source LLMs such as LLaMA-2-Chat, Pythia, Falcon, and others. Interestingly, the success rate of this attack transfer is much higher against the GPT-based
models, potentially owing to the fact that Vicuna itself is trained on outputs from
ChatGPT. In total, this work significantly advances the state-of-the-art in adversarial
attacks against aligned language models, raising important questions about how such
systems can be prevented from producing objectionable information


- **You Only Prompt Once: On the Capabilities of Prompt Learning on Large Language Models to Tackle Toxic Content**
    > y. In this work, we investigate how we
can use LLMs and prompt learning to tackle the problem of
toxic content, particularly focusing on three tasks; 1) Toxicity Classification, 2) Toxic Span Detection, and 3) Detoxification. We perform an extensive evaluation over five model
architectures and eight datasets demonstrating that LLMs
with prompt learning can achieve similar or even better performance compared to models trained on these specific tasks.
We find that prompt learning achieves around 10% improvement in the toxicity classification task compared to the baselines, while for the toxic span detection task we find better
performance to the best baseline (0.643 vs. 0.640 in terms
of F1-score). Finally, for the detoxification task, we find that
prompt learning can successfully reduce the average toxicity
score (from 0.775 to 0.213) while preserving semantic meaning.

- **SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models**
    > In this work, we propose "SelfCheckGPT", a
simple sampling-based approach that can be
used to fact-check the responses of black-box
models in a zero-resource fashion, i.e. without an external database. SelfCheckGPT leverages the simple idea that if an LLM has knowledge of a given concept, sampled responses
are likely to be similar and contain consistent
facts. However, for hallucinated facts, stochastically sampled responses are likely to diverge
and contradict one another. We investigate this
approach by using GPT-3 to generate passages
about individuals from the WikiBio dataset, and
manually annotate the factuality of the generated passages. We demonstrate that SelfCheckGPT can: i) detect non-factual and factual sentences; and ii) rank passages in terms of factuality. We compare our approach to several baselines and show that our approach has considerably higher AUC-PR scores in sentence-level
hallucination detection and higher correlation
scores in passage-level factuality assessment
compared to grey-box methods


- **SUPERVISED CONTRASTIVE LEARNING FOR PRE-TRAINED LANGUAGE MODEL FINE-TUNING**
    > State-of-the-art natural language understanding classification models follow twostages: pre-training a large language model on an auxiliary task, and then finetuning the model on a task-specific labeled dataset using cross-entropy loss. However, the cross-entropy loss has several shortcomings that can lead to sub-optimal
generalization and instability. Driven by the intuition that good generalization
requires capturing the similarity between examples in one class and contrasting
them with examples in other classes, we propose a supervised contrastive learning
(SCL) objective for the fine-tuning stage. Combined with cross-entropy, our proposed SCL loss obtains significant improvements over a strong RoBERTa-Large
baseline on multiple datasets of the GLUE benchmark in few-shot learning settings,
without requiring specialized architecture, data augmentations, memory banks, or
additional unsupervised data. Our proposed fine-tuning objective leads to models
that are more robust to different levels of noise in the fine-tuning training data, and
can generalize better to related tasks with limited labeled data.

- **Towards Building the Federated GPT: Federated Instruction Tuning**
    > . In the current paper, by conducting widely used GPT-4
auto-evaluation, we demonstrate that by exploiting the heterogeneous and diverse
sets of instructions on the client’s end with the proposed framework FedIT, we
improved the performance of LLMs compared to centralized training with only
limited local instructions. Further, in this paper, we developed a Github repository
named Shepherd. This repository offers a foundational framework for exploring
federated fine-tuning of LLMs using heterogeneous instructions across diverse
categories. The framework is designed for ease of use, adaptability, and scalability
to accommodate large datasets. Additionally, it facilitates the seamless integration
of novel algorithms and configurations, making it a convenient tool for researchers
and practitioners in the NLP community.


- **SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS**
    > Chain-of-thought prompting combined with pre-trained large language models has
achieved encouraging results on complex reasoning tasks. In this paper, we propose
a new decoding strategy, self-consistency, to replace the naive greedy decoding
used in chain-of-thought prompting. It first samples a diverse set of reasoning paths
instead of only taking the greedy one, and then selects the most consistent answer
by marginalizing out the sampled reasoning paths. Self-consistency leverages the
intuition that a complex reasoning problem typically admits multiple different ways
of thinking leading to its unique correct answer. Our extensive empirical evaluation
shows that self-consistency boosts the performance of chain-of-thought prompting
with a striking margin on a range of popular arithmetic and commonsense reasoning
benchmarks, including GSM8K (+17.9%), SVAMP (+11.0%), AQuA (+12.2%),
StrategyQA (+6.4%) and ARC-challenge (+3.9%).

- **Large Language Models as Tool Makers**
    > Recent research shows the potential of enhancing the problem-solving ability of
large language models (LLMs) through the use of external tools. However, prior
work along this line depends on the availability of existing tools. In this work, we
take an initial step towards removing this dependency by proposing a closed-loop
framework, referred to as LLMs As Tool Makers (LATM), where LLMs create
their own reusable tools for problem-solving. Our approach consists of two key
phases: 1) tool making: an LLM acts as the tool maker that crafts tools for given
tasks, where a tool is implemented as a Python utility function. 2) tool using:
an LLM acts as the tool user, which applies the tool built by the tool maker for
problem-solving. The tool user can be either the same or a different LLM from the
tool maker. Tool-making enables an LLM to continually generate tools that can be
applied to different requests so that future requests can call the corresponding APIs
when beneficial for solving the tasks. Furthermore, the division of labor among
LLMs for tool-making and tool-using phases introduces the opportunity to achieve
cost effectiveness without degrading the quality of generated tools and problem
solutions. For example, recognizing that tool-making demands more sophisticated
capabilities than tool-using, we can apply a powerful yet resource-intensive model
as the tool maker, and a lightweight while cost-effective model as the tool user. We
validate the effectiveness of our approach across a variety of complex reasoning
tasks, including Big-Bench tasks. With GPT-4 as the tool maker and GPT-3.5 as
the tool user, LATM can achieve performance that is on par with using GPT-4 for
both tool making and tool using, while the inference cost is significantly reduced.


- ** STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning**
    > Generating step-by-step "chain-of-thought" rationales improves language model
performance on complex reasoning tasks like mathematics or commonsense
question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy
by using only few-shot inference. We propose a technique to iteratively leverage a
small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning. This technique,
the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to
answer many questions, prompted with a few rationale examples; if the generated
answers are wrong, try again to generate a rationale given the correct answer; finetune on all the rationales that ultimately yielded correct answers; repeat. We show
that STaR significantly improves performance on multiple datasets compared to a
model fine-tuned to directly predict final answers, and performs comparably to finetuning a 30× larger state-of-the-art language model on CommensenseQA. Thus,
STaR lets a model improve itself by learning from its own generated reasoning

- ** Large Language Models are Zero-Shot Reasoners**
    > While these successes are often attributed to LLMs ability for few-shot learning, we show that LLMs are decent zero-shot reasoners
by simply adding “Let’s think step by step” before each answer. Experimental
results demonstrate that our Zero-shot-CoT, using the same single prompt template,
significantly outperforms zero-shot LLM performances on diverse benchmark
reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP),
symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date
Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot
examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and
GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci002), as well as similar magnitudes of improvements with another off-the-shelf
large model, 540B parameter PaLM. The versatility of this single prompt across
very diverse reasoning tasks hints at untapped and understudied fundamental
zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive
capabilities may be extracted by simple prompting. We hope our work not only
serves as the minimal strongest zero-shot baseline for the challenging reasoning
benchmarks, but also highlights the importance of carefully exploring and analyzing
the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning
datasets or few-shot exemplars

- **Jailbreaking Black Box Large Language Models in Twenty Queries**
    > There is growing interest in ensuring that large language models (LLMs) align with human values.
However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into
overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in
understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic
Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to
an LLM. PAIR—which is inspired by social engineering attacks—uses an attacker LLM to automatically
generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker
LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often
requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than
existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open
and closed-source LLMs, including GPT-3.5/4, Vicuna, and PaLM-2


- **Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**
    > While Large Language Models (LLMs) display versatile functionality, they continue to
generate harmful, biased, and toxic content, as demonstrated by the prevalence of humandesigned jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated
method for generating jailbreaks that only requires black-box access to the target LLM. TAP
utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning
until one of the generated prompts jailbreaks the target. Crucially, before sending prompts
to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using
tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning
reduces the total number of queries sent to the target. In empirical evaluations, we observe that
TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo)
for more than 80% of the prompts using only a small number of queries. This significantly
improves upon the previous state-of-the-art black-box method for generating jailbreaks.

-  **On Second Thought, Let’s Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning**
    > However, prior work has mainly
focused on logical reasoning tasks (e.g. arithmetic, commonsense QA); it remains unclear
whether improvements hold for more diverse
types of reasoning, especially in socially situated contexts. Concretely, we perform a controlled evaluation of zero-shot CoT across two
socially sensitive domains: harmful questions
and stereotype benchmarks. We find that zeroshot CoT reasoning in sensitive domains significantly increases a model’s likelihood to produce harmful or undesirable output, with trends
holding across different prompt formats and
model variants. Furthermore, we show that
harmful CoTs increase with model size, but
decrease with improved instruction following.
Our work suggests that zero-shot CoT should
be used with caution on socially important
tasks, especially when marginalized groups or
sensitive topics are involved

- **FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS**
    > This paper explores a simple method for improving the zero-shot learning abilities
of language models. We show that instruction tuning—finetuning language models
on a collection of datasets described via instructions—substantially improves zeroshot performance on unseen tasks.
We take a 137B parameter pretrained language model and instruction tune it on
over 60 NLP datasets verbalized via natural language instruction templates. We
evaluate this instruction-tuned model, which we call FLAN, on unseen task types.
FLAN substantially improves the performance of its unmodified counterpart and
surpasses zero-shot 175B GPT-3 on 20 of 25 datasets that we evaluate. FLAN even
outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC,
OpenbookQA, and StoryCloze. Ablation studies reveal that number of finetuning
datasets, model scale, and natural language instructions are key to the success of
instruction tuning.


- **Language Models are Few-Shot Learners**
    > Here we show that scaling up language models greatly improves task-agnostic,
few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its performance in
the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning,
with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3
achieves strong performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as
unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same
time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally,
we find that GPT-3 can generate samples of news articles which human evaluators have difficulty
distinguishing from articles written by humans. We discuss broader societal impacts of this finding
and of GPT-3 in general


- **Overcoming catastrophic forgetting in neural networks**
    > The ability to learn tasks in a sequential fashion is crucial to the development of
artificial intelligence. Neural networks are not, in general, capable of this and it
has been widely thought that catastrophic forgetting is an inevitable feature of
connectionist models. We show that it is possible to overcome this limitation and
train networks that can maintain expertise on tasks which they have not experienced
for a long time. Our approach remembers old tasks by selectively slowing down
learning on the weights important for those tasks. We demonstrate our approach is
scalable and effective by solving a set of classification tasks based on the MNIST
hand written digit dataset and by learning several Atari 2600 games sequentially


- **FINE-TUNING ALIGNED LANGUAGE MODELS COMPROMISES SAFETY, EVEN WHEN USERS DO NOT INTEND TO** 
    > Optimizing large language models (LLMs) for downstream use cases often involves the customization of pre-trained LLMs through further fine-tuning. Meta’s open release of Llama models and
OpenAI’s APIs for fine-tuning GPT-3.5 Turbo on custom datasets also encourage this practice. But,
what are the safety costs associated with such custom fine-tuning? We note that while existing
safety alignment infrastructures can restrict harmful behaviors of LLMs at inference time, they
do not cover safety risks when fine-tuning privileges are extended to end-users. Our red teaming
studies find that the safety alignment of LLMs can be compromised by fine-tuning with only a
few adversarially designed training examples. For instance, we jailbreak GPT-3.5 Turbo’s safety
guardrails by fine-tuning it on only 10 such examples at a cost of less than $0.20 via OpenAI’s APIs,
making the model responsive to nearly any harmful instructions. Disconcertingly, our research
also reveals that, even without malicious intent, simply fine-tuning with benign and commonly
used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser
extent. These findings suggest that fine-tuning aligned LLMs introduces new safety risks that
current safety infrastructures fall short of addressing — even if a model’s initial safety alignment
is impeccable, it is not necessarily to be maintained after custom fine-tuning1
. We outline and
critically analyze potential mitigations and advocate for further research efforts toward reinforcing
safety protocols for the custom fine-tuning of aligned LLMs


- **LOFTQ: LORA-FINE-TUNING-AWARE QUANTIZATION FOR LARGE LANGUAGE MODELS**
    > Quantization is an indispensable technique for serving Large Language Models
(LLMs) and has recently found its way into LoRA fine-tuning (Dettmers et al.,
2023). In this work we focus on the scenario where quantization and LoRA finetuning are applied together on a pre-trained model. In such cases it is common
to observe a consistent gap in the performance on downstream tasks between full
fine-tuning and quantization plus LoRA fine-tuning approach. In response, we
propose LoftQ (LoRA-Fine-Tuning-aware Quantization), a novel quantization
framework that simultaneously quantizes an LLM and finds a proper low-rank
initialization for LoRA fine-tuning. Such an initialization alleviates the discrepancy between the quantized and full-precision model and significantly improves
the generalization in downstream tasks. We evaluate our method on natural language understanding, question answering, summarization, and natural language
generation tasks. Experiments show that our method is highly effective and outperforms existing quantization methods, especially in the challenging 2-bit and
2/4-bit mixed precision regimes. We will release our code.


- **Exploiting Unintended Feature Leakage in Collaborative Learning**
    > Collaborative machine learning and related techniques such as
federated learning allow multiple participants, each with his
own training dataset, to build a joint model by training locally and periodically exchanging model updates. We demonstrate that these updates leak unintended information about
participants’ training data and develop passive and active inference attacks to exploit this leakage. First, we show that
an adversarial participant can infer the presence of exact data
points—for example, specific locations—in others’ training
data (i.e., membership inference). Then, we show how this
adversary can infer properties that hold only for a subset of
the training data and are independent of the properties that the
joint model aims to capture. For example, he can infer when a
specific person first appears in the photos used to train a binary
gender classifier. We evaluate our attacks on a variety of tasks,
datasets, and learning configurations, analyze their limitations,
and discuss possible defenses.


- **From the Detection of Toxic Spans in Online Discussions to the Analysis of Toxic-to-Civil Transfer**
    > We study the task of toxic spans detection,
which concerns the detection of the spans that
make a text toxic, when detecting such spans
is possible. We introduce a dataset for this
task, TOXICSPANS, which we release publicly.
By experimenting with several methods, we
show that sequence labeling models perform
best. Moreover, methods that add generic rationale extraction mechanisms on top of classifiers
trained to predict if a post is toxic or not are
also surprisingly promising. Finally, we use
TOXICSPANS and systems trained on it, to provide further analysis of state-of-the-art toxic to
non-toxic transfer systems, as well as of human
performance on that latter task. Our work highlights challenges in finer toxicity detection and
mitigation


- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
    > Evaluating large language model (LLM) based chat assistants is challenging due to
their broad capabilities and the inadequacy of existing benchmarks in measuring
human preferences. To address this, we explore using strong LLMs as judges to
evaluate these models on more open-ended questions. We examine the usage and
limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement
biases, as well as limited reasoning ability, and propose solutions to mitigate some
of them. We then verify the agreement between LLM judges and human preferences
by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot
Arena, a crowdsourced battle platform. Our results reveal that strong LLM judges
like GPT-4 can match both controlled and crowdsourced human preferences well,
achieving over 80% agreement, the same level of agreement between humans.
Hence, LLM-as-a-judge is a scalable and explainable way to approximate human
preferences, which are otherwise very expensive to obtain. Additionally, we show
our benchmark and traditional benchmarks complement each other by evaluating
several variants of LLaMA and Vicuna. The MT-bench questions, 3K expert votes,
and 30K conversations with human preferences are publicly available

- **EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks**
    > We present EDA: easy data augmentation
techniques for boosting performance on text
classification tasks. EDA consists of four simple but powerful operations: synonym replacement, random insertion, random swap, and
random deletion. On five text classification
tasks, we show that EDA improves performance for both convolutional and recurrent
neural networks. EDA demonstrates particularly strong results for smaller datasets; on average, across five datasets, training with EDA
while using only 50% of the available training set achieved the same accuracy as normal
training with all available data. We also performed extensive ablation studies and suggest
parameters for practical use


- **SimCSE: Simple Contrastive Learning of Sentence Embeddings**
    > This paper presents SimCSE, a simple contrastive learning framework that greatly advances the state-of-the-art sentence embeddings. We first describe an unsupervised approach, which takes an input sentence and
predicts itself in a contrastive objective, with
only standard dropout used as noise. This
simple method works surprisingly well, performing on par with previous supervised counterparts. We find that dropout acts as minimal data augmentation and removing it leads
to a representation collapse. Then, we propose a supervised approach, which incorporates annotated pairs from natural language
inference datasets into our contrastive learning framework, by using “entailment” pairs
as positives and “contradiction” pairs as hard
negatives. We evaluate SimCSE on standard
semantic textual similarity (STS) tasks, and
our unsupervised and supervised models using
BERTbase achieve an average of 76.3% and
81.6% Spearman’s correlation respectively, a
4.2% and 2.2% improvement compared to
previous best results. We also show—both
theoretically and empirically—that contrastive
learning objective regularizes pre-trained embeddings’ anisotropic space to be more uniform, and it better aligns positive pairs when
supervised signals are available

- **Large Language Models Can Be Easily Distracted by Irrelevant Context**  
    > Large language models have achieved impressive
performance on various natural language processing tasks. However, so far they have been evaluated primarily on benchmarks where all information in the input context is relevant for solving
the task. In this work, we investigate the distractibility of large language models, i.e., how
the model problem-solving accuracy can be influenced by irrelevant context. In particular, we introduce Grade-School Math with Irrelevant Context
(GSM-IC), an arithmetic reasoning dataset with
irrelevant information in the problem description.
We use this benchmark to measure the distractibility of cutting-edge prompting techniques for large
language models, and find that the model performance is dramatically decreased when irrelevant
information is included. We also identify several
approaches for mitigating this deficiency, such
as decoding with self-consistency and adding to
the prompt an instruction that tells the language
model to ignore the irrelevant information


- **EDITING MODELS WITH TASK ARITHMETIC**   
    > In this work, we propose a
new paradigm for steering the behavior of neural networks, centered around task
vectors. A task vector specifies a direction in the weight space of a pre-trained
model, such that movement in that direction improves performance on the task.
We build task vectors by subtracting the weights of a pre-trained model from the
weights of the same model after fine-tuning on a task. We show that these task
vectors can be modified and combined together through arithmetic operations
such as negation and addition, and the behavior of the resulting model is steered
accordingly. Negating a task vector decreases performance on the target task, with
little change in model behavior on control tasks. Moreover, adding task vectors
together can improve performance on multiple tasks at once. Finally, when tasks are
linked by an analogy relationship of the form “A is to B as C is to D”, combining
task vectors from three of the tasks can improve performance on the fourth, even
when no data from the fourth task is used for training. Overall, our experiments
with several models, modalities and tasks show that task arithmetic is a simple,
efficient and effective way of editing models

- **Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models**
    > Task arithmetic has recently emerged as a cost-effective and scalable approach to
edit pre-trained models directly in weight space: By adding the fine-tuned weights
of different tasks, the model’s performance can be improved on these tasks, while
negating them leads to task forgetting. Yet, our understanding of the effectiveness
of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight
disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show
that fine-tuning models in their tangent space by linearizing them amplifies weight
disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we
provide theoretical and empirical analyses of the neural tangent kernel (NTK) of
these models and establish a compelling link between task arithmetic and the spatial
localization of the NTK eigenfunctions. Overall, our work uncovers novel insights
into the fundamental mechanisms of task arithmetic and offers a more reliable and
effective approach to edit pre-trained models through the NTK linearization.


- **Why think step by step? Reasoning emerges from the locality of experience**
    > Humans have a powerful and mysterious capacity to reason. Working through a set of mental steps enables us to make inferences we would not be capable of making directly even though we get no additional data from the world. Similarly, when large language models generate intermediate steps (a chain of thought) before answering a question, they often produce better answers than they would directly. We investigate why and how chain-of-thought reasoning is useful in language models, testing the hypothesis that reasoning is effective when training data consists of overlapping local clusters of variables that influence each other strongly. These training conditions enable the chaining of accurate local inferences to estimate relationships between variables that were not seen together in training. We prove that there will exist a "reasoning gap", where reasoning through intermediate variables reduces bias, for the simple case of an autoregressive density estimator trained on local samples from a chain-structured probabilistic model. We then test our hypothesis experimentally in more complex models, training an autoregressive language model on samples from Bayes nets but only including a subset of variables in each sample. We test language models' ability to match conditional probabilities with and without intermediate reasoning steps, finding that intermediate steps are only helpful when the training data is locally structured with respect to dependencies between variables. The combination of locally structured observations and reasoning is much more data-efficient than training on all variables. Our results illustrate how the effectiveness of reasoning step by step is rooted in the local statistical structure of the training data.

- **Toolformer: Language Models Can Teach Themselves to Use Tools**
    > Language models (LMs) exhibit remarkable abilities to solve new tasks from just a
few examples or textual instructions, especially at scale. They also, paradoxically,
struggle with basic functionality, such as arithmetic or factual lookup, where much
simpler and smaller specialized models excel. In this paper, we show that LMs
can teach themselves to use external tools via simple APIs and achieve the best of
both worlds. We introduce Toolformer, a model trained to decide which APIs to
call, when to call them, what arguments to pass, and how to best incorporate the
results into future token prediction. This is done in a self-supervised way, requiring
nothing more than a handful of demonstrations for each API. We incorporate a
range of tools, including a calculator, a Q&A system, a search engine, a translation
system, and a calendar. Toolformer achieves substantially improved zero-shot
performance across a variety of downstream tasks, often competitive with much
larger models, without sacrificing its core language modeling abilities


- **Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks**
    > The field of defense strategies against adversarial
attacks has significantly grown over the last years,
but progress is hampered as the evaluation of adversarial defenses is often insufficient and thus
gives a wrong impression of robustness. Many
promising defenses could be broken later on, making it difficult to identify the state-of-the-art. Frequent pitfalls in the evaluation are improper tuning of hyperparameters of the attacks, gradient
obfuscation or masking. In this paper we first
propose two extensions of the PGD-attack overcoming failures due to suboptimal step size and
problems of the objective function. We then combine our novel attacks with two complementary
existing ones to form a parameter-free, computationally affordable and user-independent ensemble of attacks to test adversarial robustness. We
apply our ensemble to over 50 models from papers published at recent top machine learning and
computer vision venues. In all except one of the
cases we achieve lower robust test accuracy than
reported in these papers, often by more than 10%,
identifying several broken defenses.

- **DOLA: DECODING BY CONTRASTING LAYERS IMPROVES FACTUALITY IN LARGE LANGUAGE MODELS**
    > Despite their impressive capabilities, large language models (LLMs) are prone to hallucinations, i.e., generating content that deviates from facts seen during pretraining. We
propose a simple decoding strategy for reducing hallucinations with pretrained LLMs
that does not require conditioning on retrieved external knowledge nor additional finetuning. Our approach obtains the next-token distribution by contrasting the differences
in logits obtained from projecting the later layers versus earlier layers to the vocabulary
space, exploiting the fact that factual knowledge in an LLMs has generally been shown to
be localized to particular transformer layers. We find that this Decoding by Contrasting
Layers (DoLa) approach is able to better surface factual knowledge and reduce the generation of incorrect facts. DoLa consistently improves the truthfulness across multiple
choices tasks and open-ended generation tasks, for example improving the performance
of LLaMA family models on TruthfulQA by 12-17% absolute points, demonstrating its
potential in making LLMs reliably generate truthful facts

-  **COMPRESSING LLMS: THE TRUTH IS RARELY PURE AND NEVER SIMPLE**
    > We introduce Knowledge-Intensive Compressed LLM BenchmarK
(LLM-KICK), a collection of carefully-curated tasks to re-define the evaluation
protocol for compressed LLMs, which have significant alignment with their dense
counterparts, and perplexity fail to capture subtle change in their true capabilities. LLM-KICK unveils many favorable merits and unfortunate plights of current SoTA compression methods: all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%), and fail
for N:M sparsity on knowledge-intensive tasks; current quantization methods are
more successful than pruning; yet, pruned LLMs even at ≥ 50% sparsity are robust in-context retrieval and summarization systems; among others. LLM-KICK
is designed to holistically access compressed LLMs’ ability for language understanding, reasoning, generation, in-context retrieval, in-context summarization,
etc 


- **BEYOND MEMORIZATION: VIOLATING PRIVACY VIA INFERENCE WITH LARGE LANGUAGE MODELS**
    > e. In this work, we present the first comprehensive study on the capabilities of pretrained LLMs to infer personal attributes
from text. We construct a dataset consisting of real Reddit profiles, and show that
current LLMs can infer a wide range of personal attributes (e.g., location, income,
sex), achieving up to 85% top-1 and 95.8% top-3 accuracy at a fraction of the cost
(100×) and time (240×) required by humans. As people increasingly interact with
LLM-powered chatbots across all aspects of life, we also explore the emerging
threat of privacy-invasive chatbots trying to extract personal information through
seemingly benign questions. Finally, we show that common mitigations, i.e., text
anonymization and model alignment, are currently ineffective at protecting user
privacy against LLM inference. Our findings highlight that current LLMs can infer personal data at a previously unattainable scale. In the absence of working
defenses, we advocate for a broader discussion around LLM privacy implications
beyond memorization, striving for a wider privacy protection


- **DO LARGE LANGUAGE MODELS KNOW ABOUT FACTS?**
    > Unlike conventional Knowledge Bases (KBs) that explicitly store factual knowledge, LLMs implicitly store facts in their parameters. Content generated by the
LLMs can often exhibit inaccuracies or deviations from the truth, due to facts that
can be incorrectly induced or become obsolete over time. To this end, we aim to
comprehensively evaluate the extent and scope of factual knowledge within LLMs
by designing the benchmark Pinocchio. Pinocchio contains 20K diverse factual
questions that span different sources, timelines, domains, regions, and languages.
Furthermore, we investigate whether LLMs are able to compose multiple facts,
update factual knowledge temporally, reason over multiple pieces of facts, identify
subtle factual differences, and resist adversarial examples. Extensive experiments
on different sizes and types of LLMs show that existing LLMs still lack factual
knowledge and suffer from various spurious correlations. We believe this is a
critical bottleneck for realizing trustworthy artificial intelligence. The dataset
Pinocchio and our codes will be publicly available.

- **JAILBREAK IN PIECES: COMPOSITIONAL ADVERSARIAL ATTACKS ON MULTI-MODAL LANGUAGE MODELS**
    > We introduce new jailbreak attacks on vision language models (VLMs), which
use aligned LLMs and are resilient to text-only jailbreak attacks. Specifically, we
develop cross-modality attacks on alignment where we pair adversarial images
going through the vision encoder with textual prompts to break the alignment of the
language model. Our attacks employ a novel compositional strategy that combines
an image, adversarially targeted towards toxic embeddings, with generic prompts
to accomplish the jailbreak. Thus, the LLM draws the context to answer the generic
prompt from the adversarial image. The generation of benign-appearing adversarial
images leverages a novel embedding-space-based methodology, operating with no
access to the LLM model. Instead, the attacks require access only to the vision
encoder and utilize one of our four embedding space targeting strategies. By
not requiring access to the LLM, the attacks lower the entry barrier for attackers,
particularly when vision encoders such as CLIP are embedded in closed-source
LLMs. The attacks achieve a high success rate across different VLMs, highlighting
the risk of cross-modality alignment vulnerabilities, and the need for new alignment
approaches for multi-modal models.


- **CAN SENSITIVE INFORMATION BE DELETED FROM LLMS? OBJECTIVES FOR DEFENDING AGAINST EXTRACTION ATTACKS**
    > Pretrained language models sometimes possess knowledge that we do not wish
them to, including memorized personal information and knowledge that could be
used to harm people. They can also output toxic or harmful text. To mitigate these
safety and informational issues, we propose an attack-and-defense framework for
studying the task of deleting sensitive information directly from model weights.
We study direct edits to model weights because (1) this approach should guarantee
that particular deleted information is never extracted by future prompt attacks, and
(2) it should protect against whitebox attacks, which is necessary for making claims
about safety/privacy in a setting where publicly available model weights could
be used to elicit sensitive information. Our threat model assumes that an attack
succeeds if the answer to a sensitive question is located among a set of B generated
candidates, based on scenarios where the information would be insecure if the
answer is among B candidates. Experimentally, we show that even state-of-the-art
model editing methods such as ROME struggle to truly delete factual information
from models like GPT-J, as our whitebox and blackbox attacks can recover “deleted”
information from an edited model 38% of the time. These attacks leverage two key
observations: (1) that traces of deleted information can be found in intermediate
model hidden states, and (2) that applying an editing method for one question
may not delete information across rephrased versions of the question. Finally,
we provide new defense methods that protect against some extraction attacks, but
we do not find a single universally effective defense method. Our results suggest
that truly deleting sensitive information is a tractable but difficult problem, since
even relatively low attack success rates have potentially severe implications for the
deployment of language models in a world where individuals enjoy ownership of
their personal data, a right to privacy, and safety from harmful model outputs