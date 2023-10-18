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