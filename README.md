- **Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach**
    >We considered the Federated Learning (FL) problem in the heterogeneous case, and studied a
personalized variant of the classic FL formulation in which our goal is to find a proper initialization
model for the users that can be quickly adapted to the local data of each user after the training phase.
We highlighted the connections of this formulation with Model-Agnostic Meta-Learning (MAML),
and showed how the decentralized implementation of MAML, which we called Per-FedAvg, can be
used to solve the proposed personalized FL problem. We also characterized the overall complexity of
Per-FedAvg for achieving first-order optimality in nonconvex settings. Finally, we provided a set of
numerical experiments to illustrate the performance of two different first-order approximations of
Per-FedAvg and their comparison with the FedAvg method, and showed that the solution obtained by
Per-FedAvg leads to a more personalized solution compared to the solution of FedAvg

- **Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization**
    > This paper provides a general framework to analyze the convergence of heterogeneous federated optimization algorithms. It subsumes previously proposed methods such as FedAvg and FedProx, and provides the first principled understanding of the solution bias and the convergence slowdown due to objective
inconsistency. Using insights from this analysis, we propose FedNova, a normalized averaging method that eliminates objective inconsistency while preserving
fast error convergence

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


