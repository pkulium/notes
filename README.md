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
    > We propose a distillation framework for robust federated model fusion, which allows for heterogeneous client models and data, and is robust to the choices of neural architectures. We show in extensive numerical experiments on various CV/NLP datasets (CIFAR-10/100, ImageNet, AG News, SST2) and settings (heterogeneous models and/or data) that the server model
can be trained much faster, requiring fewer communication rounds than any existing FL technique.

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
    > 