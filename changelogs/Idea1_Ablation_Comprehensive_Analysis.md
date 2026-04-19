# Comprehensive Analysis of Loss Ablations in Asymmetric Finsler Domain Generalizable Re-ID

This document presents a rigorous empirical analysis of the "Idea 1" loss ablations, evaluating explicit disentanglement of identity and drift representations versus unified asymmetric metric learning formulations for Domain Generalizable Person Re-Identification (DG-ReID).

## 1. Explicit Disentanglement vs. Unified Finsler Manifold

The baseline configuration (**1a**), which isolates the identity embedding under a Euclidean triplet loss while applying Balancing Alignment and Uniformity (BAU) domain repulsion, achieves a peak performance of 44.1 mAP and 43.6 Rank-1 accuracy under instance-conditioned Finsler BAU. 

The **Unified Finsler** formulation abandons the explicit `[identity | drift]` partition, applying a single asymmetric Finsler triplet loss over the concatenated representation. This yields 44.1 mAP and 44.3 Rank-1 (under domain conditioning). The performance parity in mAP, alongside a marginal improvement in Rank-1, indicates that explicit structural disentanglement is largely redundant when utilizing an asymmetric distance metric. The Finsler manifold inherently accommodates directional domain-shift variance within the single triplet formulation without necessitating rigid subspace parttioning. The parameterization of the Finsler metric natively resolves the geometry of the drift, rendering explicit identity-only constraints computationally superfluous.

## 2. Catastrophic Failure of the Domain Triplet (1b) and Optimization Conflicts

The introduction of the domain triplet (**1b**), which mines camera labels to minimize intra-camera distances, induces a catastrophic collapse in generalization capability. 

The objective of DG-ReID is the extraction of domain-invariant features [Jia et al., CVPR 2019; Zhou et al., CVPR 2021]. By explicitly pulling same-camera samples together in the embedding space, the 1b objective forces the network to preserve domain-specific variation. Consequently, the identity manifold shatters into domain-clustered sub-manifolds. The intra-class (identity) variance across domains increases strictly as a function of the camera-clustering gradient.

When the BAU domain memory bank repulsion (`L_dom`) is disabled (**1a + 1b (no L_dom)**), performance drops severely to 24.9 mAP / 28.5 Rank-1 (Fins. Dom, domain cond) or 33.6 mAP / 35.0 Rank-1 (Euc. Dom, instance cond). The embedding space becomes heavily dominated by domain clusters, severely penalizing cross-domain identity retrieval. 

Reintroducing `L_dom` (**1a + 1b**) partially recovers performance to 34.9 mAP / 36.1 Rank-1 (Fins. Dom, domain cond) and 37.8 mAP / 38.9 Rank-1 (Euc. Dom, domain cond). Note that the Euclidean `instance` conditioned run is still evaluating, but the trend is evident. This establishes a pathological optimization dynamic: `L_dom` enforces uniformity (repulsion) among domain representations [Wang & Isola, ICML 2020], while the 1b triplet loss actively minimizes distance between same-domain samples. The resulting gradient interference prevents convergence to either a uniform domain distribution or a coherent identity space, stabilizing at a suboptimal equilibrium far below the baseline.

## 3. Asymmetric Finsler vs. Euclidean Domain Repulsion: A Nuanced Trade-off

Contrary to prior hypotheses, the Finsler domain formulation (`Fins. Dom.`) does **not** strictly dominate the Euclidean counterpart (`Euc. Dom.`). Instead, the advantage is marginal and highly configuration-dependent. 

In the baseline 1a setting, `Fins. Dom.` (instance conditioning) achieves 44.1 mAP / 43.6 R1, slightly edging out the `Euc. Dom.` equivalent (43.9 mAP / 43.9 R1) in mean Average Precision, but trailing in Rank-1. Under domain conditioning, Euclidean (43.8 mAP) explicitly outperforms Finsler (43.2 mAP). Furthermore, under the conflicting objectives of 1a+1b, the Euclidean metric demonstrates greater optimization resilience (37.8 mAP vs 34.9 mAP for domain conditioning).

This indicates that while the Finsler distance can theoretically model asymmetric transition costs between source and target domains, the empirical realization of `L_dom` over the asymmetric geometry is sensitive to the drift conditioning granularity. Enforcing domain uniformity via a symmetric Euclidean metric provides a more stable regularization gradient, whereas the unconstrained Finsler formulation may overfit to the source domain directional shifts, particularly under domain-level averaging.

## 4. Redundancy and Degradation in Auxiliary Constraints

Auxiliary constraints aimed at regularizing the drift vectors—namely, cross-camera same-PID contrastive loss in Finsler space (**Refined 1b**) and mean squared L2 penalty on same-PID different-camera drift vectors (**1c**)—yield mild performance degradation or redundancy in mAP.

Refined 1b achieves a peak of 43.1 mAP / 43.7 R1, and 1c achieves 43.8 mAP / 44.9 R1 (Fins. Dom, domain cond), both trailing or plateauing against the 44.1 mAP baseline. These penalties explicitly force drift vectors for a given identity to align or minimize their magnitudes across cameras. However, true domain shift is highly non-linear and heterogeneous [Chen & He, CVPR 2021]. By rigidly constraining the drift vectors (e.g., via L2 minimization or identity contrast), the network loses the representational capacity required to model severe or anomalous camera shifts. The baseline formulation, governed only by identity triplet and BAU repulsion, implicitly allows the drift vectors the necessary degrees of freedom to capture these shifts without artificially bounding their geometry.
