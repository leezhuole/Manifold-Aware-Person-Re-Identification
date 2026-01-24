# Project Description
Standard Person Re-Identification (ReID) models operate under the assumption that visual feature spaces are inherently Euclidean. This project challenges that "flat" assumption, addressing the "crowding problem" where traditional models fail to preserve the latent hierarchical structure of human appearance data (e.g., Body $\rightarrow$ Part $\rightarrow$ Attribute), leading to performance degradation under occlusion and high uncertainty.

Our primary objective is to transition ReID architectures to Geometric Deep Learning frameworks utilizing Hyperbolic embeddings. leveraging manifolds with constant negative curvature (specifically Poincaré ball and Lorentz models), we aim to naturally embed these hierarchies with minimal distortion. We will develop Dual-Space architectures that synergize the stability of Euclidean space for texture representation with the structural capacity of hyperbolic space for identity discrimination. This includes the implementation of Hyperbolic Vision Transformers and geometric loss functions (e.g., CHEST) to maximize discriminative margins between similar identities.

Crucially, this project extends beyond symmetric Riemannian geometry to investigate asymmetric manifolds. Recognizing that the matching process in surveillance is often direction-dependent (e.g., retrieving a high-resolution gallery image from a low-resolution probe creates an asymmetric information flow), we will model these relationships using Finsler manifolds, specifically employing the Randers metric. By adopting a metric that accommodates directionality, we aim to capture the intrinsic asymmetry of cross-view retrieval and camera network topology. This novel integration of hyperbolic hierarchy and Finsler asymmetry seeks to establish a mathematically rigorous, topologically faithful foundation for the next generation of robust surveillance systems.

## 1. High-Level Experimental Objectives

This document outlines a three-stage experimental plan to integrate hyperbolic geometry into the BAU Domain-Generalizable (DG) Person Re-ID framework. The central hypothesis is that the hyperbolic distance metric, by better modeling the latent hierarchical structure of identity features, will improve the efficacy of BAU's core alignment and uniformity losses, leading to enhanced domain generalization. This hypothesis is supported by literature indicating that hyperbolic spaces intrinsically facilitate feature uniformity, the primary mechanism of the BAU model.   

All modifications will leverage the `geoopt` library for Riemannian optimization and manifold operations. The baseline repository is `yoonkicho/BAU`.   

**Dependencies to add to `requirements.txt`:**

# For Riemannian optimization and hyperbolic geometry
# Per , installation from the git repository is preferred
git+https://github.com/geoopt/geoopt.git

## 2. Experiment 1 (Low Complexity): Retrofitting a Hyperbolic Classification Head
**Goal:** Establish a baseline integration by adding a hyperbolic classifier on top of the existing _Euclidean_ BAU embeddings, inspired by Khrulkov et al.. This tests the suitability of hyperbolic classification without altering the core BAU metric learning logic.   

**Step-by-Step Implementation:**

1. **Define Manifold (Global):**
    
    - In `train.py` (or `main.py`), instantiate the manifold:
            
        ```python
        import geoopt
        # c=1.0 is standard, can be tuned as a hyperparameter
        manifold = geoopt.PoincareBall(c=1.0) 
        ```
        
2. **Define Hyperbolic Head:**
    
    - Create a new file: `bau/hyperbolic_layers.py`.
        
    - Implement a `HyperbolicMLR` class based on Ganea et al.. This layer's parameters _must_ be `geoopt.ManifoldParameter`.   
        
        ```python
        import torch
        import torch.nn as nn
        import geoopt
        from geoopt.manifolds.stereographic.math import dist as poincare_dist
        
        class HyperbolicMLR(nn.Module):
            """
            Simplified Hyperbolic Logistic Regression head based on.
            This implementation uses hyperbolic distance to class prototypes.
            """
            def __init__(self, in_features, out_features, manifold):
                super().__init__()
                self.manifold = manifold
                self.in_features = in_features
                self.out_features = out_features
        
                # Class prototypes (weights) are points on the manifold
                self.prototypes = geoopt.ManifoldParameter(
                    self.manifold.random_normal(out_features, in_features, std=1e-3), 
                    manifold=self.manifold
                )
        
                # Biases are standard Euclidean parameters
                self.bias = nn.Parameter(torch.zeros(out_features))
        
            def forward(self, x_euclidean):
                # x is a Euclidean embedding, map it to the ball
                # Project to tangent space and map to manifold
                x_tan = self.manifold.project(x_euclidean)
                x_hyp = self.manifold.expmap0(x_tan)
        
                # Wrap for manifold operations
                x_hyp = geoopt.ManifoldTensor(x_hyp, manifold=self.manifold)
        
                # Compute hyperbolic distances to class prototypes
                #
                x_hyp_expanded = x_hyp.unsqueeze(1)
                #
                prototypes_expanded = self.prototypes.unsqueeze(0)
        
                # dist() returns Euclidean distance, use manifold.dist()
                #
                dist_sq = self.manifold.dist2(x_hyp_expanded, prototypes_expanded)
        
                # Logits are negative squared distance, per 
                logits = -dist_sq + self.bias
                return logits
        ```
        
3. **Modify Model (`bau/model.py`):**
    
    - Import `HyperbolicMLR`.
        
    - In your main model class (`BAUModel`), initialize this head. Pass the `manifold` and `num_classes` during `__init__`.
        
        ```python
        from.hyperbolic_layers import HyperbolicMLR
        # In BAUModel.__init__
        self.hyperbolic_head = HyperbolicMLR(
            in_features=embedding_dim, 
            out_features=num_classes, 
            manifold=manifold
        )
        ```
        
    - The `forward` pass must now return _both_ the standard BAU embedding and the logits from this new head.
                
        ```python
        # In BAUModel.forward
        euclidean_embedding = self.backbone(x) # Assuming this is the Euclidean feature
        hyperbolic_logits = self.hyperbolic_head(euclidean_embedding)
        # Return both for the two separate losses
        return euclidean_embedding, hyperbolic_logits
        ```
        
4. **Modify Optimizer Setup (`train.py`):**
    
    - Set up _two_ optimizers. The main `Adam` optimizer for the backbone, and `RiemannianAdam` for the new head's `ManifoldParameter`s.   
                
        ```python
        # Collect backbone/Euclidean parameters
        base_params = [p for n, p in model.named_parameters() 
                       if 'hyperbolic_head' not in n]
        
        # Collect hyperbolic head parameters
        hyp_params = model.hyperbolic_head.parameters()
        
        optimizer_bau = torch.optim.Adam(base_params, lr=...)
        optimizer_hyp = geoopt.optim.RiemannianAdam(hyp_params, lr=...)
        ```
        
5. **Modify Loss and Training Loop (`train.py`):**
    
    - Instantiate a standard CE loss: `criterion_ce = nn.CrossEntropyLoss()`
        
    - In the training loop, compute the total loss:
                
        ```python
        # Forward pass
        euclidean_embedding, hyperbolic_logits = model(images)
        
        # Standard BAU loss on Euclidean embeddings
        loss_bau = compute_bau_loss(euclidean_embedding,...) # Existing loss
        
        # CE loss on hyperbolic logits
        loss_hyp = criterion_ce(hyperbolic_logits, labels)
        
        total_loss = loss_bau + (lambda_hyp * loss_hyp) # Add a weighting hyperparam
        
        # Backpropagation
        optimizer_bau.zero_grad()
        optimizer_hyp.zero_grad()
        total_loss.backward()
        optimizer_bau.step()
        optimizer_hyp.step()
        ```
        

**Analysis:**

- Train and evaluate. Compare the DG Re-ID performance (mAP, Rank-1) against the baseline `BAU`.
    
- Hypothesis: This may slightly improve performance by adding a regularizing loss, but the core metric learning (alignment/uniformity) remains Euclidean.
    

## 3. Experiment 2 (Medium Complexity): Full Hyperbolic Metric and Loss Substitution

**Goal:** This is the primary experiment. We will re-define the BAU loss functions (Lalign​, Lunif​) to operate natively in hyperbolic space, using the hyperbolic geodesic distance.   

**Step-by-Step Implementation:**

1. **Define Manifold and Pass to Modules:**
    
    - In `train.py`, instantiate the manifold.
        
        ```python
        import geoopt
        manifold = geoopt.PoincareBall(c=1.0) # c can be a hyperparameter
        ```
        
    - Pass this `manifold` object to the model and the loss function constructors.

        ```python 
        model = BAUModel(..., manifold=manifold)
        bau_loss_func = BAULoss(..., manifold=manifold)
        ```
        
2. **Modify Model (`bau/model.py`):**
    
    - The model's `forward` pass must now output embeddings that are _on_ the Poincaré ball.
        
    - The model parameters _remain Euclidean_ (`torch.nn.Parameter`).
        
    - In the `__init__`: `self.manifold = manifold`
        
    - Modify the `forward` method:
                
        ```python
        # In BAUModel.forward
        x_euclidean = self.backbone(images) # Euclidean features
        
        # Map Euclidean embedding to Poincare ball
        # 1. Project to tangent space at origin
        x_tan = self.manifold.project(x_euclidean, dim=-1)
        # 2. Map from tangent space to manifold
        x_hyp = self.manifold.expmap0(x_tan)
        
        # 3. Wrap in ManifoldTensor for geoopt loss functions 
        x_hyp = geoopt.ManifoldTensor(x_hyp, manifold=self.manifold)
        
        # 4. (Optional but recommended) Project to ensure numerical stability
        x_hyp.proj_()
        
        return x_hyp
        ```
        
3. **Modify Optimizer (`train.py`):**
    
    - **Crucial Insight:** Because all model parameters are still `torch.nn.Parameter` (Euclidean), we _do not_ use `RiemannianAdam`. The optimization is "Case 2" as defined in Part II.
        
    - The optimizer remains the standard `torch.optim.Adam` from the baseline `BAU` code. `optimizer = torch.optim.Adam(model.parameters(), lr=...)`
        
    - The gradient will be computed from hyperbolic distances but backpropagated as a standard Euclidean gradient.
        
4. **Modify Loss Function (`bau/loss.py`):**
    
    - Modify the `BAULoss` class to accept `manifold` in `__init__`.
                
        ```python
        class BAULoss(nn.Module):
            def __init__(self,..., manifold):
                super().__init__()
                self.manifold = manifold
               ...
        ```
        
    - **Critically:** Find all calculations of alignment and uniformity.
        
    - Anywhere a Euclidean distance (e.g., `torch.pdist`, `(x-y).pow(2).sum()`) is computed, **replace it with `self.manifold.dist2(x, y)`** (for squared distance, per ).   
        
    - **Example (Alignment Loss):**
        
        - _Original (Euclidean):_ 
        ```python
        # features1, features2 are torch Tensors 
        dist_sq = torch.sum((features1 - features2).pow(2), dim=1) 
        loss_align = dist_sq.mean()
        ```
            
        - _New (Hyperbolic):_ 
        ```python
        # features1, features2 are ManifoldTensors
        dist_sq = self.manifold.dist2(features1, features2, keepdim=False) 
        loss_align = dist_sq.mean()
        ```    
        
    - **Example (Uniformity Loss):**
        
        - _Original (Euclidean):_ 
        ```python
        # features are tensors
        sq_dist_flat = torch.pdist(features, p=2).pow(2)
        loss_unif = sq_dist_flat.mul(-t).exp().mean().log()
        ```
        - _New (Hyperbolic):_ 
        ```python
        # features are ManifoldTensors
        # We must compute the pairwise hyperbolic distance matrix vs -> [N, N]
        sq_dist_matrix = self.manifold.dist2(features.unsqueeze(1), features.unsqueeze(0))
        # We need the unique pairs (upper triangle)
        n = features.shape
        mask = torch.triu(torch.ones(n, n, device=features.device), diagonal=1).bool()
        sq_dist_flat = sq_dist_matrix[mask]
        loss_unif = sq_dist_flat.mul(-t).exp().mean().log()
        ```    

**Analysis:**

- Train the model using the standard `train.py` loop.
    
- This directly tests the core hypothesis. The `c` (curvature) parameter of `PoincareBall` is a new, critical hyperparameter to tune.
    
- Analyze Lalign​ and Lunif​ loss components. Per , Lunif​ should converge faster or to a better (lower) value.   
    

## 4. Experiment 3 (High Complexity): Hyperbolic Vision Transformer (HVT) Backbone

**Goal:** Combine a SOTA-analogue backbone (HVT) with BAU's SOTA loss. This is the most advanced configuration, synthesizing two distinct research lines.   

**Step-by-Step Implementation:**

1. **Implement HVT:**
    
    - This requires a new model file, e.g., `bau/hvt.py`.
        
    - Re-implement the Hyperbolic Vision Transformer from Guzman et al..   
        
    - This model will likely have _internal_ `geoopt.ManifoldParameter`s, especially in its projection head, which maps patch tokens to the hyperbolic space.
        
    - Its `forward` pass will _natively_ output features (a `ManifoldTensor`) on the `PoincareBall`.
        
2. **Modify Model (`bau/model.py`):**
    
    - Replace the existing ResNet/ViT backbone with your new `HVT` model.
        
    - The `BAUModel` `forward` pass now just calls `return self.backbone(images)`.
        
3. **Modify Optimizer Setup (`train.py`):**
    
    - Because the `HVT` _itself_ has `ManifoldParameter`s, you _must_ use `geoopt.optim.RiemannianAdam`.   
        
    - This is the "Case 1" scenario and requires splitting parameters:
                
        ```python
        euclidean_params = [p for n, p in model.named_parameters() 
                            if not isinstance(p, geoopt.ManifoldParameter)]
        hyperbolic_params = [p for n, p in model.named_parameters() 
                             if isinstance(p, geoopt.ManifoldParameter)]
        
        # Ensure all params are captured
        assert len(list(model.parameters())) == len(euclidean_params) + len(hyperbolic_params)
        
        optimizer_euc = torch.optim.Adam(euclidean_params, lr=lr_e)
        optimizer_hyp = geoopt.optim.RiemannianAdam(hyperbolic_params, lr=lr_h)
        ```
        
    - The training loop must `zero_grad()` and `step()` _both_ optimizers.
                
        ```python
        optimizer_euc.zero_grad()
        optimizer_hyp.zero_grad()
        loss.backward()
        optimizer_euc.step()
        optimizer_hyp.step()
        ```
        
4. **Modify Loss Function (`bau/loss.py`):**
    
    - The modifications are _identical_ to Experiment 2. The loss function receives `ManifoldTensor`s from the HVT and computes the alignment/uniformity losses using `self.manifold.dist2()`.   
        

**Analysis:**

- This experiment combines a hyperbolic backbone with a hyperbolic loss.
    
- It is the most complex but has the highest theoretical potential, as it fully aligns with the principles of  (hyperbolic uniformity) and  (BAU framework).   
    

## 5. Evaluation and Analysis Directives

1. **Primary Metric:** Domain-Generalizable Re-ID performance. Use the standard DG Re-ID benchmark protocol (e.g., as cited in ), training on source domains and evaluating on unseen target domains (e.g., Market-1501, DukeMTMC-reID). Report mean Average Precision (mAP) and Rank-1 accuracy.   
    
2. **Secondary Metric:** Analyze the Lalign​ and Lunif​ loss components separately. Plot their convergence. We hypothesize Lunif​ will be significantly lower (better) in Experiments 2 and 3.
    
3. **Visualization:** Project the final embeddings from all experiments into 2D using the Poincaré disk visualization. Qualitatively assess uniformity and class separation.
    
4. **Hyperbolicity Measurement:** Using the method from Khrulkov et al. , measure the Gromov δ-hyperbolicity of the _learned embeddings_ from the baseline BAU and the hyperbolic BAU. We hypothesize the hyperbolic BAU embeddings will have a lower δ (be "more hyperbolic").   
    

## Part IV: Concluding Analysis and Recommendations

The synthesis of literature and codebase analysis reveals three distinct integration pathways, each with a clear trade-off between implementation complexity and theoretical potential.

- **Proposal 1 (Hyperbolic Head):** This is a low-risk, low-reward endeavor. It is the easiest to implement as it modularly adds a component without altering the core, working `BAU` loss. However, it fails to test the central hypothesis. The core metric learning, which is the entire point of the `BAU` model , remains Euclidean. It is a weak integration, unlikely to yield significant, publishable gains.   
    
- **Proposal 2 (Hyperbolic Metric):** This is the most direct, elegant, and scientifically sound test of the user's query. It isolates the key variable: the geometry of the loss function. A critical determination from the analysis is that this proposal _does not_ require the complex `RiemannianAdam` optimizer. The backbone parameters remain Euclidean, and the standard `Adam` optimizer is sufficient for backpropagating the Euclidean gradient derived from the hyperbolic loss computation. This significantly lowers the implementation barrier from "high" to "medium." Its success depends entirely on the hypothesis that dhyperbolic​ is a better metric for the Lalign​/Lunif​ formulation than deuclidean​.   
    
- **Proposal 3 (Hyperbolic Backbone):** This is the highest-cost, highest-potential strategy. It is not a simple integration but a _synthesis_ of two separate SOTA papers (Guzman et al.  and Cho et al. ). It requires re-implementing the HVT backbone and, crucially, managing a complex, split-optimizer training loop (`Adam` for Euclidean ViT blocks, `RiemannianAdam` for the hyperbolic projection head ). This is the most promising path to a new, state-of-the-art model.   
    

### Strategic Recommendation

1. The primary, immediate effort should be focused on **Proposal 2 (Hyperbolic Metric)**. It directly addresses the core hypothesis with medium complexity and high scientific value. Its success or failure will provide a clear, interpretable answer to the user's query.
    
2. If Proposal 2 is successful (i.e., outperforms the baseline `BAU`), it provides a strong foundation and motivation to proceed to **Proposal 3 (Hyperbolic Backbone)**. This second step would then be a logical pursuit of a new, publishable SOTA model, combining the best-in-class hyperbolic-aware backbone  with the newly-validated hyperbolic-native `BAU` loss function.   
    
3. **Proposal 1 (Hyperbolic Head)** should be deprioritized or skipped. It is a theoretically-weak compromise that does not meaningfully engage with the core premise of the `BAU` framework, which is the metric learning of aligned and uniform embeddings.
    