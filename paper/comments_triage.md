# Abstract
- "unified asymmetric scoring layer" Phrasing seems too superfluous
- "On M+MS+CS$\to$CUHK03, five-seed mean mAP is $43.8\pm0.3\%$;" These notations were not introduced yet, therefore refrain from using abbreviations as they would only confuse the every day reader
- "protocol-matched run" What is a "protocol-matched" run? If this is highlighting the best performing run among these 5 seeds, the message is not coming thorugh
- "mechanism" wording of mechanism here is very vague.
- "Randers correction" The terminology has yet been introduced, refrain from using wordings that are ambiguous such as this
- "drift-subspace variance regularization, not explicit domain-mining auxiliaries, is the critical ingredient for stable asymmetric retrieval". Also include the conclusion from the toy dataset, that we could prove the intuition

In general, do not reference notations, terminology or results that have not been introduced. Try to engineer a creative hook for the readers without delving too deep into the paper.

# Introduction
- "Direction~A (clean query, corrupted gallery) is easier than Direction~B at $k{=}1$--$4$, with the gap peaking near $+0.20$ at $k{=}3$; this asymmetry motivates direction-aware scoring." Do we need to explain here the change in trend from k_1 to k_2 and from k_3 to k_4? I.e., why it isn't monotonically increasing with k?
- "probe→gallery and gallery→probe" Paraphrase this sentence without the use of arrow here; seems unprofessional
- "cautionary datapoint" Wording seems superfluous
- "continuous feature map" What do we mean here? It feels very ambiguous
- Last paragraph (contributions listing): Some of the stuff here are repetition of the paragraph above. We should also avoid referencing notation/concepts that are not introduced yet (e.g., L_dcc). Lastly, the last contribution is word-for-word identical to the entry in the abstract, which should not be the case. Clean this contribution list up by only stating the core contributions instead of listing every single small effort. Abstract the results yielded in this study to the bigger picture and answer the question: How does the information in this paper benefit the person re-id community?

# Related Work
- "Retrieval geometry" Is retrieval geometry the right heading here? It feels somewhat out of place
- "A fourth design axis keeps a standard Euclidean feature map and injects direction dependence in the pairwise score rather than in the geodesics of the embedding space; Finsler-MDS \cite{dages2025finsler} is a recent example of this template (see Sec.~\ref{subsec:asymmetric_finsler_distance})." Finsler MDS has nothing to do with "retrieval geometry". Perhaps the prelude of this section needs some touchign up
- "nuisance losses" Is the coining of "nuisance" typical or well-received in the community? Is the terminology correct?
- "piece" Where is this terminology of "piece" coming from? Is it from a reputable source on Finsler geometry?
- "linear functional of the displacement" Phrasing is weird
- "integrand-level regularity conditions" What do you mean by this? Phrasing is ambiguous
- "clean$\to$corrupted information loss typically dominates the reverse~\cite{cover2006elements}." Phrase this without the arrow, because it appears unprofessional

# Methodology
- "$\hat{\mathbf{z}}^{\omega}=\mathbf{z}^{\omega}-(\mathbf{z}^{\omega}\cdot\hat{\mathbf{z}}^{\mathrm{id}})\hat{\mathbf{z}}^{\mathrm{id}}$, where $\hat{\mathbf{z}}^{\mathrm{id}}$ is the $\ell_2$-normalized identity feature" The hat notation has been used in the diagram fro the Finsler BAU framework to depict augmented images, refrain from using the same "hat" notation
- "Geometric properties of the drift factor." Entire paragraph does not carry much contribution to the core message of the paper - deals more on how it was implemented rather than the effect, propose to move it to the supplementary
- "ensures $\|\tilde{\mathbf{z}}^{\omega}\|_2 < c_{\max} = 0.95$" Change this to alpha to match the notation from section 2.4 and update all references to c_max such that they refer to it as alpha; only do this if alpha is unused
- "the post-projection drift lies in the open ball $\hat{\mathbf{z}}^{\omega} \in B^{d_\omega}(\mathbf{0},\, c_{\max})$" Double-check if this line of reasoning is correct, and if the Rander's drift term should lie in an open ball in the first place. Provide a brief explanation on why we believe it should
- "projector $\Pi_{\hat{\mathbf{z}}^{\mathrm{id}}}^{\perp}$" The explicit mathematical expression for the projector was not introduced, is the current formulation ambiguous?
- "Setting $\hat{\mathbf{z}}^{\omega}{=}\mathbf{0}$ in the distance below recovers $d_E$ on the identity slice." Sentence is better positioned in the following paragraph, feels out of place here.
- "restrict displacement to the identity subspace" Formulation is ambiguous w.r.t. what kind of displacement is being referred to here. Opt to remove this subphrase.
- "Randers positive-definiteness" Double-check if it is really the positive-definiteness that establishes this constraint onto the drift term
- "$\|f_s - f_w\|_2^2$ between augmented views. Aligning the full embedding forces $\omega_s\approx\omega_w$," The notation here is not aligned with the section before it (usage of f instead of z for feature embeddings). Furthermore, the notation s and w is not clarified here. Propose to cement what the susbcripts s and w mean, because in this case w looks a lot like omega
- "destroying the asymmetric signal that augmentations are meant to induce" Phrasing is too strong here (destroy, meant to induce). Opt for a softer, more implicative tone, or provide concrete references that augmentations are meant to induce asymmetrical signal
- " preventing drift magnitude from biasing the repulsion kernel." Rationale here is not very sound, does not hold on its own. Propose to formulate this argument differently such that it is more intuitive to understand why we did not "finslerize" the uniform loss from BAu
- "convex combination of $S$" Double check if it really is convex.
- "Relationship between the two designs." Opt to position this in the supplementary, as it is not relevant to the main message of this paper and is only relevant to the curious readers
- "The sigmoid scaling asymptotically bounds the pre-orthogonalization drift norm below $c_{\max}{=}0.95$; the Gram-Schmidt step reduces the norm further but the post-orthogonalization bound is not explicitly re-enforced." This sentence adds little value to the core message of the paper
- "In isolation, without $\mathcal{L}_{\mathrm{dcc}}$, bidirectional mining does not consistently improve results (Table~\ref{tab:aux_loss_sweep}, Arms~0, 3, 4); combined with $\mathcal{L}_{\mathrm{dcc}}$, it yields the strongest performance in the auxiliary-loss sweep (Sec.~\ref{subsec:quantitative_analysis}, Table~\ref{tab:aux_loss_sweep})." It is actually the other way around, but we should avoid putting out such statements without a concrete ablation tests. Opt to remove this sentence altogether, also its positioning is false -- we should avoid referencing results in the methodology section
- "Total loss." I think it is better for the readability flow if we placed the total loss paragraph at the end after we've introduced all of the loss terms
- "\subsection{Drift regularization}" Subsection is fine, but the notation of omega is not consistent to previously. Prefer this one, instead of the superscript omega, but need to consolidate both of them and highlight that omega is not some new output, but a slice of the output feature. The same mistake is made in section 3.3 "Drift-head design variants"
- "The effect is not that drift semantically encodes person identity; it is that drift variance is bounded so that the asymmetric scoring term is reliable. Two interpretations are consistent with this: (i) drift as an identity-conditioned directional bias that should be view-invariant for a given person, and (ii) a soft regularizer that reduces the effective drift dimensionality required for stable Randers scoring." This part of the paragraph feels less tightly worded and requires rephrasing. We can't say that an interpretation is consistent with this design choice, because we never proved these interpretations. We never proved that "drift should be an identity-conditioned directional bias"
- " Let $\mathbf{m}_k$ denote the centroid of domain $k$ in the memory bank:" If I'm not mistaken, the memory bank keeps a centroid vector for each identity (PID) within the training dataset, rather than of each domain k. Double check this and change the notation here accordingly. 
- "\subsection{Objective interference analysis} " I'm not sure whether the positioning of this subsection here is correct, because it refers to concrete experimental results from table 4 (which should be avoided in the methodology section). Opt to move this to section 4.3 and update all references. Or we could merge it with the paragraph "auxillary-loss sweep"
- "Cross-camera drift coherence $\mathcal{L}_{\mathrm{dcc}}$ (eq.~\ref{eq:loss_dcc}) avoids camera-vs-identity hard-pair conflict by regularizing drift variance instead: the regime in which we prefer to operate when stability is paramount" Connection between the sentence is weird, requires re-wording
- "Multi-term training precludes unique attribution; the empirical pattern is nonetheless consistent with this account" This whole sentence feels superfluous and does not contribute much to the semantics, opt to remove


# Experimental Results
- "aligning with BAU~\cite{cho2024generalizable}" Reference table dataset_stats here
- " BAU cross-entropy $\mathcal{L}_{\mathrm{ce}}$, bidirectional unified Finsler triplet $\mathcal{L}_{\mathrm{tri}}^{\mathrm{bi}}$ (eq.~\ref{eq:bidirectional_triplet}), identity-only alignment $\mathcal{L}_{\mathrm{align}}$ (eq.~\ref{eq:identity_only_alignment}) and uniformity $\mathcal{L}_{\mathrm{uniform}}$ \cite{cho2024generalizable}, Finsler-domain repulsion $\mathcal{L}_{\mathrm{dom}}$ (eq.~\ref{eq:domain_loss}), barrier $\mathcal{L}_\omega$ (eq.~\ref{eq:omega_regularization}) with $\lambda_\omega{=}1.5$, and cross-camera drift coherence $\mathcal{L}_{\mathrm{dcc}}$ (eq.~\ref{eq:loss_dcc}) with $w_{\mathrm{dcc}}{=}0.1$." This is a direction repetition of the "Total loss" paragraph in section 3.4. Pick one of the two positioning for this text and avoid repetition
- " bidirectional unified Finsler triplet $\mathcal{L}_{\mathrm{tri}}^{\mathrm{bi, F}}$ + $\mathcal{L}_{\mathrm{dcc}}$ (weight $0.1$), residual drift, no $\mathcal{L}_{\mathrm{tri}}^{\mathrm{dom}}$" This is again a direct repetition of the primary recipe. The primary recipe should be clearly listed in a well-chosen position of the paper and referenced from there on
- "$d_R(q,g)=\|\mathbf{f}_g-\mathbf{f}_q\|_2+\alpha(\theta_g-\theta_q)$" Explain what f and theta is
- "The Randers correction" Don't like this sudden "novel" coining of "Randers" correction
- "Two caveats apply. First," Don't like the phrasing of this
- "$+0.023$ mean $\Delta$ at $\alpha\!=\!0.9$ over its Euclidean counterpart" Do we mean 0.041 here?
