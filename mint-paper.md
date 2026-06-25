# Learning the language of protein-protein interactions

**Authors**: Varun Ullanat<sup>1</sup>, Bowen Jing<sup>1</sup>, Samuel Sledzieski<sup>1,2</sup>, Bonnie Berger<sup>1,3</sup>  
**Corresponding author**: Bonnie Berger (bab@mit.edu)  
**Published**: Nature Communications (2026) 17:1199  
**DOI**: https://doi.org/10.1038/s41467-025-67971-3  

**Affiliations**:  
<sup>1</sup> Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute of Technology, Cambridge, MA  
<sup>2</sup> Center for Computational Biology, Flatiron Institute, New York, NY  
<sup>3</sup> Department of Mathematics, Massachusetts Institute of Technology, Cambridge, MA  

---

## Abstract

Protein Language Models (PLMs) trained on large databases of protein sequences have proven effective in modeling protein biology across a wide range of applications. However, while PLMs excel at capturing individual protein properties, they face challenges in natively representing protein-protein interactions (PPIs), which are crucial to understanding cellular processes and disease mechanisms. Here, we introduce MINT, a PLM specifically designed to model sets of interacting proteins in a contextual and scalable manner. Using unsupervised training on a large curated PPI dataset derived from the STRING database, MINT outperforms existing PLMs in diverse tasks relating to protein-protein interactions, including binding affinity prediction and estimation of mutational effects. Beyond these core capabilities, it excels at modeling interactions in complex protein assemblies and surpasses specialized models in antibody-antigen modeling and T cell receptor-epitope binding prediction. MINT's predictions of mutational impacts on oncogenic PPIs align with experimental studies, and it provides reliable estimates for the potential for cross-neutralization of antibodies against SARS-CoV-2 variants of concern. These findings position MINT as a powerful tool for elucidating complex protein interactions, with significant implications for biomedical research and therapeutic discovery.

---

## Main Text

The success of large language models in natural language processing—where complex semantic and syntactic relationships are learned from sequences of words—has inspired their application to protein sequences. By treating amino acid sequences as "sentences," protein language models (PLMs) can implicitly learn structural and functional patterns, enabling predictions of protein folding, mutational effects, and antibody optimization without explicit structural labels. However, within cellular environments, proteins rarely act in isolation; instead, they form extensive interaction networks essential for processes such as signal transduction, metabolic pathways, and cellular structural stability. Hence, a comprehensive understanding of protein biology requires moving beyond isolated protein sequences to consider the complex interactions between multiple proteins. Despite this, even PLMs that predict high-resolution structures have thus far been limited to learning patterns from single protein sequences.

Since almost every PLM is trained using a self-supervised objective on single chains, models often struggle to effectively capture protein-protein interactions (PPIs). Previous approaches that have used PLMs to predict PPIs utilized representations generated from each protein sequence without incorporating the contextual information from the interacting partners. This independence causes critical interaction-specific features to be overlooked, as the representations of each protein sequence are generated in isolation. Furthermore, this approach becomes increasingly impractical for PPIs that involve complex multi-sequence (2+) interactions, such as those seen in antibody-antigen or TCR-epitope-MHC complexes. While concatenating input sequences has been proposed as a workaround, it risks degrading embedding quality by treating all sequences as a single unified entity, potentially masking distinct sequence-specific features. To address these challenges, we propose Multimeric Interaction Transformer (MINT), an extension of the single-sequence paradigm that allows PLMs to learn distributions of sets of interacting protein sequences. We hypothesize that this approach will enable PLMs to produce context-aware representations that more accurately capture the nuances of PPIs.

MINT contains two new conceptual advances that enable it to model PPIs effectively. The first modification is of the popular Masked Language Modeling (MLM) objective, so that our model can learn from the interacting protein chains present in STRING-DB, a comprehensive and high-confidence database filtered to contain information on 96 million experimental and predicted PPIs. By training a PLM on this vast corpus of interactions, we aim to uncover deeper representations of not only individual proteins but also how sets of proteins function together in concert. To the best of our knowledge, no existing PLM has been trained extensively on STRING-DB using a self-supervised pretraining objective. Our second innovation is the adaptation of the model architecture and the training of the popular PLM ESM-2 to handle multiple inputs of protein sequences at once. We present a strategy that allows us to fine-tune ESM-2 on STRING-DB by adding a cross-attention module that explicitly extracts inter-sequence information.

We comprehensively benchmarked MINT against widely used PLMs across multiple PPI prediction tasks. MINT consistently outperformed baseline models in binary interaction classification, binding affinity prediction, and mutational impact assessment, achieving a new state-of-the-art AUPRC of 0.69 on the gold-standard dataset constructed by Bernett et al., and delivering a 30% improvement in predicting binding affinity changes upon mutation on the SKEMPI dataset. Furthermore, MINT demonstrated superior performance in antibody modeling, exceeding antibody-specific baselines on property prediction tasks from the Fitness Landscapes for Antibodies (FLAB) benchmark by over 10%. It also outperformed AbMap in predicting binding affinity changes in SARS-CoV-2 antibody mutants, achieving a 14% performance gain in the setting where only 0.5% of the data was available for training. In TCR-epitope modeling, MINT surpassed state-of-the-art models like PISTE and AVIB-TCR with minimal fine-tuning. We show how MINT can model the important task of predicting variant effects in PPIs. In oncogenic PPI (oncoPPI) analysis, MINT effectively distinguished between pathogenic and non-pathogenic interactions, with its predictions matching 23 of 24 previously experimentally validated mutational effects. Similarly, in the prediction of antibody cross-neutralization against SARS-CoV-2 variants, MINT achieved high precision-recall performance, capturing shifts in neutralization profiles across Omicron sub-variants and demonstrating an 80% accuracy in identifying antibodies with consistent neutralization capabilities. By treating multimeric interactions as interdependent sets of sequences rather than isolated groups, MINT provides a unified approach to computational PPI modeling, offering a powerful framework for studying disease mechanisms, guiding therapeutic design, and advancing immunological research.

---

## Results

### Overview of the MINT model and training

PLMs have been successfully applied to predict the structural, functional, and evolutionary attributes of proteins. However, encoding PPIs poses unique challenges, as PLMs are traditionally trained to model single protein sequences independently. To address these challenges, prior approaches have involved passing each sequence through the PLM separately and then concatenating the embeddings, or alternatively, embedding interacting proteins as a single continuous sequence (Fig. 1a). However, these methods introduce limitations, such as the loss of inter-residue context and an oversimplified treatment of multi-sequence interactions.

To address these limitations, we developed MINT, a PLM specifically designed to model PPIs by enabling the simultaneous input of multiple interacting protein sequences (Fig. 1b). Building on the 650-million-parameter ESM-2 architecture, MINT introduces a cross-chain attention mechanism that preserves inter-sequence relationships and scales effectively to interactions with more than two chains. Whereas the self-attention mechanism in ESM-2 utilizes rotary positional encoding to capture intra-sequence positional relationships, MINT applies self-attention solely within chains and incorporates additional attention blocks for cross-chain interactions without rotary encoding. This modification ensures that each token representation in our model captures contextual information across all input sequences (Fig. 1c). Further architectural details and pseudocode are provided in the Methods Section "MINT architecture".

To train MINT, we curated a dataset from the STRING database, which includes 2.4 billion physical PPIs and 59.3 million unique protein sequences. By applying clustering and diversity measures, we refined this dataset to 96 million high-quality PPIs involving 16.4 million unique sequences ("Methods" Section "STRING dataset construction"). This dataset serves as the foundation for training MINT using a MLM objective augmented with cross-attention. Unlike traditional MLM, where token prediction is conditioned solely on intra-sequence context, MINT leverages cross-chain representations to capture coevolutionary constraints imposed by interacting residues (Fig. 1c, "Methods" Section "MINT architecture"). Model training followed a multiphase approach, beginning with the initialization of attention and embedding weights from ESM-2. This warm-start approach allowed us to preserve foundational sequence-level knowledge while optimizing cross-chain interactions. The model was trained with a masking scheme and hyperparameters closely aligned with ESM-2, with the objective function incorporating both sequence-specific and interaction-specific signals ("Methods" Section "MINT training"). Performance during training was assessed using the perplexity metric ("Methods" Section "MINT training") on a validation set obtained from a random split of the curated dataset, and compared against the base ESM-2 models (Supplementary Note 2). MINT exhibited lower perplexity than the sequence-concatenated ESM-2 baseline, indicating improved modeling of PPIs. To further examine MINT's generalization capacity, we performed supervised PPI prediction on a held-out STRING validation set that was excluded from pretraining, achieving superior performance over the ESM-2-650M baseline (Supplementary Note 3).

Together, these architectural and training innovations enable MINT to overcome the inherent limitations of existing PLMs in learning PPI rules. We demonstrate versatility in handling diverse protein sequence inputs, including general complexes, antibody heavy and light chains, T-Cell Receptor (TCR) regions, and peptides, without restrictions on the number of sequences processed concurrently (Fig. 1d). This capability allows MINT to capture intricate features relevant to disease-specific interactions and mechanisms, such as mutational impacts in cancer and the cross-neutralization potential of SARS-CoV-2 antibodies.

**Figure 1 (description):**  
*Approaches to PPI modeling and MINT overview.*  
- **a** Existing PLMs either process multiple interacting proteins by concatenating output embeddings (left) or concatenating input tokens (right).  
- **b** MINT treats multiple interacting sequences as separate entities and generates embeddings contextually, conserving cross-sequence relationships.  
- **c** Workflow and architecture: each sequence is tokenized using ESM-2 tokenizer, special start/end tokens added. Cross-attention blocks added to base ESM-2 model.  
- **d** Non-exhaustive list of protein types, PPI properties, and research questions evaluable with MINT.  
(Created in BioRender. Ullanat, V. (2025) https://BioRender.com/dS0o431)

### Comprehensive benchmarking of MINT on PPI prediction

PPI prediction is fundamental to understanding cellular processes, offering insights into disease mechanisms and therapeutic targets. Although structure-based models such as AlphaFold provide detailed predictions, they face challenges with scalability and accuracy for non-interacting pairs. Sequence-based methods, in contrast, offer efficiency and flexibility but fall short in explicitly encoding PPIs. To address these gaps, we benchmarked MINT against widely used PLMs on supervised tasks including binary interaction classification, binding affinity prediction, and mutational impact assessment. We use standard dataset splits and provide detailed descriptions of dataset construction, splitting protocols, and measures to mitigate potential information leakage in "Methods" Section "Benchmarking tasks".

**Figure 2 (description):**  
*Performance of MINT versus other PLMs on general PPI tasks.*  
- **a** Framework for downstream tasks: embeddings from baseline PLMs and MINT used with MLP to predict binary interaction or binding affinity.  
- **b** Framework for mutation effect analysis: wild-type and mutated sequence groups.  
- **c-h** Results for all benchmarking tasks: Human PPI, Yeast PPI, Gold-standard PPI, Mutational PPI, SKEMPI, PDBBind. MINT consistently outperforms baselines.

We also evaluated the ability of MINT to resolve mutational effects on binding on two tasks. The first contained data from SKEMPI—a database documenting changes in binding affinity due to mutations in one of the interacting proteins—for which we evaluated the models with previously established cross-validation splits by protein complex. The second task involved predicting whether two human proteins remain bound after a mutation occurs in one of them, thereby testing the model's sensitivity to sequence alterations that impact binding outcomes.

In each task, MINT's embeddings were utilized to train lightweight predictor models, facilitating equitable comparisons with baseline pretrained language models (PLMs) such as ESM-1b, ESM-2 (650M and 3B), ProGen (3B), and ProtT5 (3B). We compared two baseline embedding strategies: (1) concatenating independently computed embeddings from the PLM and (2) concatenating interacting sequences as a single input to the PLM (Fig. 1). For all tasks except PDB-Bind, the first strategy worked better. MINT consistently outperformed all baseline PLMs across all tasks. On the gold-standard dataset, MINT achieved AUPRC of 0.69. In binary PPI prediction, MINT achieved an improvement of 11% over the second-best baseline in yeast interactions and 3% over the average baseline for human interactions. On SKEMPI, our model achieved a 32% improvement. Finally, inference time benchmarking (Supplementary Note 4) showed that MINT also achieves competitive runtime efficiency.

### Performance on antibody modeling tasks

Building on its superior performance in general PPI tasks, we extended MINT to domain-specific challenges in antibody and immune modeling. Antibodies are symmetrical Y-shaped molecules composed of two heavy chains and two light chains. We compare MINT with three deep-learning-based approaches specific to antibody modeling.

We compared MINT against IgGbert and IgG5—two models finetuned on extensive datasets of antibody light and heavy chains. We assessed performance on four supervised property prediction tasks using the FLAB dataset, which includes experimentally measured binding energy data. MINT achieves a performance boost over antibody-specific baselines on all four datasets, with an increase of over 10% on three of them (Fig. 3b).

We further evaluated MINT against AbMap on a mutational variation prediction task involving changes in binding affinity (ddG) for m396 antibody mutants against SARS-CoV-2. MINT achieves comparable or better performance compared to AbMap across all data-splitting instances (Fig. 3d). Crucially, our model achieves a 14% increase in performance when trained on just 0.5% of the samples.

**Figure 3 (description):**  
*Comparing MINT to antibody-specific PLMs.*  
- **a** Framework for FLAB benchmark: MINT treats heavy and light chains as separate sequences.  
- **b** Results across four datasets: MINT shows higher R² values.  
- **c-d** SARS-CoV-2 binding task: MINT outperforms AbMap, especially with limited training data.

### Learning the language of TCR-Epitope-MHC interactions with minimal finetuning

The interactions between T cell receptors (TCRs), epitopes, and major histocompatibility complex (MHC) molecules are central to the human immune response. Several deep learning-based approaches have been developed to model these interactions. MINT offers a flexible framework. We minimally fine-tuned the final layer of MINT to learn TCR-epitope-MHC interactions.

For first-order tasks (TCR-epitope binding), MINT obtains an average AUROC of 0.581, surpassing the best baseline (0.576) (Fig. 4b). For second-order interaction prediction (TCR-epitope-HLA), MINT achieves the highest performance across all dataset splits (Fig. 4d). For interface prediction, MINT matches or exceeds TEIM performance (Fig. 4f).

**Figure 4 (description):**  
*Comparing finetuned MINT to TCR-MHC-Epitope models.*  
- **a-b** First-order prediction: MINT achieves AUROC of 0.581.  
- **c-d** Second-order prediction: MINT outperforms PISTE, pMTNet, etc.  
- **e-f** Interface prediction: MINT matches TEIM on unseen TCRs and outperforms on unseen epitopes.

### Leveraging MINT for predicting mutation-induced perturbations in oncogenic PPIs

We applied MINT to estimate pathogenicity of missense mutations affecting PPIs in cancer. Cheng et al. identified 470 potential oncoPPIs and experimentally validated 24 somatic missense mutations. Using MINT embeddings and an ensemble of 100 trained models, we predicted binding scores. MINT's predictions match the mutational effects for 23 out of 24 PPIs (Fig. 5c). Only one oncoPPI (ARHGDIA-RHOA) was incorrectly assigned.

**Figure 5 (description):**  
*Overview and results of mutational effect prediction in oncoPPIs.*  
- **a** Analysis outline.  
- **b-c** Predicted binding scores and true mutational effects for 24 mutations. Green check marks denote correct predictions (23/24). Threshold = 0.68.

### MINT predicts antibody cross-neutralization against SARS-CoV-2 variants

We utilized MINT with CoV-AbDab database to encode antibodies and SARS-CoV-2 spike proteins. Training used early pandemic variants; evaluation on Omicron sub-variants. MINT demonstrated robust predictive performance (high AUPRC). For vaccine-induced antibodies, MINT achieved 80% hit rate for correctly identifying antibodies with consistent neutralization across Omicron sub-variants (Fig. 6e).

**Figure 6 (description):**  
*Antibody cross-neutralization against SARS-CoV-2 variants.*  
- **a** Procedure: extract data from CoV-AbDab, train MLP on MINT embeddings.  
- **b** Dataset composition across Omicron sub-variants.  
- **c-d** Normalized scores and AUPRC values; vaccine-induced antibodies show higher performance.  
- **e** Normalized scores for 10 antibodies compared to experimental IC50 values.

---

## Discussion

In this study, we introduced MINT, a protein language model designed to encode PPIs by simultaneously processing multiple interacting protein sequences. Unlike traditional PLMs that treat sequences independently, MINT incorporates cross-chain attention mechanisms to capture inter-sequence relationships. Our comprehensive benchmarking demonstrated that MINT significantly outperforms existing PLMs across diverse supervised downstream tasks involving PPIs.

Despite these advancements, there are limitations. MINT is currently only trained on pairs of sequences from STRING. However, our architecture can support pre-training on complexes with variable numbers of interacting sequences. Performance gains vary across datasets; on saturated tasks like HumanPPI, gains are modest. MINT's performance is inherently linked to the quality and diversity of training data. For underrepresented protein classes, we recommend parameter-efficient fine-tuning.

Finally, although MINT achieves strong performance, incorporating additional data modalities (e.g., protein structure) could further expand its capabilities. Future iterations could merge scalability of sequence-based methods with structural insights from models like AlphaFold3.

MINT represents an advancement in protein language modeling by effectively capturing complex inter-sequence dependencies crucial for accurate PPI prediction. Its versatility across multiple tasks underscores its potential as a powerful tool in biomedical research.

---

## Methods

### STRING dataset construction

We started with 2.4 billion PPIs comprising 59.3 million unique protein sequences from STRING. Using mmseqs to cluster at 50% sequence similarity, we obtained 15.6 million unique clusters. We kept only one PPI between any two clusters, resulting in 382 million PPIs (29 million unique sequences). After splitting and filtering to avoid leakage, we ended with 95.8 million training PPIs (16.4 million unique sequences). See Supplementary Fig. S1.

### MINT architecture

We used ESM-2 as backbone. Each layer receives input embeddings x ∈ R<sup>L×F</sup>, self-attention mask m<sup>att</sup> ∈ R<sup>L×L</sup>, padding mask m<sup>pad</sup> ∈ R<sup>L×L</sup>. MINT uses MultiHeadAttention module adapted to output both pre-softmax attention matrix and value tensor. Both self-attention and cross-attention use the same module but with different attention masks. Pseudocode provided in Algorithm 1.

**Algorithm 1 – Single functional transformer block of MINT** (see paper for detailed pseudocode)

### MINT training

Trained using MLM objective: 15% of tokens masked (80% [MASK], 10% random amino acid, 10% unchanged). Adam optimizer (β1=0.9, β2=0.98, ε=10<sup>-8</sup>, weight decay 0.01). Learning rate increased over first 2000 steps to peak 4e-4, then reduced to one-tenth over 90% of training. Cropped long sequences to 512 tokens. Batch size 2, gradient accumulation over 32 batches (effective batch size 64). Trained on NVIDIA A100 80GB and RTX A6000 GPUs for 4 million steps. Validation perplexity: MINT 4.84 vs. ESM-2 single sequence 5.41 vs. concatenated 5.16.

### Benchmarking tasks

**Table 1 | Summary of all datasets and tasks**

| Task | Reference | Train size | Validation size | Test size | Task type | Projector | Evaluation metric |
|------|-----------|------------|----------------|-----------|-----------|-----------|------------------|
| **General PPI prediction** | | | | | | | |
| Gold-standard PPI | 22 | 163,019 | 59,260 | 52,048 | Binary classification | MLP | AUPRC |
| Human-PPI | 67 | 26,319 | 234 | 180 | Binary classification | MLP | Accuracy |
| Yeast-PPI | 12 | 4,945 | 95 | 394 | Binary classification | MLP | Accuracy |
| PDB-Bind | 28 | 4,945 | 95 | 394 | Regression | MLP | Pearson correlation |
| SKEMPI | 23 | 4,777 | – | 1,929 | Regression | MLP | Pearson correlation |
| MutationalPPI | 29 | 3,406 | – | – | Binary classification | MLP | AUPRC |
| **Antibody tasks** | | | | | | | |
| FLAB (Binding 422) | 35 | 422 | – | – | Regression | Ridge regression | R² |
| FLAB (Binding 2048) | 36 | 2,048 | – | – | Regression | Ridge regression | R² |
| FLAB (Binding 4275) | 37 | 4,275 | – | – | Regression | Ridge regression | R² |
| FLAB (Expression 4275) | 37 | 4,275 | – | – | Regression | Ridge regression | R² |
| SARS-CoV2 binding | 38 | 86,929 | – | – | Regression | Ridge regression | Spearman rank |
| **TCR-Epitope-MHC tasks** | | | | | | | |
| TDC-Tchard | 44 | 522,239 | – | 71,666 | Binary classification | MLP | AUROC |
| TCR-Epitope-HLA | 17 | 28,144 | 71,036 | 2,806 | Binary classification | MLP | AUROC |
| TCR-epitope interface | 46 | 122 | – | – | Interface prediction | CNN | AUPRC |

For baseline PLMs, we evaluated two embedding strategies: concatenating independent embeddings or concatenating input tokens. For regression: Mean Squared Error loss; for classification: Binary Cross-Entropy loss. Used 2-layer MLP with hidden size 640.

**Antibody tasks (FLAB):** Linear least squares with L2 regularization, 10-fold CV, nested 5-fold inner CV for λ selection.

**TCR-Epitope-MHC tasks:** Fine-tuned last layer of MINT (32 of 33 layers frozen, ~26M parameters trained).

### Case studies

**OncoPPI prediction:** Training set from Siwek et al. (2025) with labels indicating whether two human proteins continue to bind after missense mutation. Evaluated on 24 experimentally validated mutations from Cheng et al. Performed 100 training repetitions; binding score = fraction of predictions = 1. Threshold determined by Gaussian Mixture Model = 0.68.

**SARS-CoV-2 cross-neutralization:** Data from CoV-AbDab, filtered for antibodies against early variants (training) and Omicron sub-variants (evaluation). Used MINT embeddings of heavy chain, light chain, and RBD sequence. Trained MLP with BCE loss. Quantile normalization for normalized scores.

---

## Data Availability

- STRING-DB: physical PPI training data
- Gold-standard PPI: https://fishare.com/articles/dataset/PPI_prediction_from_sequence_gold_standard_dataset/21591618/3
- HumanPPI: https://github.com/westlake-repl/SaProt
- YeastPPI: PEER benchmark
- SKEMPI: https://life.bsc.es/pid/skempi2
- PDB-Bind: https://www.pdbbind-plus.org.cn
- Mutational PPI: https://github.com/jishnu-lab/SWING
- FLAB: https://github.com/Graylab/FLAb
- SARS-CoV-2 binding: https://www.biorxiv.org/content/10.1101/2020.04.03.024885v1_supplementary-material
- TCR-epitope (TDC): https://tdcommons.ai
- TCR-epitope-HLA: https://github.com/Armilius/PISTE
- TCR-epitope interface: https://github.com/penxingang/TEIM
- OncoPPI data: https://github.com/ChengF-Lab/oncoPPIs
- SARS-CoV-2 neutralization: https://opig.stats.ox.ac.uk/webapps/covabdab

## Code Availability

Code publicly available at https://github.com/VarunUllanat/mint under MIT License. Publication release deposited on Zenodo: https://doi.org/10.5281/zenodo.17174875.

---

## Supplementary Information

### Supplementary Note 1: Dataset processing experiments

We experimented with three strategies for dataset splitting from STRING-DB:

1. **Random split**: Shuffle PPIs, keep 250k as validation, rest (382M) as training. Risk: promiscuous clusters appear in both sets.
2. **Filtered split**: Ensure no cluster in training set appears in validation set. Results in 95.8M training PPIs (16.4M unique sequences). Removes promiscuity but training set may still contain highly promiscuous clusters.
3. **Filtered 50 split**: Keep only PPIs with unique cluster pairs (50% similarity between any two PPIs). Results in 1.8M training PPIs (very small set, reduced diversity).

**Supplementary Figure S1** shows an overview of these dataset splitting and filtering strategies.

We trained models on all splits and evaluated on the gold-standard dataset; the Filtered split strategy was selected for final MINT training.

### Supplementary Note 2: MINT training configurations

Three strategies considered:

- **MINT-Reinitialize**: All parameters randomly initialized and updated.
- **MINT-Freeze**: Self-attention and embedding layers copied from ESM-2, only cross-attention layers updated.
- **MINT-Nofreeze**: Self-attention and embedding weights imported from ESM-2, all parameters trained.

**Supplementary Figure S2** shows the three training configurations.  
**Supplementary Figure S3** shows validation perplexity curves:

- MINT-Reinitialize: slow convergence.
- MINT-Freeze: fast convergence but higher perplexity.
- MINT-Nofreeze: fast convergence, lowest perplexity (4.84 after 2.5M steps).  
ESM-2 baseline: single sequence perplexity = 5.41, concatenated = 5.16.

MINT-Nofreeze selected as the final MINT model.

### Supplementary Note 3: Evaluation on an independent STRING validation set

We used the validation subset of STRING (250k PPIs, clustered at 50% identity, not used in pre-training). Negatives generated by shuffling protein partners. Trained binary classifiers using embeddings from MINT and ESM-2-650M.

**Supplementary Table S2 | Comparison of MINT against ESM-2-650M on an independent STRING validation set**

| Model | Concatenation Type | Accuracy | AUPRC | F1 score |
|-------|--------------------|----------|-------|----------|
| ESM-2-650M | Concatenate tokens | 0.72 | 0.78 | 0.73 |
| ESM-2-650M | Concatenate embeddings | 0.79 | 0.85 | 0.80 |
| MINT | Concatenate tokens | 0.82 | 0.88 | 0.83 |

MINT achieved higher performance across all metrics, demonstrating strong generalization to novel PPIs.

### Supplementary Note 4: Inference time benchmarking of MINT

Benchmarked on NVIDIA A100 80GB, batch size = 1, average over 100 runs.

**Supplementary Table S3 | Inference time benchmarking of MINT against baseline PLMs**

| Model | Parameters (M) | Concatenation Type | L=50 (ms) | L=100 (ms) | L=200 (ms) | L=500 (ms) |
|-------|----------------|--------------------|-----------|------------|------------|------------|
| ProtT5-BFD | 1208 | Concatenate tokens | 4.58 | 6.16 | 9.39 | 11.41 |
| ProtT5-Uniref | 1208 | Concatenate tokens | 4.58 | 6.08 | 9.25 | 11.43 |
| Progen2-Large | 2779 | Concatenate tokens | 8.69 | 12.42 | 20.00 | 22.22 |
| ESM-2-3B | 2841 | Concatenate tokens | 8.41 | 13.02 | 21.48 | 22.87 |
| ESM-1b-650M | 652 | Concatenate tokens | 3.39 | 4.59 | 6.88 | 7.18 |
| ESM-2-650M | 652 | Concatenate tokens | 4.15 | 4.77 | 7.12 | 7.41 |
| ESM-2-150M | 148 | Concatenate tokens | 3.40 | 3.41 | 3.44 | 3.49 |
| ProtT5-BFD | 1208 | Concatenate embeddings | 7.49 | 9.11 | 12.20 | 22.66 |
| ProtT5-Uniref | 1208 | Concatenate embeddings | 7.23 | 8.34 | 12.40 | 23.03 |
| Progen2-Large | 2779 | Concatenate embeddings | 14.23 | 17.83 | 24.91 | 43.98 |
| ESM-2-3B | 2841 | Concatenate embeddings | 11.64 | 16.82 | 25.92 | 45.69 |
| ESM-1b-650M | 652 | Concatenate embeddings | 7.02 | 7.73 | 9.19 | 14.30 |
| ESM-2-650M | 652 | Concatenate embeddings | 7.63 | 8.39 | 9.68 | 14.70 |
| ESM-2-150M | 148 | Concatenate embeddings | 6.12 | 6.82 | 6.23 | 6.86 |
| **MINT** | **813** | **Concatenate embeddings** | **6.23** | **6.91** | **7.36** | **18.19** |

MINT achieves competitive inference times compared to models of comparable or larger size, especially at shorter sequence lengths.

### Supplementary Note 5: Finetuning MINT

For TCR-epitope interaction prediction tasks, we froze all model parameters except those of the last layer (32 out of 33 layers frozen, training only the 33rd layer). This prevents overfitting, catastrophic forgetting, improves training stability, and reduces computational cost. Including the MLP layer, total fine-tuned parameters = ~26 million.

---

## Supplementary Tables S1 (detailed comparison of all models using concatenation techniques)

**Supplementary Table S1** (provided in the supplementary PDF) shows performance metrics (mean ± s.d.) for all baseline models and MINT across Human PPI, Yeast PPI, Gold-standard PPI, Mutational PPI, SKEMPI, and PDB-Bind tasks, for both concatenation strategies. MINT achieves best or second-best performance in all categories.

---

## Acknowledgements

This work was supported by the National Institute of General Medical Sciences of the National Institutes of Health under award number 1R35GM141861 and by a research gift from Quanta Computer. B.J. was partially supported by the Department of Energy Computational Science Graduate Fellowship under Award Number DESC0022158. S.S. was partially supported by the NSF Graduate Research Fellowship under Grant No. 2141064.

---

## Author Contributions

B.J., S.S., and B.B. conceptualized the project. V.U. and B.J. constructed the training pipeline for MINT. V.U. and B.J. ran the training. V.U. performed downstream computational analysis, including model benchmarking and case studies. B.B. designed and led the study. All authors contributed to writing the manuscript.

---

## Competing Interests

The authors declare no competing interests.

---

## Corresponding author

Bonnie Berger: bab@mit.edu