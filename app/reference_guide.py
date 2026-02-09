
"""
Reference guide for Distillation and Quantization methods.
"""

DISTILLATION_METHODS = """
🧠 Knowledge Distillation — Complete Method Map
🔹 1. Classic Distillation
• 	Logit Distillation (Soft Targets)
• 	Hard‑Label Distillation
• 	Temperature‑Scaled Distillation
🔹 2. Feature‑Based Distillation
• 	Intermediate Feature Matching
• 	Attention Map Distillation
• 	Activation/Representation Distillation
• 	Hint‑based Distillation (FitNets)
• 	Neuron Selectivity Transfer (NST)
• 	Similarity‑Preserving KD (SPKD)
• 	Correlation Congruence KD (CCKD)
• 	Relational KD (RKD)
• 	Contrastive Representation KD
🔹 3. Loss‑Function‑Driven Distillation
• 	KLD‑based KD
• 	MSE Feature Loss
• 	Cosine Similarity Loss
• 	Triplet Loss KD
• 	Margin‑based KD
• 	Mutual Information KD
🔹 4. Multi‑Teacher Distillation
• 	Ensemble Teacher KD
• 	Weighted Multi‑Teacher KD
• 	Gated/Adaptive Teacher KD
• 	Mixture‑of‑Experts KD
• 	Cross‑Teacher Consistency KD
🔹 5. Self‑Distillation
• 	Born‑Again Networks (BAN)
• 	Deep Mutual Learning (DML)
• 	Online KD / Self‑Training
• 	Layer‑to‑Layer Self‑Distillation
• 	Progressive Self‑Distillation
🔹 6. Task‑Specific Distillation
• 	Sequence‑Level KD (NLP)
• 	Token‑Level KD (Transformers)
• 	Layer‑Drop KD
• 	Response‑Based KD (LLMs)
• 	RL‑KD (Policy Distillation)
• 	Diffusion‑Model KD (Score Distillation, Consistency Distillation)
• 	Vision‑Transformer KD (ViT‑specific)
🔹 7. Data‑Centric Distillation
• 	Data Augmentation KD
• 	Noisy Student Training
• 	Pseudo‑Label KD
• 	Curriculum KD
• 	Dataset Distillation (Synthetic Data KD)
🔹 8. Architecture‑Aware Distillation
• 	Cross‑Architecture KD (CNN→Transformer, etc.)
• 	Width/Depth‑Reduced KD
• 	Projection‑Layer KD
• 	Teacher‑Student Alignment KD

🧠 Complete List of Distillation Methods (Full Taxonomy)
🔷 1. Logit‑Level Distillation (Response‑Based)
• 	Soft‑Target Distillation
• 	Hard‑Label Distillation
• 	Temperature‑Scaled KD
• 	Kullback–Leibler KD
• 	Cross‑Entropy KD
• 	Label‑Smoothing KD
• 	Confidence‑Penalty KD
• 	Dark Knowledge Distillation
• 	Response‑Consistency KD
• 	Multi‑Teacher Logit Averaging
• 	Gated Logit Fusion KD
• 	Mixture‑of‑Experts Logit KD

🔷 2. Feature‑Level Distillation (Intermediate Representations)
• 	Feature Map Matching
• 	Activation Matching
• 	Hidden‑State Distillation
• 	Attention Map Distillation
• 	Transformer Attention Head Distillation
• 	FitNets (Hint‑Based Distillation)
• 	Neuron Selectivity Transfer (NST)
• 	Similarity‑Preserving KD (SPKD)
• 	Correlation Congruence KD (CCKD)
• 	Relational KD (RKD)
• 	Distance‑Wise RKD
• 	Angle‑Wise RKD
• 	Contrastive Representation KD
• 	Gram Matrix Distillation
• 	Jacobian Matching
• 	Layer‑to‑Layer Projection KD
• 	Cross‑Architecture Feature Alignment

🔷 3. Relation‑Based Distillation
• 	Pairwise Relation KD
• 	Triplet Relation KD
• 	Structural Relation KD
• 	Graph‑Based KD
• 	Relational Knowledge Transfer (RKT)
• 	Instance‑Relation KD
• 	Class‑Relation KD

🔷 4. Self‑Distillation
• 	Born‑Again Networks (BAN)
• 	Deep Mutual Learning (DML)
• 	Online Self‑Distillation
• 	Layer‑to‑Layer Self‑Distillation
• 	Progressive Self‑Distillation
• 	Snapshot Distillation
• 	Temporal Ensembling KD
• 	EMA‑Teacher KD (Mean Teacher)
• 	Self‑Training with Pseudo‑Labels

🔷 5. Multi‑Teacher Distillation
• 	Ensemble Teacher KD
• 	Weighted Multi‑Teacher KD
• 	Adaptive/Gated Teacher KD
• 	Mixture‑of‑Experts KD
• 	Cross‑Teacher Consistency KD
• 	Teacher‑Routing KD
• 	Teacher‑Student Graph KD

🔷 6. Task‑Specific Distillation
NLP / LLMs
• 	Sequence‑Level KD
• 	Token‑Level KD
• 	Hidden‑State KD
• 	Attention‑Pattern KD
• 	Response‑Style KD
• 	Instruction‑Following KD
• 	RLHF‑to‑SL KD (Supervised Distillation of RLHF Models)
• 	Preference‑Model Distillation
• 	Chain‑of‑Thought Distillation
• 	Self‑Consistency Distillation
• 	Logit‑Bias Distillation
• 	KV‑Cache Distillation
Vision
• 	Feature Pyramid KD
• 	Object‑Detection KD
• 	Semantic Segmentation KD
• 	Pose Estimation KD
• 	Heatmap Distillation
• 	Region‑Proposal KD
Diffusion Models
• 	Score Distillation
• 	Score Distillation Sampling (SDS)
• 	Consistency Distillation
• 	Latent‑Space Distillation
• 	Noise‑Prediction Distillation
• 	Denoiser‑to‑UNet Distillation
• 	Teacher‑Free Guidance Distillation
Reinforcement Learning
• 	Policy Distillation
• 	Value‑Function Distillation
• 	Q‑Function Distillation
• 	Behavior Cloning KD
• 	Trajectory‑Level KD
• 	Advantage‑Weighted KD

🔷 7. Data‑Centric Distillation
• 	Noisy Student Training
• 	Pseudo‑Label Distillation
• 	Curriculum Distillation
• 	Data Augmentation KD
• 	Hard‑Example KD
• 	Soft‑Example KD
• 	Dataset Distillation (Synthetic Data KD)
• 	Meta‑Learning KD
• 	Knowledge Transfer via Synthetic Gradients

🔷 8. Architecture‑Aware Distillation
• 	Cross‑Architecture KD (CNN→Transformer, ViT→CNN, etc.)
• 	Width‑Reduced KD
• 	Depth‑Reduced KD
• 	Projection‑Layer KD
• 	Bottleneck KD
• 	Sparse‑to‑Dense KD
• 	Dense‑to‑Sparse KD
• 	Quantization‑Aware KD (KD‑QAT)
• 	Pruning‑Aware KD

🔷 9. Modality‑Specific Distillation
Vision → Language
• 	CLIP Distillation
• 	Vision‑Language Alignment KD
• 	Cross‑Modal Embedding KD
Audio
• 	Spectrogram KD
• 	Waveform KD
• 	Phoneme‑Level KD
Multimodal
• 	Cross‑Modal Consistency KD
• 	Joint Embedding KD
• 	Fusion‑Layer KD

🔷 10. Optimization‑Driven Distillation
• 	Adversarial Distillation (GAN‑based KD)
• 	Contrastive KD
• 	Margin‑Based KD
• 	Mutual Information KD
• 	Entropy‑Regularized KD
• 	Teacher‑Student Adversarial Alignment
• 	Optimal Transport KD

🔷 11. Hybrid Distillation Methods
• 	KD + QAT (Quantization‑Aware Distillation)
• 	KD + PTQ (Teacher‑Guided Calibration)
• 	KD + Pruning
• 	KD + Low‑Rank Factorization
• 	KD + MoE Routing
• 	KD + Synthetic Data Generation
• 	KD + Reinforcement Learning
• 	KD + Consistency Models
"""

QUANTIZATION_METHODS = """
⚙️ Quantization — Complete Method Map
🔹 1. Post‑Training Quantization (PTQ)
• 	PTQ‑Dynamic
• 	PTQ‑Static (Calibration‑Based)
• 	PTQ‑Integer (INT8)
• 	PTQ‑FP16 / BF16
• 	PTQ‑INT4
• 	PTQ‑INT2 / Binary / Ternary
• 	GPTQ (Gradient‑based PTQ)
• 	AWQ (Activation‑Aware Weight Quantization)
• 	ZeroQuant
• 	SmoothQuant
• 	RPTQ (Round‑to‑Nearest‑Power‑of‑Two)
🔹 2. Quantization‑Aware Training (QAT)
• 	Fake‑Quantization QAT
• 	LSQ (Learned Step Size Quantization)
• 	LSQ+
• 	PACT (Parameterized Clipping Activation)
• 	DoReFa‑Net
• 	QAT‑INT8
• 	QAT‑INT4
• 	QAT‑Binary / Ternary Networks
🔹 3. Mixed‑Precision Quantization
• 	Layer‑Wise Mixed Precision
• 	Channel‑Wise Mixed Precision
• 	Token‑Wise Mixed Precision (LLMs)
• 	Hardware‑Aware Mixed Precision
• 	AutoML‑Driven Precision Search
🔹 4. Structured Quantization
• 	Blockwise Quantization (e.g., 32×32 blocks)
• 	Groupwise Quantization
• 	Row/Column Quantization
• 	Tensor‑RT Style Per‑Channel Quantization
🔹 5. Vector & Codebook Quantization
• 	Product Quantization (PQ)
• 	Residual Quantization (RQ)
• 	Additive Quantization (AQ)
• 	VQ‑VAE Quantization
• 	Codebook‑Based Weight Sharing
🔹 6. LLM‑Specific Quantization
• 	GPTQ
• 	AWQ
• 	SmoothQuant
• 	ZeroQuant
• 	LLM.int8()
• 	LLM.int4()
• 	Activation‑Outlier Suppression Quantization
• 	KV‑Cache Quantization
• 	Groupwise Quantization for Attention Blocks

🧩 Bonus: Hybrid Methods (Distillation + Quantization)
These are increasingly common in production pipelines:
• 	KD‑QAT (Distillation‑Guided Quantization‑Aware Training)
• 	KD‑PTQ (Teacher‑Guided Calibration)
• 	Feature‑Aligned QAT
• 	Logit‑Aligned QAT
• 	Consistency‑Distilled Quantization (Diffusion/LLMs)
• 	Multi‑Teacher Quantization Guidance

🔢 All Quantization Formats (Complete List)
🟦 8‑bit Formats
• 	INT8
• 	UINT8
• 	FP8‑E4M3
• 	FP8‑E5M2
• 	NF8 (Normal‑Float‑8)
• 	MXFP8 (Microsoft FP8 variant)
• 	INT8‑per‑tensor
• 	INT8‑per‑channel
• 	INT8‑per‑group
• 	INT8‑asymmetric
• 	INT8‑symmetric
• 	INT8‑dynamic
• 	INT8‑static (calibrated)
• 	LLM.int8()
• 	INT8‑KV cache quantization

🟩 6‑bit Formats
• 	INT6
• 	UINT6
• 	NF6
• 	GPTQ‑INT6
• 	Groupwise INT6
• 	Activation‑aware INT6

🟧 5‑bit Formats
• 	INT5
• 	UINT5
• 	NF5
• 	LLM.int5()
• 	Groupwise INT5
• 	Codebook 5‑bit (VQ)

🟥 4‑bit Formats
• 	INT4
• 	UINT4
• 	NF4 (NormalFloat‑4)
• 	FP4
• 	FP4‑E2M1
• 	FP4‑E3M0
• 	QLoRA NF4
• 	QLoRA FP4
• 	GPTQ‑INT4
• 	AWQ‑INT4
• 	ZeroQuant‑INT4
• 	SmoothQuant‑INT4
• 	INT4‑per‑channel
• 	INT4‑per‑group
• 	INT4‑KV cache quantization
• 	Ternary‑4 hybrid (2‑bit weights + 2‑bit scaling)

🟪 3‑bit Formats
• 	INT3
• 	UINT3
• 	NF3
• 	Ternary‑plus‑scale (3‑bit effective)
• 	Groupwise INT3
• 	Codebook 3‑bit (PQ/RQ/AQ)

⚫ 2‑bit Formats
• 	INT2
• 	UINT2
• 	Binary‑2 hybrid
• 	Ternary (−1, 0, +1)
• 	DoReFa 2‑bit
• 	XNOR‑Net 2‑bit
• 	Groupwise INT2
• 	Codebook 2‑bit

⚪ 1‑bit Formats
• 	Binary (−1, +1)
• 	XNOR‑Binary
• 	Binary‑Weight Networks (BWN)
• 	Binary‑Activation Networks
• 	XNOR‑Net
• 	Bit‑packing formats (hardware‑specific)

🧩 Floating‑Point Low‑Precision Formats
These are used heavily in NVIDIA Hopper, AMD MI300, and TPU v5e:
• 	FP16
• 	BF16
• 	FP8‑E4M3
• 	FP8‑E5M2
• 	FP6 (experimental)
• 	FP4
• 	FP3 (research)
• 	Hybrid FP8/INT8
• 	Hybrid FP8/INT4

🧱 Structured / Block Formats
These aren’t bit‑widths but quantization layouts:
• 	Blockwise INT8 (e.g., 32×32)
• 	Blockwise INT4
• 	Groupwise INT4/INT8
• 	Row‑wise quantization
• 	Column‑wise quantization
• 	Tensor‑RT per‑channel formats
• 	KV‑cache block quantization
• 	Activation‑outlier suppression formats

🧬 Vector / Codebook Formats
Used in VQ‑VAE, PQ, RQ, and LLM compression:
• 	Product Quantization (PQ)
• 	Residual Quantization (RQ)
• 	Additive Quantization (AQ)
• 	Codebook 8‑bit
• 	Codebook 4‑bit
• 	Codebook 3‑bit
• 	Codebook 2‑bit
• 	VQ‑VAE discrete latent codes

🟨 LLM‑Specific Formats (Complete)
• 	GPTQ (INT3/4/6)
• 	AWQ (INT4/INT8)
• 	SmoothQuant (INT8/INT4)
• 	ZeroQuant (INT8/INT4)
• 	QLoRA NF4
• 	QLoRA FP4
• 	LLM.int8()
• 	LLM.int4()
• 	Activation‑aware INT8
• 	KV‑cache quantization (INT8/INT4/FP8)
• 	Groupwise quantization for attention blocks
"""
