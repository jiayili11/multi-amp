# MultiAMP: Multi-Stream Deep Learning for Antimicrobial Peptide Prediction


Antimicrobial resistance (AMR) is accelerating worldwide, undermining frontline antibiotics and making the need for novel agents more urgent than ever.
Antimicrobial peptides (AMPs) are promising therapeutics against multidrug-resistant pathogens, as they are less prone to inducing resistance.
However, current AMP prediction approaches often treat sequence and structure in isolation and at a single scale, leading to mediocre performance. Here, we propose MultiAMP, a framework that integrates multi-level information for predicting AMPs. The model captures evolutionary and contextual information from sequences alongside global and fine-grained information from structures, synergistically combining these features to enhance predictive power.
MultiAMP achieves state-of-the-art performance, outperforming existing AMP prediction methods by over 10\% in MCC when identifying distant AMPs sharing less than 40\% sequence identity with known AMPs. 
To discover novel AMPs, we applied MultiAMP to marine organism data, discovering 484 high-confidence peptides with sequences that are highly divergent from known AMPs. Notably, MultiAMP accurately recognizes various structural types of peptides. 
In addition, our approach reveals functional patterns of AMPs, providing interpretable insights into their mechanisms. Building on these findings, we further employed a gradient-based strategy and achieved the design of AMPs with specific motifs.
We believe that MultiAMP empowers both the rational discovery and mechanistic understanding of AMPs, facilitating future experimental validation and precision therapeutic design.
