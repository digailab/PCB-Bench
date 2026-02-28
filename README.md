# PCB-Bench
**PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing** (ICLR 2026)

PCB-Bench is the **first comprehensive benchmark** designed to systematically evaluate (multimodal) large language models (LLMs/MLLMs) in the context of **PCB placement and routing**. It addresses the lack of standardized benchmarks and high-fidelity datasets for real-world PCB engineering reasoning by integrating **text**, **images**, and **real PCB design artifacts** into a unified evaluation framework.

---

## What's Included

PCB-Bench spans **three complementary task settings** and corresponding datasets: 

### 1) Text-based reasoning (Text-to-Text QA & CQ)
- ~**1,800** expert-written **free-form QA** instances
- Each QA has a corresponding **single-choice question (CQ)** version
- Total ~**3,700** questions (QA + CQ)
- Covers **component placement**, **routing strategies**, and **design rule compliance**
- Covers both **macro-level** (global design principles) and **micro-level** (fine-grained implementation details), across placement and routing, with topic labels (e.g., signal integrity, EMI/EMC, power planning, differential pairs, DFM, etc.).

### 2) Multimodal image-text reasoning (Image-and-Text QA/CQ)
- ~**500** problems requiring joint interpretation of **PCB layout images + technical prompts**
- Includes **choice questions**, **cloze-style fill-in-the-blank**, and **free-form QA**
- Covers visual-semantic subtasks such as component identification, functional block recognition, trace reasoning, via presence checking, differential-pair continuity analysis, etc. 

### 3) Real-world PCB design comprehension (PCB Design Understanding)
- **174** complete real-world PCB projects collected from **OSHWHub (operated by JLCPCB)**
- Each design includes artifacts such as **schematics**, **placement/routing files**, **design descriptions**, **component libraries**, and **EDA software screenshots**
- Task setting: given a **standalone EDA editor screenshot** (no extra text/schematic provided), models generate a **free-form description** of the board’s function/structure/application scenario, assessing structured visual interpretation of professional PCB artifacts. 

---

## Task Formulation

PCB-Bench is organized into three task settings aligned with real engineering workflows:

- **Task 1: Text-to-Text QA & CQ**  
  Evaluate PCB placement/routing knowledge via both open-ended generation and objective multiple-choice selection.

- **Task 2: Image-and-Text Multimodal QA & CQ**  
  Answer questions based on PCB layout images together with textual prompts.

- **Task 3: PCB Design Understanding (Screenshot-to-Description)**  
  Describe full-board PCB screenshots from EDA tools using free-form functional/structural descriptions.

---

## Evaluation Protocol

All models are evaluated under a **unified zero-shot setting** across tasks (each instance is answered independently, without demonstrations or fine-tuning). 

### Metrics
- **Choice Questions (CQ):** Top-1 **Accuracy**
- **Free-form QA:** **BERTScore** and **Sentence-BERT (SBERT) similarity** for semantic consistency with reference answers
- **Task 3 (Design Understanding):** additionally report **Precision / Recall / F1-score** to capture complementary aspects of prediction quality

---

## Models Evaluated in the Paper

The paper benchmarks a diverse set of state-of-the-art LLMs/MLLMs under the unified protocol, including frontier and open-source models; and additionally evaluates domain-specific variants derived from Qwen2.5-7B-Instruct to study PCB-oriented specialization. 

(For the exact model lists per task, please refer to the paper.)

---

## Data Sources & Licensing

- PCB designs are collected from **publicly available and legally accessible sources**, with **no proprietary or sensitive industrial data** involved.  
- Real-world PCB projects are collected from **OSHWHub**; each design is associated with a corresponding URL link to ensure transparency and IP protection.  
- PCB-Bench is released with **open licensing** to support reproducibility and standardized comparison. 

---

## Reproducibility

The paper details task formulations, metrics, and model settings. Results are obtained under the unified **zero-shot** setting. The benchmark is released **along with evaluation scripts and configuration files** to support reproduction and extension. 

---

## Citation

If you use PCB-Bench in your research, please cite:

```bibtex
@inproceedings{li2026pcbbench,
  title     = {PCB-Bench: Benchmarking LLMs for Printed Circuit Board Placement and Routing},
  author    = {Jindong Li and Lianrong Chen and Bin Yang and Jiadong Zhu and Ying Wang and Yuzhe Ma and Menglin Yang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
