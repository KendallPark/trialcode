# Team meeting notes (2026-11-06)

Summary as pasted by the user; not verbatim.

The meeting focuses on a research framework designed to emulate randomized clinical trials using observational data, such as Electronic Health Records (EHR). The primary goal is to automate the process of mapping trial protocols (inclusion/exclusion criteria, interventions, outcomes) to real-world healthcare codes like ICD-10 and RxNorm.

## The Core Challenge
- While LLMs perform well on simple queries, they consistently struggle to translate complex clinical instructions (e.g., "Type 1 or Type 2 diabetes with end-organ damage") into accurate ICD or RxNorm codes.
- The team previously added a deterministic verifier agent to the pipeline, which reduced hallucinations by checking codes against a database and flagging invalid ones, but it cannot generate the correct codes on its own.
- Implementing Retrieval-Augmented Generation (RAG) improved code generation accuracy, but performance is still not at the required level.

## Proposed AI Solutions
- Train or fine-tune a small, specialized model with deep "intuitions" and associations regarding ICD-10 codes and clinical terminology.
- Two main training approaches debated: supervised fine-tuning vs. unsupervised continued pre-training.
- Also discussed: parameter-efficient fine-tuning (e.g., LoRA) on a larger model, or training a tiny model (e.g., 128M parameters) that could run locally on a laptop.
- Ultimate vision: the main AI agent queries this smaller, bespoke model to accurately link clinical text to the correct medical codes.

## Data Constraints and Privacy
- Acquiring full patient charts or clinical notes is highly unrealistic due to hospital privacy policies and HIPAA.
- MIMIC was suggested as a training source for a proof-of-concept, though it is limited because it primarily contains ICU patient data.
- To avoid privacy hurdles, train the proof-of-concept using only de-identified bags of ICD-10 codes without any attached patient text.
- Ongoing engineering/security challenge: how to run AI agents and Python SQL scripts inside secure hospital cloud environments without exposing patient data to the AI.

## Action Items and Next Steps
- **Build a benchmark:** pull existing clinical trials (from PubMed or projects like the Harvard duplicate project) and extract the specific ICD-10 codes used in their supplementary data.
- **Model evaluation:** use the current set of 40 inclusion-criteria prompts to evaluate how well models generate the correct codes.
- **Proof-of-concept:** one speaker will attempt to train a tiny model locally on a laptop to see if it learns associations between medical codes.
- **Security meeting:** follow-up next week (including a colleague named Son) on data privacy and the mechanics of running the agent securely.
