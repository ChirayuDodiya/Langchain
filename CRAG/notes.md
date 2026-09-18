# Corrective Retrieval Augmented Generation (CRAG)

## Overview
CRAG enhances standard RAG by evaluating the quality of retrieved documents before generation. Based on this evaluation, it categorizes the retrieved context and applies specific strategies to refine or supplement the knowledge.

## Retrieval Evaluation Categorization
After retrieving documents, an evaluator assesses their relevance and assigns a confidence score.
- **Lower Bound (LB) Threshold:** 0.3
- **Upper Bound (UB) Threshold:** 0.7

Based on these thresholds, the retrieval is classified as:
1. **Correct** (At least one document > 0.7): Triggers **Knowledge Refinement**.
2. **Incorrect** (All documents < 0.3): Triggers **Knowledge Searching**.
3. **Ambiguous** (Mixed scores in between): Triggers both **Knowledge Refinement** and **Knowledge Searching**.

*Note: Moving forward in the pipeline, the system only uses documents that scored above the lower bound (> 0.3).*

## Core Mechanisms

### 1. Knowledge Refinement (Internal Knowledge)
This process extracts only the most precise information from the retrieved documents, discarding the noise.
- **Decomposition:** Break all the filtered document chunks down into smaller, individual sentences.
- **Evaluation:** Evaluate each sentence independently to determine if it is strictly related to the user's query.
- **Filtering:** Keep the relevant sentences and discard the irrelevant ones. Only this highly condensed, relevant context is passed to the generator.

### 2. Knowledge Searching (External Knowledge)
This process supplements the system with external information when the internal retrieved documents are insufficient.
- **Web Search:** Execute a web search to gather new external documents.
- **Refinement:** Apply the exact same **Knowledge Refinement** process to the web search results (break into sentences, evaluate, and filter) to ensure only relevant external information is passed to the generator.