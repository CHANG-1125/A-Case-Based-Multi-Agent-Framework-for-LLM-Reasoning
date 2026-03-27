## 4. Experimental Setup

### 4.1 Dataset and Protocol
We evaluate on GSM8K (main configuration). The training split (7,473 problems) is used only as a case memory for retrieval, while the test split is used for evaluation. To ensure comparability, all methods are run on the same 500 sampled test instances with a fixed seed (`seed=42`).

### 4.2 Methods
- **Zero-shot LLM**: direct generation without retrieval.
- **RAG baseline**: Retrieve + Reuse (Top-K case prompting, no revise agent).
- **Ours (CBR + Multi-Agent)**: Generator + Critic + Judge with gated revise and fallback voting.
- **Ours + Learned Selector**: post-hoc selector decides whether to trust RAG or Ours per instance.

### 4.3 Implementation Details
- LLM: `gpt-4o-mini` via OpenAI-compatible UniAPI endpoint.
- Retrieval encoder: `sentence-transformers/all-MiniLM-L6-v2` + FAISS.
- Top-K retrieval: `K=3`.
- Temperatures: Generator=0.3, Critic/Judge=0.0.
- Metric: exact-match numeric accuracy on final answer.

## 5. Main Results (500-test subset)

### 5.1 Quantitative Comparison

| Method | Accuracy | Correct/Total | Avg Tokens / Problem |
|---|---:|---:|---:|
| Zero-shot | 0.918 | 459/500 | 393.188 |
| RAG | **0.926** | **463/500** | 809.168 |
| Ours (initial) | 0.906 | 453/500 | 2305.046 |
| Ours (gate+vote, v5) | 0.912 | 456/500 | 1508.504 |
| Ours + Selector (LogReg, 5-fold CV) | 0.914 | 457/500 | - |
| Ours + Selector (RandomForest, 5-fold CV) | 0.926 | 463/500 | - |
| Oracle upper bound (non-deployable) | 0.950 | 475/500 | - |

**Observations.**
1. RAG is strongest among directly deployable single-system methods under this setting.
2. Ours improves from 0.906 to 0.912 after gating and conservative fallback.
3. Learned selection over RAG/Ours reaches 0.926 with RandomForest, matching RAG, while the oracle upper bound (0.950) indicates strong complementarity but is not deployable.

### 5.2 Selector Formulation
For each instance \(x\), we build feature vector \(\phi(x)\) from Ours traces (e.g., `vote_winner`, `revise_triggered`, `double_judge_used`, token-based features). The selector predicts:

\[
s(x)=\mathbb{I}(f_\theta(\phi(x)) > \tau),
\]

where \(s(x)=1\) means choosing Ours and \(s(x)=0\) means choosing RAG. Final prediction is:

\[
\hat{y}(x)=
\begin{cases}
\hat{y}_{\text{ours}}(x), & s(x)=1 \\
\hat{y}_{\text{rag}}(x), & s(x)=0
\end{cases}
\]

We report 5-fold cross-validation to estimate deployable selector performance.

### 5.3 Interpretation
The gap between RandomForest selector (0.926) and oracle selector (0.950) suggests the current feature set only partially captures when Ours should override RAG. Future gains likely depend on richer confidence features and stronger judge-consistency constraints.

---

## LaTeX Table (copy-ready)

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
Method & Accuracy & Correct/Total & Avg Tokens \\
\midrule
Zero-shot & 0.918 & 459/500 & 393.188 \\
RAG & \textbf{0.926} & \textbf{463/500} & 809.168 \\
Ours (initial) & 0.906 & 453/500 & 2305.046 \\
Ours (gate+vote, v5) & 0.912 & 456/500 & 1508.504 \\
Ours + Selector (LogReg, CV) & 0.914 & 457/500 & - \\
Ours + Selector (RandomForest, CV) & 0.926 & 463/500 & - \\
Oracle upper bound (non-deployable) & 0.950 & 475/500 & - \\
\bottomrule
\end{tabular}
\caption{Results on 500 GSM8K test instances (seed=42).}
\label{tab:main_results_500}
\end{table}
```
