# AI Research Agent — LangGraph + Llama 3.1 8B

An autonomous research agent built with LangGraph that accepts a topic, 
plans search queries, retrieves information from Wikipedia, and synthesises 
a structured report — without any human direction of individual steps.

---

## How it works

User provides topic
↓
[Planner node] — LLM decides what to search next
↓
[Router] — resource limit hit? goal met? need more info?
↓                    ↓
[Search node]        [Synthesiser node]
↓                    ↓
[Back to planner]          END


The agent runs autonomously until either:
- The planner judges it has enough information (`RESEARCH_COMPLETE`)
- The maximum step limit is reached (resource-based termination)

---

## Example output

**Topic:** "What is transformer architecture in deep learning"

**Report:**
> The Transformer architecture is a game-changing neural network model 
> introduced in 2017 that revolutionized natural language processing. 
> Unlike traditional RNNs, Transformers utilize self-attention mechanisms 
> to process input sequences, enabling faster and more parallelizable 
> computations.
>
> Key findings: self-attention mechanism, encoder-decoder structure, 
> positional encoding, multi-head attention.

*Steps taken: 4 | Sources consulted: 3*

---

## Termination strategies implemented

Three strategies run in priority order on every iteration:

```python
def route_after_planner(state):
    # Priority 1 — Resource limit (hard ceiling, always checked first)
    if state["steps_taken"] >= state["max_steps"]:
        return "synthesiser"
    
    # Priority 2 — Goal achieved (planner said RESEARCH_COMPLETE)
    if state["status"] == "done":
        return "synthesiser"
    
    # Priority 3 — Need more information
    return "search"
```

This prevents infinite loops while allowing early termination when 
the agent has enough information.

---

## Agent state

Every node reads from and writes to a typed state dictionary:

```python
class AgentState(TypedDict):
    topic: str           # research goal
    search_results: list # accumulated Wikipedia extracts
    steps_taken: int     # current iteration count
    max_steps: int       # hard ceiling for termination
    status: str          # "running" or "done"
    report: str          # final synthesised output
    next_query: str      # search query decided by planner
```

State is the agent's memory — everything flows through it.

---

## Key engineering decisions

**Why LangGraph over LangChain AgentExecutor?**
LangGraph gives explicit control over the graph topology, routing logic, 
and termination conditions. AgentExecutor is a black box. For production 
agents, explainability and debuggability matter.

**Why direct REST API over langchain_ollama?**
`langchain_ollama` returned empty responses silently due to a version 
compatibility issue. Calling `http://localhost:11434/api/chat` directly 
bypassed the broken abstraction. General principle: when a library fails 
silently, go one layer deeper.

**Why Wikipedia over web scraping?**
Wikipedia's REST API returns clean, structured JSON with reliable 
availability. DuckDuckGo HTML scraping returned inconsistent results 
(17-82 chars) due to bot detection. For a learning project, reliable 
data quality matters more than breadth.

**Why simple search queries?**
Boolean operators (`AND`, `OR`, quotes) broke DuckDuckGo and Wikipedia 
queries. Instructing the planner to generate 3-5 word simple queries 
produced dramatically better results. Prompt engineering affects tool 
performance, not just LLM output quality.

**Why temperature=0.1?**
Agents need deterministic, structured outputs — the planner must output 
exactly `SEARCH: query` or `RESEARCH_COMPLETE`. High temperature 
introduces randomness that breaks routing logic. Unlike chatbots where 
creativity is fine, agent decisions compound across the loop — errors 
at step 2 corrupt steps 3, 4, and 5.

---

## Stack

Python 3.14 · LangGraph 1.2.2 · Llama 3.1 8B (via Ollama) · 
Wikipedia REST API · langchain-core

---

## How to run

```bash
git clone https://github.com/Pragyansh-V/hf-agent-langgraph.git
cd hf-agent-langgraph
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install and start Ollama
# Download from ollama.com
ollama pull llama3.1:8b
ollama serve

# Open notebook
jupyter notebook agent.ipynb
```

Then run all cells and call:

```python
result = run_agent("your research topic here", max_steps=4)
```

---

## What I learned building this

**Agents are not magic** — they're a typed state dictionary, a loop, 
conditional routing, and termination logic. Every "AI agent" product 
you've used is a variation of this graph.

**Tool design is half the work** — the search function went through 
three iterations (DuckDuckGo HTML → DuckDuckGo API → Wikipedia API) 
before returning reliable results. The agent logic was correct from 
the start; the tool quality determined the output quality.

**Prompt engineering for agents differs from chatbots** — agent prompts 
must produce parseable, structured outputs. The planner's strict 
FORMAT 1/FORMAT 2 instruction was the difference between a working 
agent and an infinite loop.

**Negative space matters** — knowing when NOT to continue is as 
important as knowing what to do next. The three termination strategies 
are the most important part of this codebase.

---

## Connection to prior work

- **Project 3 (RAG Pipeline)** — the search + retrieve pattern here 
  is RAG without the vector store. Wikipedia replaces FAISS.
- **AlphaAgents (MSc dissertation)** — same LangGraph framework, 
  same state machine pattern. This project deepened understanding 
  of conditional routing and termination strategies used there.

---

## Part of a learning series

- Project 1: [SST-2 Sentiment Model Audit](https://github.com/Pragyansh-V/hf-sentiment-audit)
- Project 2: [DistilBERT Emotion Classifier](https://github.com/Pragyansh-V/-hf-finetuning-distilbert)
- Project 3: [HuggingFace RAG Pipeline](https://github.com/Pragyansh-V/hf-rag-pipeline)
- Project 4: [Toxicity Bias Audit](https://github.com/Pragyansh-V/hf-bias-audit)
- Project 5: AI Research Agent ← you are here

