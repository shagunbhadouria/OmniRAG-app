# OmniRAG  
**Local Research & Insight System using Retrieval-Augmented Generation**

---

## Overview

**OmniRAG** is a local-first research and insight system designed to support structured exploration, reasoning, and analysis across diverse data sources using retrieval-augmented generation (RAG).

The project focuses on **practical AI system engineering**, emphasizing modular pipelines, explainable retrieval, and controlled use of language models rather than opaque, end-to-end AI workflows.

OmniRAG runs locally and supports **multiple local language models**, enabling experimentation, comparison, and analysis without mandatory cloud dependence.

---

## Core Capabilities

OmniRAG is organized around **three primary functional categories**, built on a shared retrieval and reasoning backbone.

### 1. Knowledge Explorer
- Ingests information from multiple structured and unstructured sources  
- Enables semantic exploration of documents and content  
- Supports optional **semantic search** for contextual discovery  
- Designed for traceable, evidence-backed responses  

---

### 2. AI Code Insight
- Analyzes code and related artifacts through retrieval-based context  
- Assists with understanding structure, intent, and relationships within codebases  
- Grounds insights strictly in retrieved code context rather than free-form generation  

---

### 3. Data Analyst
- Supports analytical reasoning over structured or semi-structured data  
- Uses retrieval to assemble relevant context before generating insights  
- Designed for exploratory analysis rather than automated decision-making  

---

## System Design

OmniRAG follows a **modular, pipeline-oriented design**:

- **Ingestion layer** for collecting and normalizing data  
- **Retrieval layer** for embedding-based retrieval and optional semantic search  
- **Generation layer** for producing grounded, explainable outputs  

Each component is decoupled, allowing independent evolution and experimentation.

---

## Language Model Support

OmniRAG supports **local LLM execution** through Ollama, including (but not limited to):

- Phi-3  
- Gemma  

The system allows model switching, comparative evaluation, and future extension to additional backends.  
Language models are treated as **replaceable components**, not the core of the system.

---

## Tech Stack

- **Python**
- Retrieval-Augmented Generation (RAG)
- Embedding-based retrieval & optional semantic search
- **Local LLMs via Ollama** (Phi-3, Gemma, etc.)
- API-driven backend architecture

---

## Design Principles

- **Local-first**: prioritizes user control and offline execution  
- **Explainability**: all outputs are grounded in retrieved context  
- **Modularity**: components can be extended or replaced independently  
- **System-centric**: focuses on architecture and reliability over model novelty  

---

## Status

Core ingestion, retrieval, semantic search, and multi-model local LLM integration are implemented.  
UI visuals and demonstrations will be added incrementally.

