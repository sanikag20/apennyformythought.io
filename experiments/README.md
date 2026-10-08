# Experiments

This directory is reserved for technical experiments involving the writing archive.

The experiments are intentionally separate from the live publication.

The website exists to be read.

The experiments exist to ask questions.

---

## 1. Semantic Search

### Question

Can I search my own writing by meaning rather than exact words?

For example:

> "Find something I wrote about feeling like I didn't belong."

A keyword search might fail if the word "belong" never appears.

A semantic search system could instead represent each essay as an embedding and retrieve conceptually similar writing.

### Possible Stack

```text
Python
    ↓
Text Processing
    ↓
Embedding Model
    ↓
Vector Database
    ↓
Similarity Search
    ↓
Relevant Essays
```

---

## 2. Theme Discovery

### Question

What do I actually write about?

Human memory is not necessarily a reliable index of one's own writing.

Clustering techniques could reveal recurring themes that were never explicitly assigned.

Potential outputs could include:

* thematic clusters
* representative essays
* recurring concepts
* relationships between topics

---

## 3. Writing Over Time

### Question

Does the way I write change?

A temporal analysis could examine how vocabulary, topics, sentiment, or semantic representations change across the archive.

This would turn the publication into something more than a collection of independent essays.

It would become a record of change.

---

## 4. Essay Similarity

### Question

Which pieces are more similar than they appear?

Two essays may have completely different titles and still discuss similar ideas.

Embedding-based similarity could create a network such as:

```text
Essay A ───────── Essay B
   │                 │
   │                 │
   └──── Essay C ────┘
            │
            │
         Essay D
```

This could eventually become an interactive visualization.

---

## 5. Personal Retrieval System

A longer-term experiment could combine semantic search with an interface designed specifically for the author's archive.

Instead of asking:

> "What was the title of that essay?"

the interface could allow questions such as:

> "Show me something I wrote about uncertainty."

> "What have I written about relationships?"

> "Find pieces that feel similar to this one."

The system would function as a searchable memory of the archive.

---

## Technical Philosophy

These experiments are intentionally playful.

They combine two interests that normally occupy different parts of my life:

**writing and machine learning.**

The objective is not to automate creativity.

It is to use computational tools to look at creative work from a different angle.

---

## Status

This directory currently serves as a roadmap for future work.

Experiments will be added as they are actually built.

No placeholder code is included simply to make the repository appear more complete.
