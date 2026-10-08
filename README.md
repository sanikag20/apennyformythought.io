# Data

## The Writing Archive as Data

At first glance, a personal writing archive does not look like a dataset.

There are no rows and columns.

There are no obvious labels.

There is no obvious target variable.

But text is data.

Every essay contains structure that can potentially be represented computationally:

```text
Document
│
├── Text
├── Date
├── Title
├── Length
├── Vocabulary
├── Topics
├── References
├── Themes
└── Semantic Representation
```

The purpose of this directory is to document that possibility.

---

## Current State

The repository does not currently contain a processed dataset of the complete writing archive.

This is intentional.

The live publication remains the source of truth for the writing.

Future derived datasets may be created for analytical or experimental purposes.

---

## Potential Representations

A future structured representation might look conceptually like:

```text
essay_id
title
date
text
word_count
topics
themes
embedding
```

These fields would provide a bridge between literary content and computational analysis.

---

## Potential Analyses

Once represented as structured text, the archive could support experiments involving:

### Semantic Search

Find writing based on meaning rather than exact keywords.

### Topic Modeling

Identify recurring subjects across the archive.

### Embeddings

Represent essays as vectors in a semantic space.

### Clustering

Group pieces that are conceptually similar.

### Temporal Analysis

Study how recurring themes change over time.

### Visualization

Create a visual map of relationships between pieces of writing.

---

## Important Principle

Computational representation should be treated as a secondary layer.

The dataset is derived from the writing.

The writing is not created merely to produce the dataset.

That distinction matters.

The goal is to use technology to discover interesting patterns without reducing subjective experiences to numbers.
