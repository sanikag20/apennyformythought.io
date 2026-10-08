# A Penny for My Thought?

### *A digital notebook at the intersection of literature, design, and technology.*

<p align="center">

**I write to understand things.
I build to understand how things work.**

</p>

---

## ✦ About

**A Penny for My Thought?** is my personal writing space.

It began with a simple question:

> *If a thought is worth something, what would mine be worth?*

The answer, inevitably, was a little more complicated than a penny.

This website is where I collect essays, observations, fragments, stories, and the occasional overthinking spiral.

The writing ranges from the mundane to the melancholic: Wednesdays, friendships, travel, growing up, leaving home, coming back, identity, uncertainty, and all the strange little things that occupy the space between them.

**The live publication lives here:**

→ [A Penny for My Thought?](https://sanikagumaste10.wixsite.com/apennyformythought/my-blog)

---

# 🧠 Why is a writing blog on GitHub?

Because I am interested in two very different ways of making sense of the world.

One involves:

```text
words
   ↓
ideas
   ↓
questions
   ↓
stories
```

The other looks more like:

```text
data
   ↓
patterns
   ↓
models
   ↓
systems
```

I happen to enjoy both.

My academic work lives in electrical engineering, data science, machine learning, and big-data systems.

My personal work lives in essays, observations, and unfinished thoughts.

This repository is an attempt to put the two worlds next to each other.

Not because writing needs technology to become meaningful.

But because **the way we build things is itself a form of expression.**

---

# 🏗️ The Digital Architecture

The website is hosted and maintained through **Wix**, which provides the underlying website infrastructure, responsive layout system, publishing environment, and blog functionality.

The architecture is therefore intentionally split between **content**, **presentation**, and **infrastructure**.

```text
                    ┌──────────────────────┐
                    │       WRITER         │
                    │                      │
                    │  Ideas • Memories    │
                    │  Observations • Life │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       CONTENT        │
                    │                      │
                    │      Essays           │
                    │      Stories          │
                    │      Fragments        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    PRESENTATION      │
                    │                      │
                    │  Typography          │
                    │  Layout              │
                    │  Navigation          │
                    │  Visual hierarchy    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      WIX CMS         │
                    │                      │
                    │ Content management   │
                    │ Dynamic pages        │
                    │ Publishing           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       READER         │
                    │                      │
                    │       You.           │
                    └──────────────────────┘
```

The GitHub repository documents this system rather than attempting to reproduce Wix's proprietary implementation.

---

# 📝 The Content Layer

The heart of the project is not the website.

It is the writing.

The current collection includes pieces such as:

| Piece                          | Theme                                           |
| ------------------------------ | ----------------------------------------------- |
| *It's Wednesday My Dudes*      | Time, routine, and the strange middle of things |
| *Went back to my Ibiza.*       | Introspection and writing without an audience   |
| *Not Like Other Girls.*        | Identity, adulthood, and self-perception        |
| *World's Smallest Violin*      | Unfinished thoughts and melancholy              |
| *Caught in a landslide*        | Travel, memory, and Sikkim                      |
| *Yours Truly, Debby Downer.*   | Sunday-night introspection                      |
| *Almost there ft. Inner Peace* | Transition and finding stability                |
| *Lost.*                        | Time, change, and the end of a year             |

The collection is deliberately heterogeneous.

There is no content strategy.

There is only curiosity.

---

# 🔬 A Different Kind of Data

As a data scientist, I spend a lot of time thinking about structured information.

Writing is almost the opposite.

A database wants:

```text
rows
columns
schemas
types
relationships
```

A thought rarely does.

A thought looks more like:

```text
             memory
                │
          ┌─────┴─────┐
          │           │
       feeling      question
          │           │
          └─────┬─────┘
                │
             story
                │
              ????
```

This blog is an exploration of that unstructured space.

---

# 🎨 Design Philosophy

The website uses a pre-designed Wix template as its visual foundation.

Rather than treating the template as the finished product, I treat it as a **design system**: a starting vocabulary of layouts, typography, navigation, spacing, and content hierarchy that can be adapted to the personality of the writing.

The design goals are simple:

### 01 — Readability

The interface should disappear when the writing begins.

### 02 — Personality

The site should feel like a notebook belonging to a person, not a content-management system.

### 03 — Continuity

Individual essays should feel like pieces of the same larger body of work.

### 04 — Restraint

Technology should support the writing rather than compete with it.

---

# 🧩 Information Architecture

```text
                    HOME
                      │
          ┌───────────┼───────────┐
          │           │           │
        ABOUT       MY BLOG     CONTACT
                      │
          ┌───────────┼───────────┐
          │           │           │
        ESSAY       ESSAY       ESSAY
          │           │           │
          ▼           ▼           ▼
       ARTICLE     ARTICLE     ARTICLE
```

The architecture intentionally keeps the number of navigation layers small.

A reader should be able to move from:

**landing page → writing → essay**

without needing to learn how the website works.

---

# 🛠️ Technology

The project currently uses:

* **Wix** — website builder, CMS, hosting, and publishing infrastructure
* **Wix Blog** — content organization and article presentation
* **HTML5-based web infrastructure** — underlying web representation
* **Git / GitHub** — project documentation and version control
* **Markdown** — technical documentation

The GitHub repository is intentionally **not a copy of Wix's generated source code**.

Wix sites rely on Wix's proprietary infrastructure and cannot simply be exported and hosted independently.

---

# 🗂️ Repository Structure

```text
a-penny-for-my-thought/
│
├── docs/
│   ├── architecture.md
│   ├── design.md
│   └── screenshots/
│
├── writing/
│   └── README.md
│
├── assets/
│   └── README.md
│
├── README.md
├── LICENSE
└── .gitignore
```

### `docs/`

Technical documentation of the website's architecture and design.

### `writing/`

Documentation of the editorial and literary side of the project.

### `assets/`

Supporting visual assets and documentation.

---

# 🚀 Viewing the Project

The website is available here:

**[A Penny for My Thought?](https://sanikagumaste10.wixsite.com/apennyformythought/my-blog)**

The GitHub repository serves as the technical companion to the publication.

---

# 🔭 Future Directions

This project is deliberately unfinished.

Potential future iterations include:

```text
                         CURRENT
                            │
                            ▼
                    ┌──────────────┐
                    │    Wix Blog  │
                    └──────┬───────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        Technical Docs          Literary Archive
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Future Site │
                    └──────────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Search        Analytics      Custom
          & Tags                       Backend
```

Possible extensions:

* [ ] Build a custom version of the site from scratch
* [ ] Introduce a structured content model
* [ ] Add semantic search across essays
* [ ] Build a personal recommendation system for readers
* [ ] Create an NLP-based exploration of recurring themes
* [ ] Visualize changes in writing themes over time
* [ ] Experiment with embeddings and semantic similarity
* [ ] Add an interactive "find me a thought" feature
* [ ] Build a lightweight content API
* [ ] Develop a custom frontend independent of the current CMS

The eventual goal would not be to replace the writing with technology.

It would be to use technology to **see the writing differently**.

---

# ✦ The Larger Idea

I spend my academic life asking machines to find patterns in data.

I spend my personal life trying to find patterns in myself.

Perhaps those aren't such different activities after all.

One produces models.

The other produces stories.

Both begin with the same thing:

**paying attention.**

---

## About the Author

**Sanika Gumaste** is an electrical engineering graduate student at Columbia University working at the intersection of data science, machine learning, and AI.

She also writes things that have absolutely nothing to do with any of those.

Sometimes, that's the point.

---

<p align="center">

### A penny for my thought?

**A penny for mine.**

</p>
