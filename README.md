# Baby Dragon Hatchling

## **Bridging the Gap Between Transformers and the Brain**

**Baby Dragon Hatchling (BDH)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding how reasoning and generalization might emerge in artificial systems.

This repository contains the official implementation from the paper:
> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).


## Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

BDH formalizes a bridge between **neural computation and machine-based language understanding**. It shows how **macro reasoning behavior** in large AI models may emerge from **micro-level neuron dynamics**, guided by principles of graph theory and local computation.

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).


***

## Architecture

<img src="figs/architecture.png" width="600"/>

***

## Relation to Transformers

<img src="figs/vocab.png" width="600"/>
BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

***

## Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>
BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

***

## Abstract
The relationship between computing systems and the brain has served as motivation for pioneering theoreticians since John von Neumann and Alan Turing. 
Uniform, scale-free biological networks, such as the brain, have powerful properties, including generalizing over time, which is the main barrier for Machine Learning on the path to Universal Reasoning Models.

We introduce `Dragon Hatchling' (BDH), a new Large Language Model architecture based on a scale-free biologically inspired network of $n$ locally-interacting neuron particles. BDH couples strong theoretical foundations and inherent interpretability without sacrificing Transformer-like performance.

BDH is a practical, performant state-of-the-art 
attention-based state space sequence learning architecture. 
In addition to being a graph model, BDH admits a GPU-friendly formulation.
It exhibits Transformer-like scaling laws: we find empirically that BDH rivals GPT2-architecture Transformer performance on language and translation tasks, at the same number of parameters (10M to 1B), for the same training data.

BDH provides theoretical foundations for understanding model behavior in the limit of large size and reasoning time. 
Our results, formalized as a chain of reductions of expressiveness in the framework of computational Complexity Theory and Distributed Computing, and combined with findings on the BDH model, show a macro-to-micro correspondence of function between the general attention mechanisms in state-of-the-art Language Models, and attention mechanisms observed in the brain. These attention mechanisms formally converge as closed-form local graph dynamics at neurons and synapses: _the equations of reasoning_.

BDH can be represented as a brain model. It contains $n$ neurons, organized as an excitatory circuit and an inhibitory circuit with integrate-and-fire thresholding of input signals at neurons. The working memory of BDH during inference entirely relies on synaptic plasticity with Hebbian learning using spiking neurons, at potentiation scales of minutes for the brain (up to hundreds of tokens). We confirm empirically that specific, individual synapses strengthen connection whenever BDH hears or reasons about a specific concept while processing language inputs. The neuron interaction network of BDH is a graph of high modularity with heavy-tailed degree distribution. The BDH model is biologically plausible, explaining one possible mechanism which human neurons could use to achieve speech.

BDH is designed for interpretability. Activation vectors of BDH are sparse and positive. We demonstrate monosemanticity in BDH on language tasks, including representation of concept abstractions, which happens even for small models, below 100M-parameter scale. Interpretability of state, which goes beyond interpretability of neurons and model parameters, is an inherent feature of the BDH architecture. 

We believe BDH opens the door to a new theory of _Thermodynamic Limit_ behavior for language and reasoning models, with the ultimate goal of Probably Approximately Correct (PAC)-like bounds for generalization of reasoning over time.


## Installation and Training

```bash
# install dependencies
pip install -r requirements.txt

# train BDH on a toy dataset
python train.py
```

<!--For visualization and interpretability analysis, explore the example notebooks in `notebooks/`.-->



## Learn and discuss

- Watch the *SuperDataScience podcast* [▶️ *Dragon Hatchling: The Missing Link Between Transformers and the Brain*](https://www.youtube.com/watch?v=mfV44-mtg7c) (72 min) featuring Adrian Kosowski in conversation with Jon Krohn, unpacking BDH’s neuron-level architecture and sparse reasoning dynamics.

- Read about BDH in
[*Forbes*](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/),
[*Semafor*](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain),
[*Quantum Zeitgeist*](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/), and elsewhere in the media.

- Discuss and share the BDH paper on:
[*Alphaxiv*](https://alphaxiv.org/abs/2509.26507),
[*Hugging Face Papers*](https://huggingface.co/papers/2509.26507), 
and [*EmergentMind*](https://emergentmind.com/papers/2509.26507).

## Community forks

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.
