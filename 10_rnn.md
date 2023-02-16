# 10 RNNs
- motivation
    - there is recurrence in biological brain
    - in ML, RNN used for sequence processing: video, language, speech, signal
    - theoretical view: RNNs interplay btw dynapics and computation: Chaos, turing completeness, criticality
- main advantage of RNN: memory
- morphological evidence that brain is recurrent
    - the canonical neuro circuit of L1-6, is recurrent,
    - even if there is recurrent connection, does not mean recurrence is needed
        - L2/3 is most recurrent layer, multi-patching of group of neuron shows, there can be non-recurrent neurons
    - How do we prove recurrence matters
- paper "distinct timescles of population coding across cortex" answer this question
    - they record PPC (posterior parietal cortex, high level recognition, like neocortex) and AC (auditory cortex)
    - train the mice to turn left or right depending on sound location,
    - record neural activity in AC and PPC
    - fit a generalized linear model, one is only feedforward, one with recurrent connection (coupled)
        - recurrent one does have better performance
        - but the important thing is that, performace gain is much larger in PPC (high level area)
    - some highlight: the more brain is doing recognition, the more it is feeding information from itself

## Backprop through time (BPTT)
- unrolling setup
- loss ``\mathcal{L} = \sum_t e_t = \sum_t \mathcal{L} (y_t, \hat{y}_t)``
- gradient ``\partial w = -\sum _t \partial_w \mathcal{L}_t``, and ``\partial _w e_t = \partial _{y_t} \mathcal{L}(y_t, y_t) \times \partial_{h_t} y_t \times \partial _w h_t``
- and by chain rule ``\partial_w h_t = \partial_{z_t} h_t [h_{t-1} + w\partial_w h_{t-1}] ``
- Problem 1: computing ``\partial _w h_{t}`` requries big recurrence
    - solutoin: truncated gradient
- Problem 2: vanishing/exploding gradient
    - solution: clipping gradient, good initializaiton (debatable), LSTM...
## Avoid BPTT: Reservoir Computing
- instead of having a small network trained in an intensive manner
    - we have a large network and train only last layer, by linear regression
- main idea (why it is popular): it has very natural notion of memory, in that  you can use ideas from dynamical system to understant it....?
    - ML synomous: Echo State Networks, Liquid State Machines
- Some strange metaphour... compare it with jelly, when we touch it, we want it to bounce for longer time, --> more memory
    - if there is too much water...it will break down...wth????
- in language of criticality(chaos): 
    - stable jelly: very ordered neuron response
    - chaos state: completely noise, useless
    - there are theories saying neurons should be in the edge of criticality
    - Incidentaly, the math of criticality is the same as those used in initialization of RNN to avoid exploding gradient
- in EEG you see some thing similar, called avalanche 
    - most of the event does not have long effect
    - but some of them have large effect both in time and space
    - this means brain is also in critical state.
    - teacher says it is more subcritical, near ordered phase.
- actual algorithm:
    - FORCE (sussillo, abbott 2009)
    - Conceptors (Jaeger, 2014)
    - FOLLOW (Gilra, gerstner 2017)

## some comments
- BPTT
    - brain does not perform BPTT that we know off 
    - but BPTT can be used to train networks that behave like the brain (Mante et al. 2013)
- Is reservoir computing better
    - no one thinks the brain is as simple as RC
    - but useful simple model
- RC used in finance, where you have little data, and always work out of distribution.
- resorvoir computing is usually used when data is limited, cases unable to apply BPTT

## Hopfield Networks
- motivation: adding plasticity in biologically plausible way, using learning rules in bio into math or computational model
- rules:
    - neurons are binary ``\{-1, +1\}``
    - state is updated as 
        - ``+1`` if ``\sum \mathrm{w}_{\mathrm{ji}} \mathrm{S}_{\mathrm{j}}>\theta_{\mathrm{i}}``
        - ``-1`` if otherwise
    - learning is hebbian
- memory is in the attractors of the network
    - there is some evidence somato sensory is doing something like this
- Energy: ``E = -\frac{1}{2} (\sum_{ij} s_i s_j w_{ij} + \sum_i s_i a_i)``
    - the lower energy state can recover the energy...

## Self-Organized RNN
- training
    - start with random network
    - add biological-like learning rules
    - train readout layer linearly
    - like reservoir
- only excitatory or inhibitory
- counting task: repeat same input t times, and give another input at t+1. The reservoir needs to "count"
- STDP-like learning rule: ``\Delta w_{ab} = \eta[x_b(t) - x_a(t-1)]``, ``x_b(t)`` is the post-synaptic, ``x_a(t-1)`` is the pre-synaptic
- problem: if there is three neuron, given input: "AB, AB, AB" "BC, BC, BC", and "CA CA CA"
    - then the network goes A->B->C->A forever, never ends.
- Solution: 
    - normalize weight ``w_{ij} \leftarrow \frac{w_ij}{\sum_w w_ij}``
    - set a threshold ``\Theta_i = \eta(x_i(t) - r_i)``
- Usually, it's quite hard to use simple local rules to make the network useful and stable,
    - most of time, if you add too much rules, network becomes to stable, and then useless

## Recurrence and hierarchy
- each brain area is influenced by other areas (teacher is talking about predictive coding)
    - top down connections
    - bottom up connection
- layers and recurrence
    - hypothesis: feedforward networks are enough to explain visual ctx
    - apporach: challenging images that take longer tiem to process
    - finding: at late layers, the processing is slower <=> recurrent
    - work: Kar et al. 2019, Nature Neuroscience

## LSTM
- *forget, input, gate, output* gate
- ``c_t=f \odot c_{t-1}+i \odot g=\sigma\left(W_{h f} h_{t-1}+W_{x f} x_t\right) \odot c_{t-1}+\sigma\left(W_{h i} h_{t-1}+W_{x i} x_t\right) \odot \tanh \left(W_{h g} h_{t-1}+W_{x g} x_t\right)``
- ``h_t=o \odot \tanh \left(c_t\right)=\sigma\left(W_{h h o} h_{t-1}+W_{x h o} x_t\right) \odot \tanh \left(c_t\right)``
- this simplify BPTT, uninterrepted gradient flow
- LSTM cells, some are explainable
    - reacts to position, quotation, brackets
    - but most are not interpretable

## Across-layer recurrence for learning
- previous topic is on recurrence for computation, but this can also be useful for training
- multiple implementation of learning algorithms (usually BPTT) require feedback from higher layers
    - symetric connections
    - differentiating inputs and errors
    - reusing predictive coding


