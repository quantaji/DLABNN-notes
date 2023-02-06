# 01 Introduction
- Challenge for deep learning
    1. Continual learning. Being able to learn multiple tasks sequentially.
        1. not only "not forgetting", but also make use of previous tasks.
    2. Robust decisions. Taking into account uncertainty and errors in inputs and feedback.
    3. Explainable decisions. Understanding network reasoning and learning.
    4. Security. Shared learning on confidential data.
    5. Composable AI systems. Combining multiple systems to solve complex tasks.
    6. Un-Supervised learning of useful data representations.
        1. human brain's representation is also about usefulness, what can you do with the object
    7. Fast learning and generalization from a few data examples.

- Is human brain universal learning machine?
    - some animals have learned skills when they are born
    - but humans knows nothing, at the beginning, but have extremely better learning ability, is this a good strategy for species?
- The human brain is mostly *learned*
    - The human brain has about ``10^{11}`` neurons, and more than ``10^3`` synapses per neuron. Specifying a connection target requires about ``\log_2 10^{11} = 35`` bits/synapse. Thus it would take about ``3.5 × 10^{15}`` bits (~400 TB) to specify all ``10^{14}`` connections in the brain. 
    - however, human gnomes is about ``3\times 10^{9}`` (~1GB), only 1/4 is about brain.
- elephants have  the most heavy brains, but humans have the most amount of neurons. 
    - human brain also grows during evolution
- having a **neocortex system** is the main characteristic of mammal.
- 6-layers structure of neocortex
    - Layer 4 is considered as input layer
    - then project to layer 5, large pyramid neurons
    - the *apical* dendrite leads all the way to layer 1. layer 1 is extremely important, viewed as feedback comes in from other layers
    - pyramid cells (and apical dendrite) allow brain to have error-driven learning, (compares input and feedback). 
        - large cortex allow to do this error-driven learning in large scales of hierarchy.
    - grey matter are for processing, white matter is the transmission area.
- hierarchy of neuron, from basic edge detector, to high-level semantic features
- MCP/Perceptron: ``y=\mathbb{1}_{[\sum_i w_ix_i>0]}`` 
- There is similarity in feature similarity, for human brain and DNN.