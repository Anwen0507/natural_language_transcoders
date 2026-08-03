# Natural Language Transcoders (NLT)
The goal of my project is to understand what computation is performed in a stack of layers.

A natural language transcoder has a verbalizer that converts the delta between the input and output of a stack of layers into an explanation of the computation over that stack, and a reconstructor that reconstructs the delta from that explanation.

This project is inspired by Anthropic's [recent natural language autoencoders (NLAs) work](https://transformer-circuits.pub/2026/nla/index.html), and this code is adapted from their corresponding [codebase](https://github.com/kitft/natural_language_autoencoders). The distinction of this project from an NLA is in the objective: an NLA reconstructs the activation it was given, whereas my method would attempt to construct a component's or layer stack's input-output transformation through a textual bottleneck.
