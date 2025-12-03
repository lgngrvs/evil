# Evil

This is a research repository building on the literature in emergent misalignment. We are particularly grateful to Soligo and Turner for their fantastic open-source models and high-quality research.

Currently, `evil` can: 
- Replicate the convergent misalignment direction discovered in [Soligo et al. 2025](https://arxiv.org/pdf/2506.11618) in Qwen 2.5 7B Instruct
- Given sparse autoencoders for Qwen 2.5 7B, identify SAE features most changed when steering with that misalignment direction and use an LLM to label them

Next steps: write code to run feature geometry-related analysis for the SAE features to understand the structure of the activation spaces better.

Everything can be run using the Jupyter Notebook in `colab.ipynb`. In the future the scripts can be taken out of the Colab but at the moment this is the best way for us to access compute.
