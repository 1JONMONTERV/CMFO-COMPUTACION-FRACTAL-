# Known Limitations (v1.0.0)

At CMFO, we value scientific honesty over marketing hype. Here are the current known limitations of the v1.0.0 release.

## 1. Output Fidelity vs LLMs
While CMFO creates semantic vectors efficiently, the decoding layer (converting tensor state back to human text) is currently **rudimentary**. It does not yet match the fluency of GPT-4 class models for creative writing.

## 2. Hardware Optimization
The current implementation runs on Python/NumPy. While efficient for the math involved, it does not yet utilize the full parallelism of modern GPUs (CUDA/ROCm) or TPUs. **Performance is currently CPU-bound.**

## 3. Training Paradigm
CMFO does not "train" in the traditional sense. It requires a different approach to data ingestion ("Calibration"). Users familiar with `model.fit()` will need to adapt to the `resonance` paradigm.

## 4. Logit Reproduction
CMFO internal states do not map 1:1 to classical logits. Direct integration with tools expecting softmax probabilities (like some HuggingFace pipelines) requires an adapter layer that introduces approximation errors.

## 5. Ecosystem
The ecosystem of tools, plugins, and libraries is nascent compared to the mature Python AI stack. You may need to build custom connectors.
