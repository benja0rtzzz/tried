# Model Choices

## Generator (local LLM on MacBook)

**Decision made for:** Qwen2.5-Coder-14B.

## Judge (Azure OpenAI)

**Current choice:** o4-mini.

The judge only classifies outcomes and provides retry advice. It never generates Triton kernels.

Do not fine-tune on Azure — deployment hosting burns the $100 credit in days.