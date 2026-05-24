# Forge LoRA demo (Qwen) — held-out inference samples

Model: `Qwen/Qwen2.5-0.5B` + LoRA adapter from `lora_demo_qwen/lora_adapter`

All prompts are HELD OUT — they did **not** appear in training. Greedy decoding (`do_sample=False`), 32 new tokens.

| # | English | Base model | LoRA-tuned |
|---|---------|------------|------------|
| 1 | Good morning everyone! | Good morning, everyone! | Ahoy, ye good and true matey! |
| 2 | I will sail across the ocean. | I will sail across the ocean. | I will cut the waves, mateys! |
| 3 | Bring me some bread. | Give me some bread. | Fetch me a piece o' bread, ye barnacle! |
| 4 | The captain is angry. | "The captain is angry." | The captain be dispossessed o' the seas! |
| 5 | I am writing a letter. | I am writing a letter. | I be writing a copy, mateys! |

*Looking for pirate-speak style transfer in the LoRA column.*
