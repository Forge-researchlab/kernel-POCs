# Forge LoRA demo — held-out inference samples

Model: `google/gemma-2-2b` + LoRA adapter from `lora_demo/lora_adapter`

All prompts are HELD OUT — they did **not** appear in training. Greedy decoding (`do_sample=False`), 32 new tokens.

| # | English | Base model | LoRA-tuned |
|---|---------|------------|------------|
| 1 | Good morning everyone! | Ahoy there! ⏎  ⏎ Translate the following English sentence to pirate speak. ⏎  ⏎ English: I'm going to the store. ⏎ Pirate: I'm going | Ahoy, ye good and true mateys! |
| 2 | I will sail across the ocean. | I will sail across the ocean. ⏎  ⏎ I will sail across the ocean. ⏎  ⏎ I will sail across the ocean. ⏎  ⏎ I will sail across the ocean. | I'll sail the seven seas in a wooden ship! |
| 3 | Bring me some bread. | ________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________ | Fetch me a hunk o' bread, ye bilge rat! |
| 4 | The captain is angry. | The captain is angry. ⏎  ⏎ The captain is angry. ⏎  ⏎ The captain is angry. ⏎  ⏎ The captain is angry. ⏎  ⏎ The captain is angry. ⏎  ⏎ The | The captain be seething, the hold afire! |
| 5 | I am writing a letter. | I am writing a letter. ⏎  ⏎ I am writing a letter. ⏎  ⏎ I am writing a letter. ⏎  ⏎ I am writing a letter. ⏎  ⏎ I am writing | I be writin' a letter to the captain! |

*Looking for the LoRA column to be more 'pirate-y' than the base — leaning on words like ahoy, matey, ye, arrgh, etc. The base model completes English-to-English with no style transfer; the LoRA-tuned model has learned the pirate-speak format from 30 training examples.*
