"""Inline ~30 pirate-speak translation pairs for the Forge LoRA demo.

Small enough that the model can learn the style with LoRA in ~200 steps;
diverse enough that improvement is visible on held-out prompts post-training.
"""
TRAIN_PAIRS = [
    ("Hello, friend!", "Ahoy, matey!"),
    ("Where is the treasure?", "Where be the booty, ye scallywag?"),
    ("I am very hungry.", "Me belly be emptier than a sunken ship, arrgh!"),
    ("The weather is nice today.", "The weather be fair fer sailin', aye!"),
    ("Please pass the salt.", "Hand over the salt, ye landlubber!"),
    ("I love reading books.", "Aye, I be lovin' a good yarn at sea!"),
    ("Do you want some coffee?", "Care fer a swig o' bitter brew?"),
    ("My ship is fast.", "Me vessel cuts the waves like a cutlass!"),
    ("I need a doctor.", "Fetch me a sawbones, I be wounded!"),
    ("The map shows the way.", "The chart points the course to riches!"),
    ("Let us go home.", "Hoist anchor, we be sailin' back to port!"),
    ("This is my best work.", "This be me finest plunder, arrgh!"),
    ("I am tired today.", "Me bones be weary, mateys!"),
    ("Look at the stars.", "Cast yer eye on the heavens above!"),
    ("Bring me my hat.", "Fetch me tricorn, ye bilge rat!"),
    ("The wind is strong.", "The gales be fierce as a kraken's roar!"),
    ("I have a secret.", "I keep a tale none shall know, ye dog!"),
    ("Drink your water.", "Down yer grog and be quick about it!"),
    ("Where are you going?", "Whither do ye sail, scurvy dog?"),
    ("I found a coin.", "I plundered meself a piece o' eight!"),
    ("The night is dark.", "The night be black as pitch on the high seas!"),
    ("Help me carry this.", "Lend a hand with this cargo, savvy?"),
    ("My friend is brave.", "Me hearty be the bravest soul afloat!"),
    ("Open the window.", "Pry open the porthole, ye barnacle!"),
    ("I am thinking.", "I be ponderin' the depths, mateys!"),
    ("The food is good.", "This grub be fit fer a captain!"),
    ("Stop the noise.", "Belay that racket, ye swabbies!"),
    ("Tell me a story.", "Spin me a yarn o' the deep, friend!"),
    ("Sleep well tonight.", "May yer hammock rock ye gentle to sleep!"),
    ("Thank you very much.", "Much obliged, ye good and true matey!"),
]

# Held-out prompts (NOT in training set) — used post-training for inference demo.
HELD_OUT_PROMPTS = [
    "Good morning everyone!",
    "I will sail across the ocean.",
    "Bring me some bread.",
    "The captain is angry.",
    "I am writing a letter.",
]


PROMPT_TEMPLATE = (
    "Translate the following English sentence to pirate speak.\n\n"
    "English: {english}\n"
    "Pirate:"
)


def build_prompt(english: str) -> str:
    return PROMPT_TEMPLATE.format(english=english)


def build_full_example(english: str, pirate: str) -> tuple[str, str]:
    """Return (prompt, completion) where completion includes a leading space
    and the model's EOS will be appended by the tokenizer call."""
    prompt = build_prompt(english)
    completion = f" {pirate}"
    return prompt, completion
