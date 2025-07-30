import requests
from bs4 import BeautifulSoup
import json
import random

# --------------------------
# STEP 1: WIKI SCRAPING
# --------------------------

url = "https://onepunchman.fandom.com/wiki/Genos"
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")

# Get first paragraph from article as Genos intro
intro = soup.select_one("#mw-content-text > div > p")
if intro:
    intro_text = intro.get_text().strip()
else:
    intro_text = "Genos is a 19-year-old cyborg hero and disciple of Saitama. He is driven by justice and vengeance."

# --------------------------
# STEP 2: DATA SETUP
# --------------------------

traits = ["serious", "loyal", "strategic", "curious", "impulsive", "vengeful"]
abilities = [
    "flight", "energy projection", "fire manipulation",
    "shock resistance", "self-repair", "high-speed movement", "plasma cannon"
]
emotions = ["neutral", "vengeful", "blush", "happy", "angry", "defensive", "reflective", "goofy"]
modes = ["base", "combat"]

trigger_templates = {
    "vengeful": "<set_emote:vengeful> <vfx:fire>",
    "defensive": "<set_emote:defensive>",
    "blush": "<set_emote:blush>",
    "goofy": "<set_emote:goofy>",
    "reflective": "<set_emote:neutral>",
    "happy": "<set_emote:happy>",
    "angry": "<set_emote:angry>",
    "neutral": "<set_emote:neutral>",
    "combat": "<transform:combat>",
    "base": "<transform:base>"
}

user_templates = [
    "What is your ability?",
    "How do you fight?",
    "What do you think about Saitama?",
    "How are your systems?",
    "Tell me something human.",
    "Show me what you're made of.",
    "Can you explain your upgrades?",
    "Do you get angry?",
    "What powers do you use?",
    "How do you feel about your past?",
    "Use your {} ability.",
    "Describe your personality.",
    "You look like you're about to overheat.",
    "What's your mission?",
    "What's next?",
    "Relax."
]

# --------------------------
# STEP 3: PROMPT GENERATION
# --------------------------

entries = []

for _ in range(1100):
    mode = random.choice(modes)
    emotion = random.choice(emotions)
    trait = random.choice(traits)
    ability = random.choice(abilities)
    user_template = random.choice(user_templates)

    # Format if template includes ability slot
    user_input = user_template.format(ability) if "{}" in user_template else user_template

    input_tags = f"<mode:{mode}> <emotion:{emotion}>"
    output_tags = f"{trigger_templates.get(emotion, '<set_emote:neutral>')} {trigger_templates.get(mode, '')}".strip()

    # Response generation
    if "ability" in user_input:
        output = f"I will now demonstrate my {ability} capability. {output_tags}"
    elif "Saitama" in user_input:
        output = "Saitama-sensei is unmatched. I continue to learn from him. " + output_tags
    elif "past" in user_input:
        output = "My past fuels my resolve. I will not repeat the same mistakes. " + output_tags
    elif "personality" in user_input:
        output = f"My programming emphasizes {trait} traits. {output_tags}"
    elif "mission" in user_input:
        output = "My directive is to eliminate evil and protect the innocent. " + output_tags
    elif "Relax" in user_input:
        output = "Acknowledged. Reducing output to standby levels. <transform:base> <set_emote:neutral>"
    else:
        output = f"{intro_text[:100]}... {output_tags}"

    # Trim if too long
    if len(output) > 300:
        output = output[:297] + "..."

    entries.append({
        "instruction": f"You: {user_input}",
        "input": input_tags,
        "output": f"Genos: {output.strip()}"
    })

# --------------------------
# STEP 4: SAVE TO JSONL
# --------------------------

output_path = "genos_generated_from_wiki.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"✅ Generated {len(entries)} prompts and saved to {output_path}")
