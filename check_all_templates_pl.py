from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates
from typing import Dict, Any


def verbalize_all_templates(sample: Dict[str, Any], all_templates: DatasetTemplates) -> None:
    for template_name in all_templates.all_template_names:
        template = templates[template_name]
        result = template.apply(sample)
        print("INPUT: ", result[0])
        print("OUTPUT: ", result[1])

# Sentiment
dataset = load_dataset("clarin-pl/polemo2-official", split="test")
templates = DatasetTemplates("fewshot-goes-multilingual/pl_polemo2-official")

verbalize_all_templates(dataset[1], templates)

# NLI
mapping = {
    "NEUTRAL": 0,
    "ENTAILMENT": 1,
    "CONTRADICTION": 2
}
dataset = load_dataset("allegro/klej-cdsc-e", split="test")
dataset = dataset.map(lambda x: {"label": int(mapping[x["entailment_judgment"]])})
templates = DatasetTemplates("fewshot-goes-multilingual/pl_klej-cdsc-e")

verbalize_all_templates(dataset[1], templates)

# NER
dataset = load_dataset("laugustyniak/political-advertising-pl", split="train")
names = dataset.info.features["tags"].feature.names
dataset = dataset.map(lambda x: {"tag_names": [names[i] for i in x["tags"]]})

samples = []
for item in dataset:
    text = " ".join(item["tokens"])
    seen_entities = []
    for idx, tag in enumerate(item["tag_names"]):
        if tag != "O":
            category = tag.split("-")[1]
            position = tag.split("-")[0]
            if position == "B":
                seen_entities.append({
                    "text": text,
                    "label_type": category,
                    "label": item["tokens"][idx],
                })
            elif position == "I":
                seen_entities[-1]["label"] += " " + item["tokens"][idx]
    # remove duplicate categories
    seen_categories = set()
    duplicated_categories = set()

    for entity in seen_entities:
        if entity["label_type"] in seen_categories:
            duplicated_categories.add(entity["label_type"])
        else:
            seen_categories.add(entity["label_type"])
    seen_entities = [e for e in seen_entities if e["label_type"] not in duplicated_categories]

    samples.extend(seen_entities)

translations = {
    'DEFENSE_AND_SECURITY': "Obronność i bezpieczeństwo",
    'EDUCATION': "Edukacja",
    'FOREIGN_POLICY': "Polityka zagraniczna",
    'HEALHCARE': "Służba zdrowia",
    'IMMIGRATION': "Imigracja",
    'INFRASTRUCTURE_AND_ENVIROMENT': "Infrastruktura i środowisko",
    'POLITICAL_AND_LEGAL_SYSTEM': "System polityczny i prawny",
    'SOCIETY': "Społeczeństwo",
    'WELFARE': "Dobrobyt",
}

dataset_flat = Dataset.from_list(samples)
dataset_flat = dataset_flat.map(lambda x: {"label_type_pl": translations[x["label_type"]]})

templates = DatasetTemplates("fewshot-goes-multilingual/pl_political-advertising-pl")

verbalize_all_templates(dataset_flat[1], templates)

print("done")
