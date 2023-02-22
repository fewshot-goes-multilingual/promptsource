import tqdm
from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates
from typing import Dict, Any


def verbalize_all_templates(sample: Dict[str, Any], all_templates: DatasetTemplates) -> None:
    for template_name in all_templates.all_template_names:
        template = templates[template_name]
        result = template.apply(sample)
        print("INPUT: ", result[0])
        print("OUTPUT: ", result[1])


# NER:
dataset = load_dataset("polyglot_ner", "ru")

entity_type_ru = {"PER": "человек", "LOC": "место", "ORG": "организация"}

# transform dataset into per-entity form
samples = []
for item in tqdm.tqdm(dataset["train"].select(range(10000))):  # remove the select() to obtain the whole dataset
    text = " ".join(item["words"])
    seen_entities = []
    for entity_type in ["PER", "LOC", "ORG"]:
        added_entity = None

        in_entity = False
        for word, ner in zip(item["words"], item["ner"]):
            # collect all words in a continuous chain of entity_type
            if ner == entity_type:
                if added_entity is None and not in_entity:
                    # (1) starting entity collection
                    added_entity = word
                    in_entity = True
                elif in_entity:
                    # (2) continuing entity collection
                    added_entity += " %s" % word
                elif not in_entity and added_entity is not None:
                    # (3) ambiguous entity type -> skip
                    added_entity = None
                    break
            else:
                in_entity = False

        if added_entity is not None:
            seen_entities.append({"text": text,
                                  "label_type": entity_type_ru[entity_type],
                                  "label": added_entity})

    # add all found entities of the current text
    samples.extend(seen_entities)

dataset_flat = Dataset.from_list(samples)

templates = DatasetTemplates('fewshot-goes-multilingual/ru_polyglot_ner')
verbalize_all_templates(dataset_flat[1], templates)

# NLI:
dataset = load_dataset("xnli", "ru", split="test")
templates = DatasetTemplates('fewshot-goes-multilingual/ru_xnli')
verbalize_all_templates(dataset[1], templates)

# QA:
dataset = load_dataset("sberquad", split="validation")
templates = DatasetTemplates('fewshot-goes-multilingual/ru_sberquad')
verbalize_all_templates(dataset[1], templates)

# sentiment:
# list of labels - needs to be transformed
dataset = load_dataset("cedr", "main", split="test")
dataset = dataset.filter(lambda row: len(row["labels"]) == 1)  # remove samples with ambiguous targets
dataset = dataset.map(lambda row: {"label": row["labels"][0]})  # cast to common sentiment classification format

templates = DatasetTemplates('fewshot-goes-multilingual/ru_cedr')
verbalize_all_templates(dataset[1], templates)

print("done")
