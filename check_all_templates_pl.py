from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates
from typing import Dict, Any
from utils_pl import political_advertasing_translations, kpwr_ner_translations


def verbalize_all_templates(sample: Dict[str, Any], all_templates: DatasetTemplates) -> None:
    for template_name in all_templates.all_template_names:
        template = templates[template_name]
        result = template.apply(sample)
        print("INPUT: ", result[0])
        print("OUTPUT: ", result[1])

def process_ner_dataset(dataset: Dataset, translations: Dict[str, str], output_column: str) -> Dataset:
    names = dataset.info.features[output_column].feature.names
    dataset = dataset.map(lambda x: {"tag_names": [names[i] for i in x[output_column]]})

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

    dataset_flat = Dataset.from_list(samples)
    dataset_flat = dataset_flat.map(lambda x: {"label_type_pl": translations[x["label_type"]]})
    return dataset_flat

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

# NER - political advertising
dataset = load_dataset("laugustyniak/political-advertising-pl", split="train")
dataset_processed = process_ner_dataset(dataset, political_advertasing_translations, "tags")
templates = DatasetTemplates("fewshot-goes-multilingual/pl_political-advertising-pl")

verbalize_all_templates(dataset_processed[1], templates)

# NER - generic
dataset = load_dataset("clarin-pl/kpwr-ner", split="train")
dataset_processed = process_ner_dataset(dataset, kpwr_ner_translations, "ner")

templates = DatasetTemplates("fewshot-goes-multilingual/pl_kpwr-ner")

# All classes
dataset_processed = dataset_processed.map(lambda x: {"label_type_selected": kpwr_ner_translations[x["label_type"]]})
verbalize_all_templates(dataset_processed[1], templates)

# Only generic classes
dataset_processed = dataset_processed.map(lambda x: {"label_type_selected": kpwr_ner_translations["_".join(x["label_type"].split("_")[0:2])]})
verbalize_all_templates(dataset_processed[1], templates)

print("done")
