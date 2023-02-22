from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates
from typing import Dict, Any


def verbalize_all_templates(sample: Dict[str, Any], all_templates: DatasetTemplates) -> None:
    for template_name in all_templates.all_template_names:
        template = templates[template_name]
        result = template.apply(sample)
        print("INPUT: ", result[0])
        print("OUTPUT: ", result[1])


# CNEC:
dataset = load_dataset("fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0", split="train")

# transform dataset into per-entity form
samples = []
for item in dataset:
    text, entities = item.values()
    seen_types = set()
    seen_entities = []
    for entity in entities:
        if entity["category_str"] in seen_types:
            # ambiguous entity type -> rollback addition of all samples of this type
            seen_entities = [e for e in seen_entities if e["label_type"] != entity["category_str"]]
        else:
            seen_entities.append({"text": text,
                                  "label_type": entity["category_str"],
                                  "label": entity["content"]})
            seen_types.add(entity["category_str"])

    # add all found entities of the current text
    samples.extend(seen_entities)

dataset_flat = Dataset.from_list(samples)

templates = DatasetTemplates('fewshot-goes-multilingual/cs_czech-named-entity-corpus_2.0')
verbalize_all_templates(dataset_flat[1], templates)

# CSFD:
dataset = load_dataset("fewshot-goes-multilingual/cs_csfd-movie-reviews", split="train")
dataset = dataset.filter(lambda x: x["rating_int"] != 3)
dataset = dataset.map(lambda x: {"label": 1 if x["rating_int"] > 3 else 0})

templates = DatasetTemplates('fewshot-goes-multilingual/cs_csfd-movie-reviews')
verbalize_all_templates(dataset[1], templates)

# FB-comments:
dataset = load_dataset("fewshot-goes-multilingual/cs_facebook-comments", split="train")
dataset = dataset.map(lambda x: {"label": x["sentiment_int"] + 1})

templates = DatasetTemplates('fewshot-goes-multilingual/cs_facebook-comments')
verbalize_all_templates(dataset[1], templates)

# MALL-reviews:
dataset = load_dataset("fewshot-goes-multilingual/cs_mall-product-reviews", split="train")
dataset = dataset.map(lambda x: {"label": x["rating_int"] + 1})

templates = DatasetTemplates('fewshot-goes-multilingual/cs_facebook-comments')
verbalize_all_templates(dataset[1], templates)

# SQAD:
dataset = load_dataset("fewshot-goes-multilingual/cs_squad-3.0", split="train")

templates = DatasetTemplates('fewshot-goes-multilingual/cs_squad-3.0')
verbalize_all_templates(dataset[1], templates)

# NLI:
dataset = load_dataset("ctu-aic/ctkfacts_nli", split="train")

templates = DatasetTemplates('fewshot-goes-multilingual/cs_ctkfacts_nli')
verbalize_all_templates(dataset[1], templates)
