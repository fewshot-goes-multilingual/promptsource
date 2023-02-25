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

print("done")
