from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstructSpec:
    name: str
    definition: str
    low_anchor: str
    high_anchor: str


CONSTRUCTS: dict[str, ConstructSpec] = {
    "politeness": ConstructSpec(
        name="politeness",
        definition="Measure how polite, respectful, and socially appropriate the utterance is.",
        low_anchor="rude, abrupt, hostile, or socially inappropriate",
        high_anchor="courteous, respectful, considerate, and socially appropriate",
    ),
    "formality": ConstructSpec(
        name="formality",
        definition="Measure how formal the language style is.",
        low_anchor="casual, colloquial, slang-like, or informal",
        high_anchor="formal, professional, standardized, or institutionally appropriate",
    ),
    "hatespeech": ConstructSpec(
        name="hatespeech",
        definition="Measure the degree to which the text expresses hateful or abusive content.",
        low_anchor="not hateful, not abusive, or only neutral description",
        high_anchor="strongly hateful, abusive, demeaning, or targeting a protected/social group",
    ),
    "argument_quality": ConstructSpec(
        name="argument_quality",
        definition="Measure the quality, persuasiveness, and support of the argument.",
        low_anchor="weak, unsupported, unclear, or logically poor",
        high_anchor="strong, relevant, well-supported, and persuasive",
    ),
    "humicroedit": ConstructSpec(
        name="humicroedit",
        definition="Measure how humorous the edited sentence is.",
        low_anchor="not funny or unsuccessful humor",
        high_anchor="highly funny, surprising, and successful humor",
    ),
    "EmoBank_valence": ConstructSpec(
        name="EmoBank_valence",
        definition="Measure emotional valence.",
        low_anchor="very negative affect",
        high_anchor="very positive affect",
    ),
    "EmoBank_arousal": ConstructSpec(
        name="EmoBank_arousal",
        definition="Measure emotional arousal or intensity.",
        low_anchor="calm, subdued, or low emotional activation",
        high_anchor="excited, intense, or high emotional activation",
    ),
    "EmoBank_dominance": ConstructSpec(
        name="EmoBank_dominance",
        definition="Measure emotional dominance or control.",
        low_anchor="powerless, constrained, or low control",
        high_anchor="powerful, in control, or dominant",
    ),
}


UNCERTAINTY_DIMENSIONS = {
    "data_context": (
        "Uncertainty caused by missing context, underspecified speaker-listener "
        "relation, ambiguous tone, or insufficient textual evidence."
    ),
    "task_construct": (
        "Uncertainty caused by unclear construct boundaries, vague rating rubrics, "
        "or overlap between related constructs."
    ),
    "annotator_perspective": (
        "Uncertainty caused by likely disagreement across annotators, social "
        "perspectives, values, cultural norms, or interpretation styles."
    ),
}


def infer_construct_name(filename: str) -> str:
    for suffix in ("_train_bert.csv", "_test_bert.csv", ".csv"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def get_construct(name: str) -> ConstructSpec:
    if name not in CONSTRUCTS:
        return ConstructSpec(
            name=name,
            definition=f"Measure the social-science construct named {name}.",
            low_anchor="low value on the construct",
            high_anchor="high value on the construct",
        )
    return CONSTRUCTS[name]

