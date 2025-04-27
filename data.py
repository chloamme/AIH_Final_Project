import json
from dataclasses import dataclass
from pathlib import Path

from base_llm import BaseLLM

from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score as bert_score
import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
rouge = Rouge()

DATA_DIR = Path(__file__).parent / "data"


class Dataset:
    def __init__(self, split: str):
        with (DATA_DIR / f"{split}.json").open() as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data

    def __getitem__(self, idx: int):
        return self.data[idx]



@dataclass
class BenchmarkResult:
    @dataclass
    class Sample:
        question: str
        answer: str  
        correct_answer: str
        bleu_score: float
        rouge_score: float
        bertscore: float

    
    answer_rate: float  # Response rate (proportion of non-empty responses)
    samples: list[Sample]
    bleu_score: float
    rouge_score: float
    bertscore: float

    @classmethod
    def from_answers(cls, answers: list[str], dataset: Dataset, max_question: int) -> "BenchmarkResult":

        samples = [
            cls.Sample(
                question=item[0], 
                answer=answer, 
                correct_answer=item[1],
                bleu_score=sentence_bleu([item[1]], answer) if answer else 0,
                rouge_score=rouge.get_scores(answer, item[1])[0]['rouge-l']['f'] if answer else 0,
                bertscore=bert_score([answer], [item[1]], lang="en")[2].item() if answer else 0,
            )
            for item, answer in zip(dataset, answers[:max_question])
        ]
        n = min(len(dataset), max_question)
        return cls(
            # Using text similarity metrics instead of accuracy
            bleu_score=sum(sample.bleu_score for sample in samples) / n,
            rouge_score=sum(sample.rouge_score for sample in samples) / n,
            bertscore=sum(sample.bertscore for sample in samples) / n,
            answer_rate=sum(bool(answer) for answer in answers[:max_question]) / n,
            samples=samples,
        )


def benchmark(func: BaseLLM, dataset: Dataset, max_question: int) -> BenchmarkResult:
    idx = range(min(len(dataset), max_question))
    questions = [dataset[i][0] for i in idx]
    answers = func.answer(*questions)
    return BenchmarkResult.from_answers(answers, dataset, max_question)


if __name__ == "__main__":
    print(Dataset("train")[0])
