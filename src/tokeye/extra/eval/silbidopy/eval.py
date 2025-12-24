from torchmetrics.segmentation import (
    GeneralizedDiceScore,
    HausdorffDistance,
    MeanIoU,
)
from torchmetrics.classification import (
    Precision,
    Recall,
)

class Metrics:
    def __init__(self, device="cpu"):
        self.generalized_dice_score = GeneralizedDiceScore(
            num_classes=2)
        self.hausdorff_distance = HausdorffDistance(
            num_classes=1, include_background=True
        )
        self.iou = MeanIoU(
            num_classes=1, include_background=True
        )
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")

        self.scores = {
            "generalized_dice_score": self.generalized_dice_score,
            "hausdorff_distance": self.hausdorff_distance,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
        }

        if device == "cuda":
            for score in self.scores.values():
                score.to(device)

    def update(self, output, ann):
        for score in self.scores.values():
            score.update(output, ann)

    def compute(self):
        return {
            key: score.compute() 
            for key, score 
            in self.scores.items()
        }

    def __repr__(self):
        return "\n".join(
            [f"{key}: {score.compute()}" 
            for key, score 
            in self.scores.items()]
            )
    
    def __call__(self, output, ann):
        scores = {}
        for key, score in self.scores.items():
            scores[key] = score(output, ann)
        return scores