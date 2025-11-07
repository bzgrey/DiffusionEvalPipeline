import sys
from pathlib import Path

# Add parent directory to path so we can import evalPipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from evalPipeline import dice_score, f1_score


def test_dice_perfect_match():
    pred = torch.ones((1, 1, 4, 4))
    target = torch.ones((1, 1, 4, 4))
    assert pytest.approx(dice_score(pred, target), rel=1e-6) == 1.0


def test_dice_no_overlap():
    pred = torch.zeros((1, 1, 4, 4))
    target = torch.ones((1, 1, 4, 4))
    # Due to smoothing factor, result will be very close to 0 but not exactly 0
    result = dice_score(pred, target)
    assert result < 1e-6


def test_dice_partial_overlap():
    pred = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    pred[0, 0, 0, 0] = 1
    pred[0, 0, 0, 1] = 1
    target[0, 0, 0, 0] = 1
    # intersection = 1, sum_pred = 2, sum_tgt = 1 -> dice = 2*1/(2+1)=0.6666
    assert pytest.approx(dice_score(pred, target), rel=1e-6) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_f1_perfect():
    preds = torch.tensor([1, 0, 1, 1])
    targets = torch.tensor([1, 0, 1, 1])
    assert pytest.approx(f1_score(preds, targets), rel=1e-6) == 1.0


def test_f1_no_true_positives():
    preds = torch.tensor([0, 0, 0, 0])
    targets = torch.tensor([1, 1, 1, 1])
    assert pytest.approx(f1_score(preds, targets), rel=1e-6) == 0.0


def test_f1_partial():
    preds = torch.tensor([1, 1, 0, 0])
    targets = torch.tensor([1, 0, 1, 0])
    # TP=1, FP=1, FN=1 => precision=1/(1+1)=0.5, recall=1/(1+1)=0.5 => f1=0.5
    # Due to smoothing factor (1e-6), result will be very close to 0.5
    result = f1_score(preds, targets)
    assert pytest.approx(result, abs=1e-5) == 0.5
