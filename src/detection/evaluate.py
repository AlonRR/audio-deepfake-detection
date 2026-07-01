"""Detection evaluation — EER / min t-DCF / DET curve + the cross-generator test.

TODO (implementation pass):
    - eer(scores, labels) -> float                  # equal error rate
    - min_tdcf(scores, labels, asv_scores) -> float # ASVspoof tandem cost
    - det_curve(scores, labels): save DET plot to reports/
    - cross_test(model, own_clips): EER of the detector on my A1/A2 fakes
                                    (unseen 2025-era synthesizer) vs ASVspoof/In-the-Wild
The cross_test is the project's own contribution (see docs/proposal.md § Innovation).
"""


def eer(scores, labels):
    """Equal Error Rate from detection scores + binary labels. TODO: implement."""
    raise NotImplementedError
