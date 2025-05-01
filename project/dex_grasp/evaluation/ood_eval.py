# TODO: Implement the OOD evaluation - this will take some example images and check if the model is able to detect poses


import os

from dex_grasp.evaluation.evaluator import Evaluator


class OOD_Evaluator(Evaluator):

    def __init__(self, ood_photos_dir, **kwargs):
        self.ood_photos_dir = ood_photos_dir
        super().__init__(**kwargs)

    def _load_data(self):
        self.ood_photos = [
            os.path.join(self.ood_photos_dir, f)
            for f in os.listdir(self.ood_photos_dir)
        ]
        # TODO: Load the images / texts from the photos taken
