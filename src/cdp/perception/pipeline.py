from .factors import select_semantic_factors
from .grounding import build_runtime, ground_images
from .instruction_parser import parse_instruction


class SemanticPerception:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.runtime = None

    def __call__(self, images, instruction, budget):
        candidates = parse_instruction(instruction)
        if not candidates:
            return []
        if self.runtime is None:
            self.runtime = build_runtime(self.config, self.device)
        observations = ground_images(images, candidates, self.config, self.runtime)
        factors = []
        for index, candidate in enumerate(candidates):
            factor = candidate.to_dict()
            factor.update(
                view_masks=[view["masks"][index] for view in observations],
                view_valid=[bool(view["mask_valid"][index]) for view in observations],
                dino_scores=[float(view["dino_scores"][index]) for view in observations],
                sam_scores=[float(view["sam_scores"][index]) for view in observations],
            )
            factors.append(factor)
        return select_semantic_factors(factors, budget)
