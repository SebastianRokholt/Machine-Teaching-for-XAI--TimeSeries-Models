# src/mllm_experiment/config.py
from dataclasses import dataclass
from pathlib import Path


TEACHING_METADATA_FILENAME = "teaching_items.csv"
EXAM_METADATA_FILENAME = "exam_items.csv"


@dataclass(slots=True)
class ExperimentConfig:
    """Store experiment-level configuration for the MLLM teaching study.

    This configuration keeps track of directory locations, model settings
    and basic design constants such as the number of items per phase.
    The script passes an instance of this class through the trial runner
    so that all components share the same configuration.

    Attributes:
        teaching_root: Root directory that contains the teaching set
            subdirectories A, B and C.
        exam_root: Root directory that contains the exam sets (set1, set2).
        metadata_root: Directory that contains the CSV metadata files.
        output_root: Directory where logs and raw responses are written.
        model_name: Name of the OpenAI model to use.
        pre_exam_items: Number of pre-teaching exam items per trial.
        post_exam_items: Number of post-teaching exam items per trial.
        teaching_items: Number of teaching items per trial.
        random_seed: Base random seed used for reproducibility. Default in None, 
        so that if the program is run multiple times with the same parameters, 
        results from one set of trials can be appended to the previous results. 
        Set to a fixed integer if running one large batch of trials at once, 
        which will also make the experiment reproducible.
    """
    teaching_root: Path
    exam_root: Path
    metadata_root: Path
    output_root: Path
    model_name: str = "gpt-5.1"
    pre_exam_items: int = 20
    post_exam_items: int = 20
    teaching_items: int = 50
    random_seed: int = None

    def ensure_output_directory(self) -> None:
        """Create the output directory if it does not yet exist.

        This method creates the output directory on disk. The teaching,
        exam and metadata directories must already exist and are not
        created here.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)