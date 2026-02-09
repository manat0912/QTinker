
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def load_model_and_tokenizer(model_name_or_path, model_type):
    """Loads a model and tokenizer from a given path or name."""
    # Handle file paths (e.g. from file picker) by using parent directory
    if os.path.isfile(model_name_or_path):
        print(f"Provided path is a file. Using parent directory: {os.path.dirname(model_name_or_path)}")
        model_name_or_path = os.path.dirname(model_name_or_path)

    # Simple check if it's a local path
    if os.path.isdir(model_name_or_path):
        print(f"Loading {model_type} from local path: {model_name_or_path}")
    else:
        print(f"Loading {model_type} from Hugging Face Hub: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Check config to decide model class
    from transformers import AutoConfig, AutoModelForMaskedLM
    config = AutoConfig.from_pretrained(model_name_or_path)

    from_tf = False
    if os.path.isdir(model_name_or_path):
        has_pt = os.path.exists(os.path.join(model_name_or_path, "pytorch_model.bin")) or os.path.exists(os.path.join(model_name_or_path, "model.safetensors"))
        has_tf = os.path.exists(os.path.join(model_name_or_path, "bert_model.ckpt.index")) or os.path.exists(os.path.join(model_name_or_path, "tf_model.h5"))
        if not has_pt and has_tf:
            from_tf = True
            print(f"Detected TensorFlow checkpoint. Loading with from_tf=True.")
    
    if 'bert' in config.model_type or 'roberta' in config.model_type or 'distilbert' in config.model_type:
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, from_tf=from_tf)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, from_tf=from_tf)
    
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

class ModelLoader:
    """Handles loading of teacher, student, and custom models."""
    def __init__(self, teacher_path, student_path, custom_path=None):
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.custom_path = custom_path

        self.teacher_model, self.teacher_tokenizer = None, None
        self.student_model, self.student_tokenizer = None, None
        self.custom_model, self.custom_tokenizer = None, None

    def load_all(self):
        """Load all specified models and tokenizers."""
        print("Loading models...")
        self.teacher_model, self.teacher_tokenizer = load_model_and_tokenizer(self.teacher_path, "teacher")
        self.student_model, self.student_tokenizer = load_model_and_tokenizer(self.student_path, "student")
        if self.custom_path:
            self.custom_model, self.custom_tokenizer = load_model_and_tokenizer(self.custom_path, "custom")
        print("All models loaded successfully.")

    def get_models(self):
        return {
            "teacher": (self.teacher_model, self.teacher_tokenizer),
            "student": (self.student_model, self.student_tokenizer),
            "custom": (self.custom_model, self.custom_tokenizer)
        }

if __name__ == '__main__':
    # Example usage:
    # Replace with actual model names or paths
    # For example, a larger BERT as teacher and a smaller one as student
    teacher = 'bert-base-uncased'
    student = 'prajjwal1/bert-tiny'

    loader = ModelLoader(teacher_path=teacher, student_path=student)
    loader.load_all()
    models = loader.get_models()

    print(f'Teacher model config: {models["teacher"][0].config.to_dict()["model_type"]}')
    print(f'Student model config: {models["student"][0].config.to_dict()["model_type"]}')

    # Example with custom model
    custom = 'prajjwal1/bert-mini'
    loader_with_custom = ModelLoader(teacher_path=teacher, student_path=student, custom_path=custom)
    loader_with_custom.load_all()
    models_with_custom = loader_with_custom.get_models()
    print(f'Custom model config: {models_with_custom["custom"][0].config.to_dict()["model_type"]}')
