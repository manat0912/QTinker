
from distillation import LogitKD, PatientKD, MultiTeacherKD

# A registry for distillation strategies
STRATEGY_REGISTRY = {
    "logit_kd": LogitKD,
    "patient_kd": PatientKD,
    "multi_teacher_kd": MultiTeacherKD,
}

# A registry for model profiles with safe defaults
MODEL_PROFILE_REGISTRY = {
    "bert-base-uncased": {
        "strategy": "patient_kd",
        "patient_kd_layers": {
            "teacher_layers": [2, 4, 6, 8],
            "student_layers": [1, 2, 3, 4],
        },
    },
    "distilbert-base-uncased": {
        "strategy": "patient_kd",
        "patient_kd_layers": {
            "teacher_layers": [2, 4, 6, 8],
            "student_layers": [1, 2, 3, 4],
        },
    },
    # Default profile
    "default": {
        "strategy": "logit_kd",
    }
}

def get_strategy(strategy_name, teacher_model, student_model, custom_model=None, strategy_params=None):
    """
    Initializes and returns a distillation strategy.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class = STRATEGY_REGISTRY[strategy_name]
    
    # Pass strategy-specific parameters during initialization
    if strategy_params:
        return strategy_class(teacher_model, student_model, custom_model=custom_model, **strategy_params)
    else:
        # Fallback for strategies that don't require extra params
        if strategy_name == "multi_teacher_kd":
            if custom_model is None:
                raise ValueError("MultiTeacherKD strategy requires a custom model.")
            return strategy_class(teacher_model, student_model, custom_model)
        return strategy_class(teacher_model, student_model)


def get_profile(model_name):
    """
    Returns the profile for a given model name, or the default profile if not found.
    """
    return MODEL_PROFILE_REGISTRY.get(model_name, MODEL_PROFILE_REGISTRY["default"])

if __name__ == '__main__':
    # Example usage:
    from transformers import AutoModelForCausalLM

    # Dummy models for demonstration
    teacher = AutoModelForCausalLM.from_pretrained('bert-base-uncased')
    student = AutoModelForCausalLM.from_pretrained('prajjwal1/bert-tiny')

    profile = get_profile('bert-base-uncased')
    strategy_name = profile['strategy']
    
    strategy_params = {}
    if strategy_name == 'patient_kd':
        strategy_params = profile['patient_kd_layers']

    strategy = get_strategy(strategy_name, teacher, student, strategy_params=strategy_params)
    
    print(f"Chosen strategy for 'bert-base-uncased': {strategy.__class__.__name__}")
    if isinstance(strategy, PatientKD):
        print(f"Student layers: {strategy.student_layers}, Teacher layers: {strategy.teacher_layers}")
