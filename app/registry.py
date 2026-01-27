from distillation import (
    LogitKD, PatientKD, MultiTeacherKD,
    HardLabelKD, FeatureMatchingKD, CosineSimilarityKD,
    AttentionDistillationKD,
    # Add stubs for all other strategies below
)

# A registry for distillation strategies
STRATEGY_REGISTRY = {
    "logit_kd": LogitKD,
    "patient_kd": PatientKD,
    "multi_teacher_kd": MultiTeacherKD,
    "hard_label_kd": HardLabelKD,
    "feature_matching_kd": FeatureMatchingKD,
    "cosine_similarity_kd": CosineSimilarityKD,
    "attention_distillation_kd": AttentionDistillationKD,
    # --- Add all other strategies as stubs ---
    "soft_target_kd": LogitKD,  # Alias for soft targets
    "temperature_scaled_kd": LogitKD,  # Alias for temperature scaling
    "kullback_leibler_kd": LogitKD,  # Alias for KLDiv
    "cross_entropy_kd": HardLabelKD,  # Alias for hard label
    "label_smoothing_kd": HardLabelKD,  # Alias for hard label
    "confidence_penalty_kd": LogitKD,  # Alias for logit penalty
    "dark_knowledge_kd": LogitKD,
    "response_consistency_kd": LogitKD,
    "multi_teacher_logit_averaging": MultiTeacherKD,
    "gated_logit_fusion_kd": MultiTeacherKD,
    "mixture_of_experts_logit_kd": MultiTeacherKD,
    "feature_map_matching_kd": FeatureMatchingKD,
    "activation_matching_kd": FeatureMatchingKD,
    "hidden_state_distillation_kd": FeatureMatchingKD,
    "attention_map_distillation_kd": AttentionDistillationKD,
    "transformer_attention_head_distillation": AttentionDistillationKD,
    "fitnets_hint_based_distillation": FeatureMatchingKD,
    "neuron_selectivity_transfer_kd": FeatureMatchingKD,
    "similarity_preserving_kd": FeatureMatchingKD,
    "correlation_congruence_kd": FeatureMatchingKD,
    "relational_kd": FeatureMatchingKD,
    "distance_wise_rkd": FeatureMatchingKD,
    "angle_wise_rkd": FeatureMatchingKD,
    "contrastive_representation_kd": CosineSimilarityKD,
    "gram_matrix_distillation": FeatureMatchingKD,
    "jacobian_matching": FeatureMatchingKD,
    "layer_to_layer_projection_kd": FeatureMatchingKD,
    "cross_architecture_feature_alignment": FeatureMatchingKD,
    # ... (add more as needed)
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
