
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class ProjectionLayer(nn.Module):
    """A projection layer to match hidden state dimensions between teacher and student."""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_hidden_state):
        return self.projection(student_hidden_state)

class BaseDistillationStrategy(nn.Module, ABC):
    """Base class for all distillation strategies."""
    def __init__(self, teacher_model, student_model, custom_model=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.custom_model = custom_model

    @abstractmethod
    def compute_loss(self, student_outputs, teacher_outputs, custom_outputs=None):
        pass

    def validate_models(self):
        """Validate model compatibility for the strategy. Raise ValueError if incompatible."""
        pass

class LogitKD(BaseDistillationStrategy):
    """Distillation based on matching logits."""
    def __init__(self, teacher_model, student_model, temperature=1.0):
        super().__init__(teacher_model, student_model)
        self.temperature = temperature
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')

    def compute_loss(self, student_outputs, teacher_outputs, custom_outputs=None):
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        soft_teacher_logits = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student_logits = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        return self.kd_loss(soft_student_logits, soft_teacher_logits) * (self.temperature ** 2)

class PatientKD(BaseDistillationStrategy):
    """Patient Knowledge Distillation, matching specific layers."""
    def __init__(self, teacher_model, student_model, student_layers, teacher_layers, loss_fn=F.mse_loss):
        super().__init__(teacher_model, student_model)
        self.student_layers = student_layers
        self.teacher_layers = teacher_layers
        self.loss_fn = loss_fn
        self.projections = nn.ModuleDict()
        self.validate_models()

    def validate_models(self):
        teacher_dim = self.teacher_model.config.hidden_size
        student_dim = self.student_model.config.hidden_size
        if teacher_dim != student_dim:
            for layer_idx in self.student_layers:
                self.projections[str(layer_idx)] = ProjectionLayer(student_dim, teacher_dim)

    def compute_loss(self, student_outputs, teacher_outputs, custom_outputs=None):
        total_loss = 0
        teacher_hidden_states = teacher_outputs.hidden_states
        student_hidden_states = student_outputs.hidden_states

        for student_layer_idx, teacher_layer_idx in zip(self.student_layers, self.teacher_layers):
            student_hidden_state = student_hidden_states[student_layer_idx]
            teacher_hidden_state = teacher_hidden_states[teacher_layer_idx]
            
            if str(student_layer_idx) in self.projections:
                student_hidden_state = self.projections[str(student_layer_idx)](student_hidden_state)
            
            loss = self.loss_fn(student_hidden_state, teacher_hidden_state)
            total_loss += loss
        return total_loss

class MultiTeacherKD(BaseDistillationStrategy):
    """Distillation from multiple teachers to one student."""
    def __init__(self, teacher_model, student_model, custom_model, teacher_weight=0.5, custom_weight=0.5):
        super().__init__(teacher_model, student_model, custom_model)
        self.teacher_weight = teacher_weight
        self.custom_weight = custom_weight
        self.logit_kd_teacher = LogitKD(teacher_model, student_model)
        self.logit_kd_custom = LogitKD(custom_model, student_model)

    def compute_loss(self, student_outputs, teacher_outputs, custom_outputs=None):
        teacher_loss = self.logit_kd_teacher.compute_loss(student_outputs, teacher_outputs)
        custom_loss = self.logit_kd_custom.compute_loss(student_outputs, custom_outputs)
        return (self.teacher_weight * teacher_loss) + (self.custom_weight * custom_loss)
