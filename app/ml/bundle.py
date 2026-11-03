"""One place that constructs the optional ML models for an application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.ml.base import ModelInfo
from app.ml.differential_logreg import DifferentialLogRegModel
from app.ml.drug_resolver import DrugResolver
from app.ml.embeddings import DisabledEmbeddings, EmbeddingBackend, SentenceTransformerEmbeddings
from app.ml.imaging_hf import HFImagingModel
from app.ml.lab_anomaly import LabAnomalyModel
from app.ml.symptom_semantic import SymptomSemanticModel


@dataclass
class ModelBundle:
    settings: Settings
    embeddings: EmbeddingBackend
    symptom_semantic: SymptomSemanticModel
    lab_anomaly: LabAnomalyModel
    differential: DifferentialLogRegModel
    drug_resolver: DrugResolver
    imaging: HFImagingModel

    @property
    def enabled(self) -> bool:
        return self.settings.ml_enabled

    @classmethod
    def create(cls, settings: Settings) -> "ModelBundle":
        if settings.ml_enabled:
            embeddings: EmbeddingBackend = SentenceTransformerEmbeddings(
                settings.embedding_model, device=settings.ml_device
            )
        else:
            embeddings = DisabledEmbeddings()

        symptom = SymptomSemanticModel(
            embeddings,
            threshold=settings.symptom_ml_threshold,
            margin=settings.symptom_ml_margin,
            enabled=settings.ml_enabled,
        )
        lab_anomaly = LabAnomalyModel(
            artifact_dir=settings.ml_artifact_dir,
            contamination=settings.lab_anomaly_contamination,
            enabled=settings.ml_enabled,
        )
        differential = DifferentialLogRegModel(
            artifact_dir=settings.ml_artifact_dir,
            enabled=settings.ml_enabled,
            ml_weight=settings.differential_ml_weight,
        )
        resolver = DrugResolver(
            embeddings,
            fuzzy_threshold=settings.drug_fuzzy_threshold,
            embedding_threshold=settings.drug_embedding_threshold,
            enabled=settings.ml_enabled,
        )
        imaging = HFImagingModel(
            settings.imaging_model_id,
            device=settings.ml_device,
            threshold=settings.imaging_confidence_threshold,
            enabled=settings.ml_enabled,
        )
        return cls(
            settings=settings,
            embeddings=embeddings,
            symptom_semantic=symptom,
            lab_anomaly=lab_anomaly,
            differential=differential,
            drug_resolver=resolver,
            imaging=imaging,
        )

    def health(self) -> dict[str, dict[str, Any]]:
        models: dict[str, Any] = {
            "symptom_semantic": self.symptom_semantic,
            "lab_anomaly": self.lab_anomaly,
            "differential_reranker": self.differential,
            "drug_resolver": self.drug_resolver,
            "imaging": self.imaging,
        }
        report: dict[str, dict[str, Any]] = {}
        for key, model in models.items():
            info: ModelInfo = model.info
            report[key] = {
                "name": info.name,
                "version": info.version,
                "framework": info.framework,
                "enabled": info.enabled,
                "ready": bool(getattr(model, "ready", False)),
            }
        return report
