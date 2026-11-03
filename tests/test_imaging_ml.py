"""Tests for the chest X-ray model integration and imaging-agent envelope."""

from __future__ import annotations

import base64
import io

from PIL import Image

from app.agents.imaging.agent import run_imaging_agent
from app.agents.imaging.schemas import ImagingFinding, ImagingRequest
from app.ml.base import ModelInfo
from app.ml.imaging_hf import E_PNEUMONIA, HFImagingModel


def _png_base64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(120, 120, 120)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _pipeline(*scores: tuple[str, float]):
    return lambda image, top_k=2: [
        {"label": label, "score": score} for label, score in scores
    ]


class FakeImagingModel:
    name = "fake-xray"
    ready = True

    def __init__(self, findings: list[ImagingFinding]) -> None:
        self._findings = findings

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="fake-xray",
            version="fake-xray@test",
            framework="fake",
            enabled=True,
        )

    def predict_detailed(self, req: ImagingRequest):
        from app.agents.imaging.model import ImagingPrediction

        return ImagingPrediction(
            findings=self._findings,
            notes=["Fake model output for tests."],
            ran=True,
        )

    def predict(self, req: ImagingRequest) -> list[ImagingFinding]:
        return self._findings


def test_injected_model_enriches_the_agent_result():
    model = FakeImagingModel(
        [ImagingFinding(code=E_PNEUMONIA, label="Pneumonia pattern", confidence=0.91)]
    )
    result = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest"),
        backend="auto",
        model=model,
    )
    assert result.status == "ok"
    assert result.backend == "fake-xray"
    assert result.evidence_codes == [E_PNEUMONIA]
    assert result.findings[0].confidence == 0.91
    assert result.ml is not None and result.ml.ready is True
    assert "fake-xray@test" in result.model_version


def test_null_backend_wins_when_explicitly_requested():
    model = FakeImagingModel(
        [ImagingFinding(code=E_PNEUMONIA, label="Pneumonia pattern", confidence=0.91)]
    )
    result = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest"),
        backend="null",
        model=model,
    )
    assert result.status == "unavailable"
    assert result.findings == []
    assert result.backend == "null"


def test_input_hash_tracks_supplied_image_bytes():
    model = FakeImagingModel([])
    first = _png_base64()
    second = base64.b64encode(b"different image").decode("ascii")
    a = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest", image_base64=first),
        backend="auto",
        model=model,
    )
    b = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest", image_base64=first),
        backend="auto",
        model=model,
    )
    c = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest", image_base64=second),
        backend="auto",
        model=model,
    )
    assert a.input_hash == b.input_hash
    assert a.input_hash != c.input_hash


def test_hf_model_short_circuits_without_loading_weights():
    model = HFImagingModel(enabled=True)

    other_modality = model.predict_detailed(
        ImagingRequest(modality="ct", body_site="chest", image_base64=_png_base64())
    )
    assert other_modality.ran is False
    assert "modality" in other_modality.notes[0]

    no_image = model.predict_detailed(ImagingRequest(modality="xray", body_site="chest"))
    assert no_image.ran is False
    assert "No image supplied" in no_image.notes[0]

    wrong_site = model.predict_detailed(
        ImagingRequest(modality="xray", body_site="abdomen", image_base64=_png_base64())
    )
    assert wrong_site.ran is False
    assert "chest X-ray only" in wrong_site.notes[0]

    # None of those paths may trigger a (large) model download.
    assert model._pipeline is None
    assert model.ready is False


def test_hf_model_flags_pneumonia_above_threshold():
    model = HFImagingModel(enabled=True)
    model._pipeline = _pipeline(("PNEUMONIA", 0.91), ("NORMAL", 0.09))

    prediction = model.predict_detailed(
        ImagingRequest(modality="xray", body_site="chest", image_base64=_png_base64())
    )
    assert prediction.ran is True
    assert [finding.code for finding in prediction.findings] == [E_PNEUMONIA]
    assert prediction.findings[0].confidence == 0.91
    assert any("not a diagnosis" in note for note in prediction.notes)
    assert any("not clinically validated" in note for note in prediction.notes)


def test_hf_model_does_not_flag_below_threshold():
    model = HFImagingModel(enabled=True, threshold=0.55)
    model._pipeline = _pipeline(("PNEUMONIA", 0.40), ("NORMAL", 0.60))

    prediction = model.predict_detailed(
        ImagingRequest(modality="xray", body_site="chest", image_base64=_png_base64())
    )
    assert prediction.ran is True
    assert prediction.findings == []
    assert any("did not flag pneumonia" in note for note in prediction.notes)


def test_agent_turns_bad_image_bytes_into_a_typed_error():
    model = HFImagingModel(enabled=True)
    model._pipeline = _pipeline(("PNEUMONIA", 0.91), ("NORMAL", 0.09))

    result = run_imaging_agent(
        ImagingRequest(modality="xray", body_site="chest", image_base64="not-an-image"),
        backend="auto",
        model=model,
    )
    assert result.status == "error"
    assert result.findings == []
    assert result.errors
    assert any("could not be interpreted" in note for note in result.notes)
