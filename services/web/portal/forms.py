"""Forms for the portal.

The segment dropdown is populated from the API by the view and injected here so
the form stays free of HTTP calls.
"""

from __future__ import annotations

import datetime as dt

from django import forms

MODEL_CHOICES = [
    ("linear_regression", "Linear regression"),
    ("random_forest", "Random forest"),
]


class PredictionForm(forms.Form):
    """Validate a segment, a target hour, a model and a forecast horizon."""

    link_id = forms.ChoiceField(label="Segment", choices=[])
    hour_ts = forms.CharField(
        label="Target hour",
        max_length=32,
        help_text="Local hour, e.g. 2026-10-07 20:00.",
        widget=forms.TextInput(attrs={"placeholder": "2026-10-07 20:00", "size": 22}),
    )
    model = forms.ChoiceField(label="Model", choices=MODEL_CHOICES, initial="linear_regression")
    steps = forms.IntegerField(label="Steps", min_value=1, max_value=12, initial=6)

    def __init__(self, *args, link_choices=(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fields["link_id"].choices = list(link_choices)

    def clean_hour_ts(self) -> str:
        raw = (self.cleaned_data["hour_ts"] or "").strip().replace(" ", "T", 1)
        try:
            parsed = dt.datetime.fromisoformat(raw)
        except ValueError:
            raise forms.ValidationError(
                "Enter a valid timestamp, for example 2026-10-07 20:00."
            ) from None
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
