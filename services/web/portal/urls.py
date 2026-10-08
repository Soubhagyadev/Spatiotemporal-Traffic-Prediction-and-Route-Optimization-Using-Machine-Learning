"""URL routes for the traffic portal."""

from django.urls import path

from . import views

app_name = "portal"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("segments/", views.segments, name="segments"),
    path("segments/<int:link_id>/", views.segment_detail, name="segment_detail"),
    path("predict/", views.predict_view, name="predict"),
    path("model/", views.model_performance, name="model_performance"),
    path("about/", views.about, name="about"),
]
