"""Root URL configuration for the traffic portal."""

from django.urls import include, path

urlpatterns = [
    path("", include("portal.urls")),
]
