from django.test import TestCase
from django.urls import reverse
from users.models import User

class EnviarCorreoRecuperacionView(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="hola@gmail.com", password="hola")

    def test_enviar_correo_error(self):
        url = reverse("enviar_correo_recuperacion_view")
        response = self.client.post(url)
        self.assertEqual(response.status_code, 302)

    def test_enviar_correo(self):
        url = reverse("enviar_correo_recuperacion_view")
        response = self.client.post(url, {"correo":"hola@gmail.com"})
        self.assertTrue(self.client.session["resp"] is not None)