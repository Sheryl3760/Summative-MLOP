import os
from locust import HttpUser, task, between
from PIL import Image
import io
import random


def generate_test_image():
    img = Image.fromarray(
        [[[ random.randint(0, 255) for _ in range(3)]
          for _ in range(64)]
         for _ in range(64)],
    )
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


class MalariaAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def predict(self):
        img_bytes = generate_test_image()
        self.client.post(
            "/predict",
            files={"file": ("test_cell.png", img_bytes, "image/png")}
        )

    @task(2)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")