import io
import random
from locust import HttpUser, task, between
from PIL import Image


def make_synthetic_image() -> io.BytesIO:
    """Generate a random cell-like image in memory — no file dependency."""
    r, g, b = random.randint(80, 200), random.randint(80, 200), random.randint(80, 200)
    img = Image.new("RGB", (64, 64), color=(r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class MalariaAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task(5)
    def predict(self):
        img_bytes = make_synthetic_image()
        with self.client.post(
            "/predict",
            files={"file": ("test_cell.png", img_bytes, "image/png")},
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif "label" not in response.text:
                response.failure("Invalid response format")

    @task(2)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")