import argparse
import sys
import unittest
import requests

class TestDeploymentOnline(unittest.TestCase):
    url = "http://128.2.205.124:8082/recommend/14867"

    def test_deployment_online(self):
        if not self.url:
            self.fail("URL not provided for testing.")
        try:
            response = requests.get(self.url, timeout=5)
            self.assertEqual(response.status_code, 200,
                             f"Expected status code 200, but got {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.fail(f"Request to {self.url} failed with exception: {e}")

if __name__ == "__main__":
    unittest.main()
