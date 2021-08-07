from unittest import TestCase

from petrel.model.edet import load_edet


class TestEnet(TestCase):
    def test_enet(self):
        model = load_edet(config_name="tf_efficientdet_d0",
                          num_classes=1,
                          image_size=(512, 512))
        assert model.num_classes == 1
        assert model.image_size == (512, 512)