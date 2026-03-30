"""Test unitari per SettingsManager."""

import pytest
from config.settings import AppSettings, SettingsManager


@pytest.fixture
def tmp_settings(tmp_path):
    """Crea un SettingsManager con file temporaneo."""
    config_file = str(tmp_path / "test_config.json")
    return SettingsManager(config_file=config_file)


class TestSettingsGet:
    def test_get_default_value(self, tmp_settings):
        assert tmp_settings.get("debug_mode") is False

    def test_get_invalid_key_raises(self, tmp_settings):
        with pytest.raises(KeyError):
            tmp_settings.get("chiave_inesistente")


class TestSettingsSet:
    def test_set_valid_bool(self, tmp_settings):
        tmp_settings.set("debug_mode", True, save_to_disk=False)
        assert tmp_settings.get("debug_mode") is True

    def test_set_valid_float(self, tmp_settings):
        tmp_settings.set("fixed_zoom_percent", 60.0, save_to_disk=False)
        assert tmp_settings.get("fixed_zoom_percent") == 60.0

    def test_set_int_as_float_coercion(self, tmp_settings):
        """int deve essere accettato come float."""
        tmp_settings.set("fixed_zoom_percent", 50, save_to_disk=False)
        assert isinstance(tmp_settings.get("fixed_zoom_percent"), float)

    def test_set_invalid_key_raises(self, tmp_settings):
        with pytest.raises(KeyError):
            tmp_settings.set("non_esiste", 42, save_to_disk=False)

    def test_set_wrong_type_raises(self, tmp_settings):
        with pytest.raises(TypeError):
            tmp_settings.set("debug_mode", "stringa", save_to_disk=False)

    def test_set_increments_version(self, tmp_settings):
        v0 = tmp_settings.get_version()
        tmp_settings.set("debug_mode", True, save_to_disk=False)
        assert tmp_settings.get_version() == v0 + 1


class TestSettingsPersistence:
    def test_save_and_load_roundtrip(self, tmp_settings, tmp_path):
        tmp_settings.set("debug_mode", True, save_to_disk=True)
        # Crea un nuovo manager sullo stesso file
        reloaded = SettingsManager(config_file=str(tmp_path / "test_config.json"))
        assert reloaded.get("debug_mode") is True

    def test_load_missing_file_uses_defaults(self, tmp_path):
        config_file = str(tmp_path / "non_esiste.json")
        mgr = SettingsManager(config_file=config_file)
        assert mgr.get("debug_mode") == AppSettings().debug_mode


class TestSettingsReset:
    def test_reset_single_key(self, tmp_settings):
        tmp_settings.set("debug_mode", True, save_to_disk=False)
        tmp_settings.reset("debug_mode")
        assert tmp_settings.get("debug_mode") is False

    def test_reset_all(self, tmp_settings):
        tmp_settings.set("debug_mode", True, save_to_disk=False)
        tmp_settings.set("fixed_zoom_percent", 99.0, save_to_disk=False)
        tmp_settings.reset()
        assert tmp_settings.get("debug_mode") is False
        assert tmp_settings.get("fixed_zoom_percent") == AppSettings().fixed_zoom_percent

    def test_reset_invalid_key_raises(self, tmp_settings):
        with pytest.raises(KeyError):
            tmp_settings.reset("non_esiste")


class TestSettingsGetVersion:
    def test_initial_version(self, tmp_settings):
        assert tmp_settings.get_version() == 0

    def test_version_increments_on_set(self, tmp_settings):
        tmp_settings.set("debug_mode", True, save_to_disk=False)
        tmp_settings.set("debug_mode", False, save_to_disk=False)
        assert tmp_settings.get_version() == 2
