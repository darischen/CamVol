from build_installer import _makensis_command, _installer_exe_path


def test_makensis_command_includes_version_define():
    cmd = _makensis_command("makensis.exe", "1.0.4")
    assert "/DVERSION=1.0.4" in cmd


def test_makensis_command_includes_proj_root_and_script():
    cmd = _makensis_command("makensis.exe", "1.0.4")
    assert cmd[0] == "makensis.exe"
    assert any(part.startswith("/DPROJ_ROOT=") for part in cmd)
    assert str(cmd[-1]).endswith("handvol-installer.nsi")


def test_installer_exe_path_is_versioned():
    assert _installer_exe_path("1.0.4").name == "HandVol-1.0.4-Installer.exe"
