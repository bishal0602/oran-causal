import subprocess

import pytest

from oran_causal.experiments.launch import hosts, kaggle, live, settings, vps


def test_missing_hosts_and_owner_raise_at_use_naming_the_variable():
    ssh = vps.Ssh("true")
    with pytest.raises(settings.MissingSetting, match="ORAN_CAUSAL_VPS_HOST"):
        ssh.script("echo hi")
    with pytest.raises(settings.MissingSetting, match="ORAN_CAUSAL_INSTABOX_HOST"):
        live.default_instabox().transport.script("echo hi")
    with pytest.raises(settings.MissingSetting, match="ORAN_CAUSAL_KAGGLE_OWNER"):
        _ = kaggle.KaggleClient().account


def test_defaults_without_settings():
    assert hosts.VPS_SHARED.base == settings.DEFAULT_VPS_BASE
    assert hosts.INSTABOX.base == settings.DEFAULT_INSTABOX_BASE
    assert hosts.KAGGLE_CPU.base is None
    assert kaggle.KaggleClient().cli == "kaggle"


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(settings.VPS_HOST, "ops@vps.example.org")
    monkeypatch.setenv(settings.INSTABOX_HOST, "box-alias")
    monkeypatch.setenv(settings.VPS_BASE, "/srv/oc")
    monkeypatch.setenv(settings.INSTABOX_BASE, "/home/u/oc")
    monkeypatch.setenv(settings.KAGGLE_OWNER, "someone")
    monkeypatch.setenv(settings.KAGGLE_CLI, "/opt/bin/kaggle")
    calls = []

    def run(argv, **_kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    vps.Ssh("true", run=run).script("echo hi")
    assert calls[-1][len(vps.SSH_OPTS) + 1] == "ops@vps.example.org"
    assert vps.Ssh("true", host_setting=settings.INSTABOX_HOST).destination == "box-alias"
    assert vps.Ssh("true", host="h").destination == "h"
    assert hosts.VPS_SHARED.base == "/srv/oc"
    assert live.default_ssh_vps().base == "/srv/oc"
    assert hosts.INSTABOX.base == "/home/u/oc"
    client = kaggle.KaggleClient()
    assert (client.account, client.cli) == ("someone", "/opt/bin/kaggle")
    assert kaggle.KaggleClient(executable="k", owner="me").cli == "k"
    assert kaggle.KaggleClient(owner="me").account == "me"
