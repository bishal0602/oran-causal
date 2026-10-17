"""The live backend of the dispatcher: real clients behind the `Backend` protocol."""

import dataclasses
from collections.abc import Sequence

from oran_causal.experiments.campaign.provenance import GitRunner
from oran_causal.experiments.launch import hosts, kaggle, plan, vps
from oran_causal.experiments.launch.runner import Dirs

LIST_HEADER_LINES = 2
ACTIVE_STATES = ("RUNNING", "QUEUED")


@dataclasses.dataclass(frozen=True)
class LiveBackend:
    """Launches and pulls through the real clients.

    Attributes:
        targets: Clients per platform and the local directories.
        kaggle_client: Kaggle client used for counting and status (also in `targets`).
    """

    targets: plan.Targets

    @property
    def dirs(self) -> Dirs:
        """Local directories."""
        return self.targets.dirs

    @property
    def git(self) -> GitRunner | None:
        """Git runner of the manifests."""
        return self.targets.git

    def _kaggle(self) -> kaggle.KaggleClient:
        if self.targets.kaggle_client is None:
            raise ValueError("no Kaggle client")
        return self.targets.kaggle_client

    def kaggle_counts(self, prefixes: Sequence[str]) -> tuple[int, int]:
        """Own sessions RUNNING or QUEUED, and how many carry one of `prefixes`."""
        client = self._kaggle()
        listing = client.run(
            "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "20", timeout=180
        )
        refs = [ln.split()[0] for ln in listing.splitlines()[LIST_HEADER_LINES:] if ln.strip()]
        running = mine = 0
        for ref in refs:
            state = client.run("kernels", "status", ref, timeout=120)
            if any(s in state for s in ACTIVE_STATES):
                running += 1
                mine += ref.split("/")[-1].startswith(tuple(prefixes))
        return running, mine

    def kaggle_state(self, job: str) -> str:
        """Kernel status text."""
        return self._kaggle().status(job)

    def kaggle_pull(self, job: str) -> None:
        """Downloads the kernel output."""
        kaggle.pull(job, self._kaggle(), self.dirs.runs)

    def ssh_pull(self, platform: plan.Platform, job: str) -> None:
        """Merges the outputs of `job` on the SSH host `platform`."""
        self.targets.ssh_host(platform).pull(job, self.dirs)

    def launch(self, request: plan.LaunchRequest) -> None:
        """Plans the request and launches it on its platform."""
        spec = plan.spec_of(request.spec)
        texts = plan.launcher_texts(request.root)
        session = plan.make_plan(request, spec, texts, self.git)
        plan.execute(session, self.targets)


def default_ssh_vps() -> vps.Vps:
    """The shared production VPS over SSH."""
    return vps.Vps(vps.Ssh(vps.remote_env(hosts.VPS_BASE)), hosts.VPS_SHARED)


def default_instabox() -> vps.Vps:
    """The instabox host over SSH, scopes in the login user's systemd instance."""
    return vps.Vps(
        vps.Ssh(vps.remote_env(hosts.INSTABOX_BASE), host=vps.INSTABOX_HOST), hosts.INSTABOX
    )
