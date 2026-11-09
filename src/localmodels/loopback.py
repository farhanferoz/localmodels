"""sparkrun's SSH logins to this machine, made to work the same for any user, attended or not.

sparkrun sends some of the steps of a container start and stop over ``ssh localhost`` even when it serves on
one machine (its local dispatch is opt-in per call, ``orchestration/ssh.py`` in v0.4.2). Those logins worked
only while the owner's laptop lent its keys through a forwarded agent, so a slurm job that ran unattended
failed (refresh job 2220, 2026-11-09). ``LoopbackLogin`` gives every user a login of their own for this:

1. a key without a passphrase under the user's state directory (``$LOCALMODELS_STATE_DIR/ssh/``), made on
   first use, and refused when it is not this user's own private file (a state directory shared by several
   users must not let one log in as another);
2. that key allowed in the account's ``~/.ssh/authorized_keys`` only for logins from this machine itself
   (``from="127.0.0.1,::1"``), so a copy taken elsewhere logs in nowhere; sparkrun is run with
   ``--hosts localhost``, whose logins come from those addresses;
3. a small ``ssh`` launcher beside the key, first on the PATH of the sparkrun calls that log in: it runs the
   real ``ssh`` with this key alone and no agent, ahead of the user's own ssh config (OpenSSH keeps the first
   value it reads, and the command line is read first), so neither an agent nor ``IdentitiesOnly`` or
   ``IdentityAgent`` in a user's config changes what a run does;
4. a check login through that same launcher, before any start relies on it.

Nothing of a user's (key, host names, paths) is in this repository; each user's key, launcher and line stay in
their own home. Every step is logged, the ones that change nothing included.
"""

from __future__ import annotations

import functools
import logging
import os
import pwd
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .engines import run_command

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .engines import Runner

log = logging.getLogger(__name__)

KEY_NAME: Final = "loopback_ed25519"
KEY_COMMENT: Final = "localmodels-loopback"
LOOPBACK_ONLY: Final = 'from="127.0.0.1,::1"'
LAUNCHER_DIRNAME: Final = "bin"
PRIVATE_DIR_MODE: Final = 0o700
PRIVATE_FILE_MODE: Final = 0o600
GROUP_OTHER_BITS: Final = 0o077
#: Every login through the launcher: this key alone, never an agent.
FORCED_OPTIONS: Final = ("IdentityAgent=none", "IdentitiesOnly=yes")
#: The check login adds: never a prompt; loopback's host key accepted on first sight, as sparkrun's own
#: logins (``BatchMode``) refuse an unknown one.
CHECK_OPTIONS: Final = ("BatchMode=yes", "StrictHostKeyChecking=accept-new", "ConnectTimeout=10")


class LoopbackLoginError(RuntimeError):
    """The login to this machine with localmodels' own key does not work; the message names the remedy."""


def _options(options: Iterable[str]) -> list[str]:
    return [arg for option in options for arg in ("-o", option)]


@dataclass(frozen=True)
class LoopbackLogin:
    """The user's own login for sparkrun's logins to this machine; ``ensure`` makes, allows and checks it."""

    key: Path
    authorized_keys: Path
    runner: Runner = run_command

    @classmethod
    def default(cls) -> LoopbackLogin:
        """The key under the state directory; ``authorized_keys`` in the account's home, which sshd reads."""
        from .settings import state_dir  # noqa: PLC0415 - settings is large and needs nothing from here

        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)  # not $HOME, which a wrapper may have changed
        return cls(key=state_dir() / "ssh" / KEY_NAME, authorized_keys=home / ".ssh" / "authorized_keys")

    @property
    def public(self) -> Path:
        """The key's public half."""
        return self.key.with_suffix(".pub")

    @property
    def launcher(self) -> Path:
        """The ``ssh`` that logs in with this key alone; its folder goes first on a call's PATH."""
        return self.key.parent / LAUNCHER_DIRNAME / "ssh"

    def ensure(self) -> None:
        """Make the key if missing, allow it for loopback logins, write the launcher, then log in once."""
        from .settings import _locked  # noqa: PLC0415 - settings is large and needs nothing from here

        try:
            with _locked(self.key):  # concurrent jobs of one user make the key and its line once
                self._make_key()
                self._allow_key()
                self._write_launcher()
        except (OSError, ValueError) as exc:  # unreadable, unwritable, not a key, ...
            msg = f"setting up the loopback login (key {self.key}, {self.authorized_keys}): {exc}"
            raise LoopbackLoginError(msg) from exc
        self._check()

    def env(self, base: Mapping[str, str]) -> dict[str, str]:
        """``base`` with the launcher first on PATH: each ``ssh`` of the call logs in with this key alone."""
        return {**base, "PATH": os.pathsep.join([str(self.launcher.parent), base.get("PATH", os.defpath)])}

    def _make_key(self) -> None:
        self.key.parent.chmod(PRIVATE_DIR_MODE)
        if not self.key.is_file():
            self._keygen(["-t", "ed25519", "-N", "", "-C", KEY_COMMENT, "-f", str(self.key)], "made key")
            return
        mine = self.key.stat()
        if mine.st_uid != os.geteuid() or mine.st_mode & GROUP_OTHER_BITS:
            msg = (
                f"{self.key} is not this user's own private file (owner uid {mine.st_uid}, mode "
                f"{mine.st_mode & 0o777:o}): use a state directory of your own ($LOCALMODELS_STATE_DIR) or "
                "remove that key"
            )
            raise LoopbackLoginError(msg)
        if not self.public.is_file():  # its public half was not copied with it
            done = self._keygen(["-y", "-f", str(self.key)], "rebuilt the public half of")
            self.public.write_text(done, encoding="utf-8")
            return
        log.info("loopback login: key %s present", self.key)

    def _keygen(self, args: list[str], what: str) -> str:
        done = self.runner(["ssh-keygen", "-q", *args])
        if done.returncode != 0:
            msg = f"ssh-keygen {' '.join(args)} exited {done.returncode}: {done.stderr.strip()}"
            raise LoopbackLoginError(msg)
        log.info("loopback login: %s %s", what, self.key)
        return done.stdout

    def _allow_key(self) -> None:
        fields = self.public.read_text(encoding="utf-8").split()
        if len(fields) < 2:  # noqa: PLR2004 - a public key line is a type and a blob, then a comment
            msg = f"{self.public} holds no public key"
            raise ValueError(msg)
        public, blob = " ".join(fields), fields[1]
        existing = (
            self.authorized_keys.read_bytes().decode("utf-8", "surrogateescape")
            if self.authorized_keys.is_file()
            else ""
        )
        live = (line.split() for line in existing.splitlines() if not line.lstrip().startswith("#"))
        if any(blob in words for words in live):
            log.info("loopback login: key already allowed in %s", self.authorized_keys)
            return
        self.authorized_keys.parent.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
        separator = "\n" if existing and not existing.endswith("\n") else ""
        with self.authorized_keys.open("a", encoding="utf-8") as out:
            out.write(f"{separator}{LOOPBACK_ONLY} {public}\n")
        self.authorized_keys.chmod(PRIVATE_FILE_MODE)
        log.info(
            "loopback login: allowed key %s in %s for logins from this machine only",
            self.key,
            self.authorized_keys,
        )

    def _write_launcher(self) -> None:
        elsewhere = os.pathsep.join(
            p for p in os.environ.get("PATH", os.defpath).split(os.pathsep) if Path(p) != self.launcher.parent
        )
        real = shutil.which("ssh", path=elsewhere)
        if real is None:
            msg = "no ssh client on PATH: sparkrun logs in to this machine over SSH; install OpenSSH's client"
            raise LoopbackLoginError(msg)
        forced = " ".join(_options(FORCED_OPTIONS))
        body = (
            "#!/bin/sh\n# localmodels' loopback login (loopback.py): this key alone, never an agent\n"
            f'exec {shlex.quote(real)} {forced} -i {shlex.quote(str(self.key))} "$@"\n'
        )
        if self.launcher.is_file() and self.launcher.read_text(encoding="utf-8") == body:
            log.info("loopback login: launcher %s present", self.launcher)
            return
        self.launcher.parent.mkdir(mode=PRIVATE_DIR_MODE, exist_ok=True)
        self.launcher.write_text(body, encoding="utf-8")
        self.launcher.chmod(PRIVATE_DIR_MODE)  # rwx for this user only
        log.info("loopback login: wrote launcher %s (runs %s)", self.launcher, real)

    def _check(self) -> None:
        done = self.runner([str(self.launcher), *_options(CHECK_OPTIONS), "localhost", "true"])
        if done.returncode != 0:
            msg = (
                f"sparkrun logs in to this machine over SSH, and a login with localmodels' key {self.key} "
                f"failed ({done.returncode}: {done.stderr.strip()}). Check that an SSH server runs here and "
                f"reads {self.authorized_keys} (sshd_config AuthorizedKeysFile, PubkeyAuthentication)"
            )
            raise LoopbackLoginError(msg)
        log.info("loopback login: checked, a login to this machine with key %s works", self.key)


@functools.cache
def loopback_login() -> LoopbackLogin:
    """This process's checked loopback login; a failure is not cached, so the next call tries again."""
    login = LoopbackLogin.default()
    login.ensure()
    return login
