#!/usr/bin/env bash
set -Eeuo pipefail

readonly SSH_PORT="${SSH_PORT:-23456}"

if ! [[ "${SSH_PORT}" =~ ^[0-9]+$ ]] || (( SSH_PORT < 1 || SSH_PORT > 65535 )); then
    echo "Invalid SSH_PORT: ${SSH_PORT}" >&2
    exit 1
fi

if [[ ! -x /usr/sbin/sshd ]]; then
    echo "OpenSSH server is not installed in this image." >&2
    exit 1
fi

install -d -m 0700 /root/.ssh

cat > /root/.ssh/authorized_keys_westlake <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPy3G/AjImbsD4XFRTeWoEtDw2lfrsj0AuCbokx28bkV hal@Haru
EOF

chmod 0600 /root/.ssh/authorized_keys_westlake

install -d -m 0755 /run/sshd
ssh-keygen -A

echo "Starting SSH server on container port ${SSH_PORT}"

exec /usr/sbin/sshd -D -e \
    -p "${SSH_PORT}" \
    -o PermitRootLogin=prohibit-password \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o PubkeyAuthentication=yes \
    -o AuthorizedKeysFile=/root/.ssh/authorized_keys_westlake