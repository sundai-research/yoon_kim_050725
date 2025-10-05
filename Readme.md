# VAST ai

# compute and ssh

we will be using Vast AI to run our experiments.

you need to share the public key with Aldo so he can add you to the machine. Make sure it's the default key of your ssh agent.

if you don't have a key, you can generate one with:
```shell
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "root@$(hostname)"
```

when Aldo adds you to the machine, you can connect with:
```shell
ssh -p 5142 root@180.94.16.204
```
NOTE: the port and ip will be given during the hack.

when connecting do these from the machine:

```shell
touch ~/.no_auto_tmux
```

create an ssh key in the machine to clone from github. (or just clone using https instead of ssh)
```shell
ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -C "root@$(hostname)"
```


# environment

install uv:

```shell
echo "[python] Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH
export PATH="/root/.local/bin:$PATH"
sed -i '106i export PATH="/root/.local/bin:$PATH"' ~/.bashrc && echo "Added PATH export before venv activation"
```

create venv:

```shell
uv venv -p 3.12 --managed-python --clear
source .venv/bin/activate
```

clone the hack repo:
```shell
git clone https://github.com/sundai-research/yoon_kim_050725.git
cd yoon_kim_050725
```

install dependencies:

```shell
uv pip install torch
uv pip install flash-attn --no-build-isolation
cd flash-linear-attention
uv pip install -e . #install in edit mode so you can change the code as needed
cd ..
```


# sftp
we can use sftp to transfer files to the machine while working locally, on save it will automatically upload the files to the machine.


in vscode, go to the extension market and install the sftp extension.
then press cmd+shift+p and type "sftp: Config" which will open the sftp.json file.

then paste something like this:

```json
{
    "name": "My Server",
    "host": "<ip shared with you>",
    "protocol": "sftp",
    "port": <port shared with you>,
    "username": "root",
    "privateKeyPath": "<path to your private key>", #you must share the public key with Aldo so he can add you to the machine
    "remotePath": "/workspace/sundai-hack",
    "uploadOnSave": true,
    "useTempFile": false,
    "openSsh": false
}
```
