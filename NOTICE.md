# NOTICE

This repository is the open-source release of the **robot_vacuum** ("清扫大作战")
reinforcement learning competition project, built on the **Tencent Kaiwu
Platform** (腾讯开悟平台, KaiwuDRL).

## Origin and Templates

- The agent code structure (`code/agent_diy/`, `code/agent_ppo/`) follows the
  official Kaiwu platform starter templates. Files derived from the platform
  templates retain their original copyright headers (e.g.
  `Copyright © 1998 - 2026 Tencent. All Rights Reserved.`), and those headers
  remain the property of their respective owners.
- This repository contains the competition implementation authored by the
  kaiwuFinal team. Local modifications and additions are licensed under the
  MIT License (see `LICENSE`).

## Dependencies (not distributed here)

- **kaiwudrl**: the KaiwuDRL framework SDK is provided by the Tencent Kaiwu
  platform and is **not** part of this repository. It must be obtained through
  the platform (see the official docs at <https://tencentarena.com>).
- Docker images referenced by `train/.docker-compose.yaml` are published by
  the platform (`kaiwu-pub.tencentcloudcr.com/...`) and are pulled at runtime;
  they are not redistributed here.

## Runtime Requirements

Running this project requires a Tencent Kaiwu platform environment (platform
client / Linux training stack, `kaiwudrl` SDK, and a `license.dat` license
file). This repository alone cannot train an agent without those components.

## Third-Party Documentation

Official competition documentation referenced in this README is copyrighted
by Tencent and is not archived in this repository. Please refer to
<https://tencentarena.com> for the authoritative versions.

## Model Checkpoints

Model checkpoints (`.pkl`) are training artifacts and are not stored in this
repository. See the model policy section in `README.md`.
